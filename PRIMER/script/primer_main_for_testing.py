from pytorch_lightning.accelerators import accelerator
import torch
import os
import argparse
from torch.utils.data import DataLoader
import numpy as np
from transformers import (
    AutoTokenizer,
    LEDTokenizer,
    LEDConfig,
    LEDForConditionalGeneration,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from transformers import Adafactor
from longformer.sliding_chunks import pad_to_window_size
from longformer import LongformerEncoderDecoderForConditionalGeneration
from longformer import LongformerEncoderDecoderConfig
import pandas as pd
import pdb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from datasets import load_dataset, load_metric
from dataloader import (
    get_dataloader_summ,
    get_dataloader_pretrain,
    get_dataloader_summiter,
)
import json
from pathlib import Path
from rouge import Rouge
import py_vncorenlp
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="C:/Work/NLP/PRIMER/data-env")



def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class PRIMERSummarizerLN(pl.LightningModule):
    def __init__(self, args):
        super(PRIMERSummarizerLN, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
        self.pad_token_id = self.tokenizer.pad_token_id
        config = LongformerEncoderDecoderConfig.from_pretrained(args.primer_path)
        config.gradient_checkpointing = args.grad_ckpt
        self.model = LEDForConditionalGeneration(config)

        self.use_ddp = args.accelerator == "ddp"
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        if args.mode=='pretrain' or args.mode=='test' or args.mode =='train':
            # The special token is added after each document in the pre-processing step.
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<doc-sep>"]}
            )
            self.docsep_token_id = self.tokenizer.additional_special_tokens_ids[0]
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _prepare_input(self, input_ids):
        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=input_ids.device
        )
        attention_mask[input_ids == self.pad_token_id] = 0
        if isinstance(self.model, LongformerEncoderDecoderForConditionalGeneration):
            # global attention on one token for all model params to be used,
            # which is important for gradient checkpointing to work
            attention_mask[:, 0] = 2

            if self.args.join_method == "concat_start_wdoc_global":
                attention_mask[input_ids == self.docsep_token_id] = 2

            if self.args.attention_mode == "sliding_chunks":
                half_padding_mod = self.model.config.attention_window[0]
            elif self.args.attention_mode == "sliding_chunks_no_overlap":
                half_padding_mod = self.model.config.attention_window[0] / 2
            else:
                raise NotImplementedError

            input_ids, attention_mask = pad_to_window_size(
                # ideally, should be moved inside the LongformerModel
                input_ids,
                attention_mask,
                half_padding_mod,
                self.pad_token_id,
            )
        return input_ids, attention_mask

    def forward(self, input_ids, output_ids):

        input_ids, attention_mask = self._prepare_input(input_ids)
        decoder_input_ids = output_ids[:, :-1]
        decoder_attention_mask = decoder_input_ids != self.pad_token_id
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        )
        lm_logits = outputs[0]
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        return lm_logits

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def shared_step(self, input_ids, output_ids):
        lm_logits = self.forward(input_ids, output_ids)
        labels = output_ids[:, 1:].clone()
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.pad_token_id,
            )
        return loss


    def generate_predict(self, input_ids, gold_str):
        input_ids, attention_mask = self._prepare_input(input_ids)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            length_penalty=self.args.length_penalty,
            no_repeat_ngram_size=3 ,
        )
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
        print(generated_str)

        output_dir = os.path.join(
            self.args.model_path,
            "generated_folder"    
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        idx = len(os.listdir(output_dir))

        for ref, pred in zip(gold_str, generated_str):
            with open(os.path.join(output_dir, "prediction.txt"), "w") as of:
                of.write(pred)
            with open(os.path.join(output_dir, "prediction.jsonl_5" ), "w", encoding="utf-8") as fo:
                json.dump(pred, fo, ensure_ascii=False, indent=4)
            idx += 1
        return []

    def prediction_step(self, batch, batch_idx):
        input_ids, output_ids, tgt = batch

        generate = self.generate_predict(input_ids, tgt)
        return True

    def test_step(self, batch, batch_idx):
        return self.prediction_step(batch, batch_idx)

def predict(args):
    if args.resume_ckpt:
        model = PRIMERSummarizerLN.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = PRIMERSummarizerLN(args)
    inp= input("Enter document: ")
    print(inp)
    inp=' '.join(rdrsegmenter.word_segment(inp))
    print(inp)
    test_dict={}
    if len(inp.split('<doc-sep>'))==1:
        test_dict['document']=[inp]
    else:
        test_dict['document']=inp.split('<doc-sep>')
    test_dict['summary']='a'
    torch.save([test_dict],'data.pt')
    dataset = torch.load('data.pt')
    print(dataset)
    
    test_dataloader = get_dataloader_summ(
        args, dataset, model.tokenizer, "test", args.num_workers, False
    )
    args.compute_rouge = True
    # initialize trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.acc_batch,
        replace_sampler_ddp=False,
        log_every_n_steps=5,
        checkpoint_callback=True,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        precision=32 if args.fp32 else 16,
        accelerator=args.accelerator,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0,
    )
    trainer.test(model, test_dataloader)

