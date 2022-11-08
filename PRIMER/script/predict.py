from primer_main_for_testing import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ########################
    # General
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use")
    parser.add_argument(
        "--accelerator", default='gpu', type=str, help="Type of accelerator"
    )
    parser.add_argument(
        "--mode", default="test", choices=["pretrain", "train", "test","test_single_data"]
    )
    parser.add_argument(
        "--debug_mode", action="store_true", help="set true if to debug"
    )
    parser.add_argument(
        "--compute_rouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
        default=False,
    )
    parser.add_argument(
        "--saveRouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
    )

    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
    ####
    parser.add_argument(
        "--model_path", type=str, default="/home/redboxsa_ml/anh/PRIMER/longformer_summ_multinews"
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--saveTopK", default=3, type=int)
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        help="Path of a checkpoint to resume from",
        default=None,
    )

    ####
    parser.add_argument("--data_path", type=str, default=" /home/redboxsa_ml/sonlh/vlsp-train-validation/")
    parser.add_argument("--dataset_name", type=str, default="wcep")
    parser.add_argument("--tokenizer", type=str, default="facebook/bart-base")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to use for dataloader",
    )

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_tgt", default=1024, type=int)
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--join_method", type=str, default="concat_start_wdoc_global")
    parser.add_argument(
        "--attention_dropout", type=float, default=0.1, help="attention dropout"
    )
    parser.add_argument(
        "--attention_mode",
        type=str,
        default="sliding_chunks",
        help="Longformer attention mode",
    )
    parser.add_argument(
        "--attention_window", type=int, default=512, help="Attention window"
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument(
        "--adafactor", action="store_true", help="Use adafactor optimizer"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="default is fp16. Use --fp32 to switch to fp32",
    )
    parser.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=0,
        help="seed for random sampling, useful for few shot learning",
    )
        ########################
    # For training
    ####
    parser.add_argument(
        "--primer_path",
        type=str,
        default="/home/redboxsa_ml/sonlh/PRIMERA_model/",
    )
    parser.add_argument(
        "--limit_valid_batches",
        type=int,
        default=None,
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--report_steps", type=int, default=50, help="Number of report steps"
    )
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help="Number of steps to evaluate",
    )
    parser.add_argument(
        "--accum_data_per_step", type=int, default=16, help="Number of data per step"
    )
    parser.add_argument(
        "--total_steps", type=int, default=50000, help="Number of steps to train"
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=-1,
        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use",
    )
    parser.add_argument(
        "--remove_masks",
        action="store_true",
        help="remove all the masks in pretraining",
    )
    parser.add_argument(
        "--fix_lr",
        action="store_true",
        help="use fix learning rate",
    )
    parser.add_argument(
        "--test_imediate",
        action="store_true",
        help="test on the best checkpoint",
    )
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="whether this is a run for few shot learning",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=5000,
        help="Number of steps to evaluate in the pre-training stage.",
    )
    ########################
    # For testing
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="Number of batches to test in the test mode.",
    )
    parser.add_argument("--beam_size", type=int, default=3, help="size of beam search")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="length penalty, <1 encourage shorter message and >1 encourage longer messages",
    )

    parser.add_argument(
        "--mask_num",
        type=int,
        default=0,
        help="Number of masks in the input of summarization data",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=-1,
        help="batch size for test, used in few shot evaluation.",
    )
    parser.add_argument(
        "--applyTriblck",
        action="store_true",
        help="whether apply trigram block in the evaluation phase",
    )

    args = parser.parse_args()  # Get pad token id
    ####################

    args.acc_batch = args.accum_data_per_step // args.batch_size
    args.data_path = os.path.join(args.data_path, args.dataset_name)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # print(args)
    with open(
        os.path.join(
            args.model_path, "args_%s_%s.json" % (args.mode, args.dataset_name)
        ),
        "w",
    ) as f:
        json.dump(args.__dict__, f, indent=2)

    # predict(args)

    if not args.resume_ckpt:
        print("Resume checkpoint is not provided.")
    else:
        print("Loading PRIMER model ...")
        model = PRIMERSummarizerLN.load_from_checkpoint(args.resume_ckpt, args=args)

        docs = input("Enter documents separated by a <doc-sep>:\n")
        docs = docs.split('<doc-sep>')
        docs = [' '.join(rdrsegmenter.word_segment(doc)) for doc in docs]
        print(docs)
        dataset = {"document": docs, "summary": "_"}

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
            limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1
        )
        trainer.test(model, test_dataloader)
