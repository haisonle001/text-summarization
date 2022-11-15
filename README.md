# PRIMERA for Vietnamese
We use the official code for PRIMERA: Pyramid-based Masked Sentence Pre-training for Multi-document Summarization from [PRIMERA](https://github.com/allenai/PRIMER).

PRIMERA is a pre-trained model for multi-document representation with focus on summarization that reduces the need for dataset-specific architectures and large amounts of fine-tuning labeled data. With extensive experiments on 6 multi-document summarization datasets from 3 different domains on the zero-shot, few-shot and full-supervised settings, PRIMER outperforms current state-of-the-art models on most of these settings with large margins on English datasets.

We experience PRIMERA by pretraining and fine-tuning on Vietnamese news datasets.
 
| PRIMERA | Rouge-1 F1 | Rouge-2 F1 | Rouge-L F1 | AVG.R F1 |
| --- | ----------- |----------- |----------- |----------- |
| VMDS | 73.6 | 44.1 | 41.0 | 52.9 |
| ViMS | 74.7 | 45.6 | 40.7 | 53.7 | 
| VLSP | 70.8 | 39.5 | 38.9 | 49.7 | 
| VLSP+ ViMs+ VMDS | 73.2 | 43.6 | 42.0 | 52.9 |

## Set up
1. Create new virtual environment by
```
conda create --name primer python=3.7
conda activate primer
conda install cudatoolkit=10.0
```
2. Install requirements to run the summarization scripts by 
```
pip install -r primer_requirements.txt
```

## Download fine-tuned model checkpoints
Pretrained on NewsCorpus and fine-tuned on ViMS: [here]()

Pretrained on NewsCorpus and fine-tuned on VMDS: [here]()

Pretrained on NewsCorpus and fine-tuned on ViMs + VMDS + VLSP: [here]()


## Summarization Scripts
You can use `script/primer_main.py` for pre-train/train/test/predict PRIMERA.

You can change these line in `script/primer_main.py` to your personal dir.

``` 
Line 35: rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir= /your-save-dir/)
Line 684: "--model_path", type=str, default=  /your-model-path/
Line 696: "--data_path", type=str, default= /your-data-path/
Line 749: "--primer_path", type=str, default= /your-primer-path/
```

```
Predict:
python script/primer_main.py --mode predict  --resume_ckpt /your-checkpoint-path/
```

```
Pre-train:
python script/primer_main.py --mode pretrain  --test_imediate  --data_path /your-data-path/ --dataset_name /your-dataset-name/
```

```
Train:
python script/primer_main.py --mode train --data_path  /your-data-path/ --resume_ckpt /your-checkpoint-path/
```

```
Test:
python script/primer_main.py --mode test --data_path  /your-data-path/ --resume_ckpt /your-checkpoint-path/
```



## Datasets
```
- Pre-train format: [{"src:[...], 'tgt':[]},{"src:[...], 'tgt':[]}, ...]
- Train/test format: [{'document':[...], 'summary':[...]}, {'document':[...], 'summary':[...]}, ...]
```


## Pre-training Data Generation
Install data requirements to run the data generation scripts (you should create new virtual environment):

```
pip install -r data_requirements.txt
```

You can use `utils/pretrain_preprocess.py` to generate pre-training data. 

```
python utils/pretrain_preprocess.py --input_dir=/your-input-path/ --output_dir=/your-output-path/
```
1. Generate data with scores and entities with `--mode compute_all_scores` 
2. Generate pre-training data with `--mode pretraining_data_with_score`:
    - Entity_Pyramid: `--strategy greedy_entity_pyramid --metric pyramid_rouge`