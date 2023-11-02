# Integrating Pre-trained Language Model into Neural Machine Translation

This is the repository for paper "[Integrating Pre-trained Language Model into Neural Machine Translation](https://arxiv.org/abs/2310.19680)".
```
@misc{hwang2023integrating,
      title={Integrating Pre-trained Language Model into Neural Machine Translation},
      author={Soon-Jae Hwang and Chang-Sung Jeong},
      year={2023},
      eprint={2310.19680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Requirements
```
conda create -n pinmt python=3.7
conda activate pinmt
```
* [transformers](https://github.com/huggingface/transformers) >= 4.4.2
  ```
  pip install transformers
  ```
* Install our fairseq repo
  ```
  pip install --editable ./
  ```
* [hydra](https://github.com/facebookresearch/hydra) = 1.0.3
  ```
  pip install hydra-core==1.0.3
  ```

## Reproduction
### Preprocessing
Download and prepare IWSLT'14 dataset (if you meet warnings like `file config.json not found`, please feel safe to ignore it):
```
cd download_prepare
bash download_and_prepare_data.sh
```

After download and preprocessing, three preprocessed data bin will be shown in `download_prepare` folder:
* `data`: de->en preprocessed data for ordinary one-way translation
* `data_mixed`: bidirectional translation data
* `data_mixed_ft`: after bidirectional training, fine-tuning on unidirectional translation data

### Training
Train a model for one-way translation. Note that passing field `--use_drop_embedding` to consider number of PLM layers. Training with less GPUs should increase `--update-freq`, e.g., `update-freq=8` for 2 GPUs and `update-freq=4` for 4 GPUs.
```
bash train.sh
```

Train a model for bidirectional training and further unidirectional fine-tuning:
```
bash train-dual.sh
```

### Evaluation
Translation for one-way model:
```
bash generate.sh
```
Translation for bidirectional model:
```
bash generate-dual.sh
```

The BLEU score will be printed out in the final output after running `generate.sh`.

