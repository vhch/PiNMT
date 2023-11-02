## vocab 8k One-Directional Training modify
TEXT=./download_prepare/data/
SAVE_DIR=./models/oneway/

CUDA_VISIBLE_DEVICES=0 fairseq-train ${TEXT}de-en-databin/ --arch transformer_iwslt_de_en --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 1.0 --lr 0.0004 --lr-scheduler inverse_sqrt --lr_mul 0.01 --warmup-updates 4000 --dropout 0.3 \
--weight-decay 0.00002 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2048 --update-freq 16 --alpha 500 \
--attention-dropout 0.1 --activation-dropout 0.1 --max-epoch 150 --save-dir ${SAVE_DIR}  --encoder-embed-dim 512 --decoder-embed-dim 512 \
--no-epoch-checkpoints --save-interval 1 --pretrained_model jhu-clsp/bibert-ende --use_drop_embedding 12
