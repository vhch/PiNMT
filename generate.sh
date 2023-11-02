DATAPATH=./download_prepare/data/
STPATH=${DATAPATH}de-en-databin/
MODELPATH=./models/one-way/ 
PRE_SRC=jhu-clsp/bibert-ende
PRE=./download_prepare/8k-vocab-models
CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
--beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.en.txt
