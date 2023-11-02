DATAPATH=./download_prepare/data_mixed_ft/
STPATH=${DATAPATH}de-en-databin/
MODELPATH=./models/dual-ft/ 
PRE_SRC=jhu-clsp/bibert-ende
PRE=./download_prepare/12k-vocab-models/
CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
--beam 5 --lenpen 1.0 --remove-bpe --vocab_file=${STPATH}/dict.en.txt
