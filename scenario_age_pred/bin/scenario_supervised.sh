# Supervised learning scenario for parameters specified below

PYTORCH_CUDA_ALLOC_CONF=""
# Some architectural setup
model=mles
pretrained_encoder_path=""
setup=composite_1

# Additional settings
max_batch_train=1
max_batch_val=1
need_fedot_pretrain=false
ckpt_dir=checkpoints
embeddings_dir=embeddings
n_batches_computational=1
distributed_compression=false

train_log=$ckpt_dir/"$(echo $model)"_train_log.txt
mkdir $ckpt_dir
touch $train_log

# Encoder fitting if pretrained is not found
# Note the setup is 'raw' to avoid overfitting the following steps.
# if [$pretrained_encoder_path -eq ""]; then
#     python -m ptls.fedcore_compression.fc_train \
#     pl_module.seq_encoder.hidden_size=256 \
#     +distributed_compression=$distributed_compression \
#     +save_encoder=models/"$(echo $model)"_for_finetuning.p \
#     +need_evo_opt=false\
#     +setup=raw \
#     --config-dir conf --config-name "$(echo $model)"_params \
#     >> $train_log
# fi

# # Supervised finetuning & composite compression
python -m ptls.fedcore_compression.fc_fit_target \
  --config-dir conf --config-name pl_fit_finetuning_"$(echo $model)" \
  +setup=$setup \
  +limit_train_batches=$max_batch_train \
  +limit_valid_batches=$max_batch_val \
  +need_fedot_pretrain=$need_fedot_pretrain\
  +distributed_compression=$distributed_compression \
  +need_evo_opt=false \
  >> $train_log

# # Embeddings generation and computational efficacy estimation 
# # (to turn it on the n_batches_computational > 0 should be specified)
# mkdir $embeddings_dir
# python -m ptls.fedcore_compression.fc_inference --config-dir conf --config-name "$(echo $model)"_params\
#   +inference.output=$embeddings_dir \
#   +inference_model=$ckpt_dir \
#   +limit_predict_batches=$max_batch_train \
#   +n_batches_computational="${$n_batches_computational:- 0}"
