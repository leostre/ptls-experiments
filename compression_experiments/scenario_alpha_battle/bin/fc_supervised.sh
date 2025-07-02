export CUBLAS_WORKSPACE_CONFIG=:4096:8

ckpt_path=composition_results
max_batch=1
setup=test
echo "Encoder Training is up"
python -m fc_train \
  pl_module.seq_encoder.hidden_size=1024 \
  +pl_module.loss='{_target_: ptls.frames.coles.losses.MarginLoss, margin: 0.2, beta: 0.4}' \
  +pl_module.loss.pair_selector='{_target_: ptls.frames.coles.sampling_strategies.HardNegativePairSelector, neg_count: 5}' \
  ~pl_module.loss.sampling_strategy \
  model_path="models/mles_model_for_finetuning.p" \
  --config-dir conf --config-name mles_params
echo "Encoder Training is over"

for scenario in raw composite_1 composite_2
do
  echo "$scenario Finetuning is up"
  python -m fc_fit_target_pl\
    pretrained_encoder_path="models/mles_model_for_finetuning.p"\
    +setup=test\
    +limit_train_batches=$max_batch\
    +limit_valid_batches=$max_batch\
  --config-dir conf --config-name pl_fit_finetuning_mles\
  >> $ckpt_path/$type_/sdout.txt
  echo "$scenario Finetuning is over"
done
echo "Finetuning ended"
