ckpt_path=composition_results
max_batch=10000
need_encoder_pretrain=true

if $need_encoder_pretrain
then
  echo "Training phase is up"
  python -m fc_train \
    pl_module.seq_encoder.hidden_size=1024 \
    +pl_module.loss='{_target_: ptls.frames.coles.losses.MarginLoss, margin: 0.2, beta: 0.4}' \
    +pl_module.loss.pair_selector='{_target_: ptls.frames.coles.sampling_strategies.HardNegativePairSelector, neg_count: 5}' \
    ~pl_module.loss.sampling_strategy \
    model_path="models/mles_model_for_finetuning.p" \
    --config-dir conf --config-name mles_params &&\
  echo "Encoder Training is over" 
fi

for setup in test #raw composite_1 composite_2
do
  echo "Embeddings generation"
  rm -rf $ckpt_path/$setup/scores
  mkdir -p $ckpt_path/$setup/scores 
  python -m fc_inference --config-dir conf --config-name mles_params\
    ++inference.output="$ckpt_path/$setup/scores"\
    +inference_model="$ckpt_path/$setup/checkpoints"\
    +n_batches_computational=2\
    +limit_train_batches=$max_batch\
    +limit_valid_batches=$max_batch &&\
  echo "Embeddings for $setup generated"
  
  python -m fc_emb_eval --config-dir conf --config-name fc_embeddings_validation_short \
    +setup=$setup +emb_path="$ckpt_path"
done
echo "Unsuper phase is done"
