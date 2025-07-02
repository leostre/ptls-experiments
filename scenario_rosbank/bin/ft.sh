export CUBLAS_WORKSPACE_CONFIG=:4096:8

setup=test #quant_aware
model=cpc_tr
limit=1
# save_encoder=models/"${setup}"_"${model}".pth
# echo "Encoder will be saved as $save_encoder"
echo "Experiment: $setup & $model"
ckpt_dir=composition_results/"$setup"_"$model"/checkpoints
mkdir -p $ckpt_dir
touch $ckpt_dir/output.txt
echo "\nFC TARGET\n" >> $ckpt_dir/output.txt
python -m ptls.fedcore_compression.fc_fit_target --config-dir conf --config-name pl_fit_finetuning_"${model}" \
  +setup=$setup +limit_train_batches=$limit +limit_valid_batches=$limit \
  +model_name=$model \
  >> $ckpt_dir/output.txt
