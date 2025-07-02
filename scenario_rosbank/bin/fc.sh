setup=test #quant_aware
model=mles
save_encoder=models/"${setup}"_"${model}".pth
echo "Encoder will be saved as $save_encoder"
ckpt_dir=composition_results/"$setup"_"$model"/checkpoints
mkdir -p $ckpt_dir
touch $ckpt_dir/output.txt
echo "\nFC TRAIN" >> $ckpt_dir/output.txt
python -m ptls.fedcore_compression.fc_train --config-dir conf --config-name "${model}"_params \
  +setup=$setup +save_encoder=$save_encoder\
  >> $ckpt_dir/output.txt


# python -m ptls.pl_train_module --config-dir conf --config-name cpc_params
# python -m ptls.pl_inference --config-dir conf --config-name cpc_params