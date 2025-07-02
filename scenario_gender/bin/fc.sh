setup=quant_aware
model=nsp
save_encoder=models/"${setup}"_"${model}".pth
echo "Encoder will be saved as $save_encoder"
ckpt_dir=composition_results/$setup/checkpoints
mkdir -p $ckpt_dir
touch $ckpt_dir/output.txt
python -m ptls.fedcore_compression.fc_train --config-dir conf --config-name "${model}"_params \
  +setup=$setup +save_encoder=$save_encoder\
  >> $ckpt_dir/output.txt
