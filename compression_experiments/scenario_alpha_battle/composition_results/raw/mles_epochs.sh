# mles
checkpoints_folder="/ptls-experiments/compression_experiments/alpha_battle/raw/*.pth"
output_file_prefix="mles__"
conf_file="mles_params"
batch_size=800

for model_file in $(ls -vr $checkpoints_folder)
do
  echo "--------: $model_file"
  model_name=$(basename "$model_file")
#  model_file=${model_file//"="/"\="}

  # echo $model_name  # epoch=9-step=44889.ckpt
  epoch_num=$(echo $model_name | cut -f 1 -d "-")
  epoch_num=$(echo $epoch_num | cut -f 2 -d "=")
  epoch_num=$(printf %03d $epoch_num)
  # echo $epoch_num

  output_file=$(echo $output_file_prefix$epoch_num)

#  if [ $epoch_num = "005" ]; then
  if [ -f "data/$output_file.pickle" ]; then
    echo "--------: $output_file exists"
  else
      echo "--------: Run inference for $output_file"
      python -m ptls.pl_inference model_path=\""${model_file}"\" embed_file_name="${output_file}" inference.batch_size=${batch_size} --config-dir conf --config-name "${conf_file}"
  fi
#  fi
done

rm results/epochs_mles.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=20 \
    +report_file="results/epochs_mles.txt" \
    +auto_features=["data/mles_orig5__???.pickle"]
