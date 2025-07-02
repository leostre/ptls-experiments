# ulimit -n 32000
# export device=0
setup=test
model=mlmnsp
limit=100
# tuning_set=\
# "400 4 8 0.00008 512 96 2 64000
# 640 8 8 0.00005 512 64 2 64000"
hidden_size=128 
n_heads=4
n_layers=2
max_lr=0.00008
max_len=512
batch_size=96
accumulate_grad_batches=2
max_steps=64

ckpt_dir=composition_results/"$setup"_"$model"/checkpoints
mkdir -p $ckpt_dir
touch $ckpt_dir/output.txt
echo "\nFC TRAIN\n" >> $ckpt_dir/output.txt

python -m ptls.fedcore_compression.fc_train \
        +setup=$setup \
        +limit_train_batches=$limit \
        +limit_valid_batches=$limit \
        +model_name=$model \
        logger_name=${logger_name} \
        pl_module.hidden_size=${hidden_size} \
        pl_module.max_lr=${max_lr} \
        pl_module.seq_encoder.num_attention_heads=${n_heads} \
        pl_module.seq_encoder.num_hidden_layers=${n_layers} \
        data_module.train_data.max_len=${max_len} \
        data_module.valid_data.max_len=${max_len} \
        data_module.train_batch_size=${batch_size} \
        data_module.valid_batch_size=${batch_size} \
        trainer.accumulate_grad_batches=${accumulate_grad_batches} \
        trainer.max_steps=${max_steps} \
        model_path="models/mlmnsp__$logger_name.p" \
      --config-dir conf --config-name mlm_nsp_params \
      &>> $ckpt_dir/output.txt

# python -m ptls.fedcore_compression.fc_inference \
#         pl_module.hidden_size=${hidden_size} \
#         pl_module.seq_encoder.num_attention_heads=${n_heads} \
#         pl_module.seq_encoder.num_hidden_layers=${n_layers} \
#         inference.batch_size=1024 \
#         model_path="models/mlmnsp__$logger_name.p" \
#         embed_file_name="emb_mlmnsp_stat__${logger_name}" \
#       --config-dir conf --config-name mlm_nsp_params

# # Compare
# rm results/scenario_alpha_battle.txt
# rm -r embeddings_validation.work/
# python -m embeddings_validation \
#     --config-dir conf --config-name embeddings_validation_short +workers=10 +total_cpu_count=10 \
#     ++report_file="results/scenario_alpha_battle.txt" 
    

