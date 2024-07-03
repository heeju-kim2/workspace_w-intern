#if running on multi-gpu machine
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1

model_name="Llama-2-7b-chat-hf"
# model_name="Meta-Llama-3-8B-Instruct"
model_path="../models/${model_name}"
num_epochs=5
batch_size_training=1
gradient_accumulation_steps=4
batch_size=$(($batch_size_training*$gradient_accumulation_steps))
weight_decay=0.0
lr=1e-4


# # bf16 only
# output_dir="../samsum_outputs/${model_name}/bf16/ep_${num_epochs}_lr_${lr}_bs_${batch_size}_wd_${weight_decay}"
# mkdir -p $output_dir
# torchrun --nnodes 1 --nproc_per_node 1 \
#         llama-recipes/src/finetuning.py \
#         --enable_fsdp --use_peft --peft_method lora --dataset samsum_dataset \
#         --lr $lr --num_epochs $num_epochs --batch_size_training $batch_size_training \
#         --gradient_accumulation_steps $gradient_accumulation_steps --weight_decay $weight_decay \
#         --dist_checkpoint_root_folder $output_dir \
#         --model_name $model_path \
#         --output_dir $output_dir \
#         --pure_bf16 --optimizer anyprecision --mixed_precision False --use_fp16 False \
#         --use_wandb
#         # --max_train_step 4 --max_eval_step 4 \

# model param bfloat 16
# hidden rep bfloat 16
# logit bfloat 16
# logit float 32로 casting
# loss float 32로 계산


# # bf16 + fp32 mixed precision
# output_dir="../samsum_outputs/${model_name}/mixed/ep_${num_epochs}_lr_${lr}_bs_${batch_size}_wd_${weight_decay}"
# mkdir -p $output_dir
# torchrun --nnodes 1 --nproc_per_node 1 \
#         llama-recipes/src/finetuning.py \
#         --enable_fsdp --use_peft --peft_method lora --dataset samsum_dataset \
#         --lr $lr --num_epochs $num_epochs --batch_size_training $batch_size_training \
#         --gradient_accumulation_steps $gradient_accumulation_steps --weight_decay $weight_decay \
#         --dist_checkpoint_root_folder $output_dir \
#         --model_name $model_path \
#         --output_dir $output_dir \
#         --mixed_precision True --use_fp16 False \
#         --use_wandb
#         # --max_train_step 2 --max_eval_step 2 \
#         # --use_wandb


# param bfloat 16으로 casting
# hidden rep bfloat 16으로 계산
# logit bfloat 16
# logit float 32로 casting
# loss float 32로 계산


# fp32 only
output_dir="../samsum_outputs/${model_name}/fp32/ep_${num_epochs}_lr_${lr}_bs_${batch_size}_wd_${weight_decay}"
mkdir -p $output_dir
torchrun --nnodes 1 --nproc_per_node 1 \
        llama-recipes/src/finetuning.py \
        --enable_fsdp --use_peft --peft_method lora --dataset samsum_dataset \
        --lr $lr --num_epochs $num_epochs --batch_size_training $batch_size_training \
        --gradient_accumulation_steps $gradient_accumulation_steps --weight_decay $weight_decay \
        --dist_checkpoint_root_folder $output_dir \
        --mixed_precision False --use_fp16 False \
        --model_name $model_path \
        --output_dir $output_dir \
        --use_wandb \
        # --max_train_step 2 --max_eval_step 2 \

# param, hidden rep, logit, loss 모두 float 32