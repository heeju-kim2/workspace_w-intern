#if running on multi-gpu machine
# export LOCAL_RANK=0
# export RANK=0
export WORLD_SIZE=1

model_name="../../models/Llama-2-7b-chat-hf"
peft_method="lora"

# # pure bf16
# dtype="bfloat16"
# CUDA_VISIBLE_DEVICES=0 python src/finetuning.py \
#                         --model_name $model_name \
#                         --dtype $dtype \
#                         --use_anyprecision \
#                         --peft_method $peft_method

# # mixed bf16
# dtype="bfloat16"
# CUDA_VISIBLE_DEVICES=0 python src/finetuning.py \
#                         --model_name $model_name \
#                         --dtype $dtype \
#                         --mixed_precision \
#                         --peft_method $peft_method

# # pure fp32
dtype="float32"
#single-chip
CUDA_VISIBLE_DEVICES=0 python ../src/finetuning.py \
                        --model_name $model_name \
                        --dtype $dtype \
                        --peft_method $peft_method \


# multi-chip (In debugging)
# torchrun \
#     --nproc_per_node 2 src/finetuning.py \
#                         --model_name $model_name \
#                         --dtype $dtype



# mixed fp8
# dtype="float8"
# CUDA_VISIBLE_DEVICES=1 python src/finetuning.py \
#                         --model_name $model_name \
#                         --dtype $dtype \
#                         --mixed_precision