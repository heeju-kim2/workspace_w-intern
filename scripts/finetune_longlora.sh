#if running on multi-gpu machine
# export LOCAL_RANK=0
# export RANK=0
export WORLD_SIZE=1

model_name="meta-llama/Llama-2-7b-chat-hf"
peft_method="longlora"
output_dir="results"
dataset="redpajama_dataset" #"alpaca_long_dataset" #"redpajama_dataset" #"samsum_dataset"
model_max_length=8192

# # pure bf16
dtype="bfloat16"
output_dir="$output_dir/${dataset}/${peft_method}/bf16_only"

CUDA_VISIBLE_DEVICES=1 python src/finetuning.py \
                        --model_name $model_name \
                        --dtype $dtype \
                        --use_anyprecision \
                        --peft_method $peft_method \
                        --output_dir $output_dir \
                        --dataset $dataset \
                        --model_max_length $model_max_length \
                        --debug 

# # mixed bf16
dtype="bfloat16"
output_dir="$output_dir/${dataset}/${peft_method}/bf16_mixed"
CUDA_VISIBLE_DEVICES=1 python src/finetuning.py \
                        --model_name $model_name \
                        --dtype $dtype \
                        --mixed_precision \
                        --peft_method $peft_method \
                        --output_dir $output_dir \
                        --dataset $dataset \
                        --model_max_length $model_max_length \
                        --debug

# # pure fp32
dtype="float32"
output_dir="./$output_dir/$dataset/$peft_method/$dtype"
output_dir="output_tests"
#single-chip
CUDA_VISIBLE_DEVICES=1 python src/finetuning.py \
                        --model_name $model_name \
                        --dtype $dtype \
                        --peft_method $peft_method \
                        --output_dir $output_dir \
                        --dataset $dataset \
                        --model_max_length $model_max_length \
                        --debug 

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