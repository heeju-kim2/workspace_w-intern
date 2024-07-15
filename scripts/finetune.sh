model_name="meta-llama/Llama-2-7b-hf"


# pure bf16
# dtype="bfloat16"
# CUDA_VISIBLE_DEVICES=1 python src/finetuning.py \
#                         --model_name $model_name \
#                         --dtype $dtype \
#                         --use_anyprecision

# # mixed bf16
# dtype="bfloat16"
# CUDA_VISIBLE_DEVICES=1 python src/finetuning.py \
#                         --model_name $model_name \
#                         --dtype $dtype \
#                         --mixed_precision

# # pure fp32
# dtype="float32"
# CUDA_VISIBLE_DEVICES=1 python src/finetuning.py \
#                         --model_name $model_name \
#                         --dtype $dtype

# mixed fp8
dtype="float8"
CUDA_VISIBLE_DEVICES=1 python src/finetuning.py \
                        --model_name $model_name \
                        --dtype $dtype \
                        --mixed_precision