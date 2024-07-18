export WORLD_SIZE=1
gpu_id=0
dtype="float16"
model_name="meta-llama/Llama-2-7b-chat-hf"
peft_path="/pvc/data-hjkim/low-precision-training/outputs/float32/epoch_0"
dataset="samsum_dataset"

CUDA_VISIBLE_DEVICES=$gpu_id python src/eval.py \
                                --dtype $dtype \
                                --model_name $model_name \
                                --peft_path $peft_path \
                                --output_dir $peft_path \
                                --dataset $dataset 
