import os
import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator

from configs import eval_config as EVAL_CONFIG
from utils.config_utils import (
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)

from utils.dataset_utils import get_preprocessed_dataset

from utils.eval_utils import (
    get_inference,
    get_metrics, 
    
)


from utils.train_utils import (
    clear_gpu_cache,
    print_model_size,

)

from finetuning import set_seed, get_dataloader

"""
1. inference 
2. evaluation

"""

def main(args):
    eval_config = EVAL_CONFIG()
    update_config(eval_config, **vars(args))
    
    set_seed(eval_config.seed)

    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
            eval_config.model_name,
            load_in_8bit=True if eval_config.quantization else None,
            device_map=None if int(os.environ["WORLD_SIZE"]) > 1 else "auto",
            use_cache=True,
            torch_dtype=eval_config.dtype,
            attn_implementation="sdpa" if eval_config.use_fast_kernels else None,
            trust_remote_code=True,
        )
    
    if eval_config.use_peft:
        assert eval_config.peft_path is not None
        peft_config = PeftConfig.from_pretrained(eval_config.peft_path)
        model = PeftModel.from_pretrained(model, eval_config.peft_path)

    print_model_size(model, eval_config)
    
    tokenizer = AutoTokenizer.from_pretrained(eval_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    eval_dataloader = get_dataloader(eval_config, tokenizer, split="validation", do_eval=True)

    model = model.to(accelerator.device)
    
    results = get_inference(
        model=model,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        eval_config=eval_config, 
        accelerator=accelerator,)

    get_metrics(results, eval_config)
    
    ## Done ! 

def get_args():
    parser = argparse.ArgumentParser(description="eval for low precision finetuning")
    parser.add_argument(
        "--dtype", 
        type=str,
        default="None",
        choices=["None", "float16", "bfloat16", "float8", "float32"],
    )
    
    parser.add_argument(
        "--model_name", 
        default="meta-llama/Llama-2-7b-hf",
        required=True,
        help="enter hf model name"   
    )

    parser.add_argument(
        "--peft_method",
        type=str, 
        default="lora",
        help="set peft method",
        choices=["lora", "prompt", "adalora", "prefix", "boft"], 
    )

    parser.add_argument(
        "--peft_path",
        type=str, 
        default=None,
        required=True,
        help="set finetuned peft directory",
    )

    parser.add_argument(
        "--output_dir", 
        type=str,
        default="outputs",
        help="set model outputs dir",

    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="samsum_dataset",
        help="choose dataset to train",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()
    main(args)
