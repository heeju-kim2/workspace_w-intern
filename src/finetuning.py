import os

import dataclasses
import fire
import random
import numpy as np
import transformers 
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training
from copy import deepcopy
import math

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from data.concatenator import ConcatDataset

from configs import train_config as TRAIN_CONFIG
from policies import AnyPrecisionAdamW

from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)

from utils.dataset_utils import get_preprocessed_dataset
from utils.train_utils import (
    train,
    clear_gpu_cache,
    print_model_size,  
)

from longlora.llama_attn_replace import replace_llama_attn
from longlora.gptneox_attn_replace import replace_gpt_neox_attn

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import is_xpu_available, FP8RecipeKwargs
import argparse

MIXED_PSN_DTYPE = { "bfloat16" : "bf16", "float16":"fp16", "float8": "fp8", "None": "no"}
ANY_PSN_DTYPE = {"bfloat16" : torch.bfloat16, "float16": torch.float16, "float8": torch.float8_e5m2, "float32": torch.float32}


"""
use Accelerate when PEFT + low precision finetuning


The script can be run in any of the following configurations: 

- single GPU
- multiGPUs (using PyTorch distributed mode)
- FP16, BF16 (mixed-precision), FP32, BF16 (normal precision)

"""

def setup_wandb(train_config):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently inwstalled. "
            "Please install it using pip install wandb"
        )
    from configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **dataclasses.asdict(train_config))
    
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)

    # show configurations
    mode = "mixed" if train_config.mixed_precision else "normal"
    dtype = train_config.dtype
    
    run.tags =[mode, dtype]
    run.name = f"run_ep_{train_config.num_epochs}_lr_{train_config.lr}_bs_{train_config.batch_size_training * train_config.gradient_accumulation_steps}_wd_{train_config.weight_decay}"

    return run 
        

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)

def get_dataloader(config, tokenizer, split="train", do_eval=False):
    dataset_config = generate_dataset_config(config)
    dataset = get_preprocessed_dataset(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        split=split,
        do_eval=do_eval,
        num_examples=config.debug_n_example if config.debug else None, 
    )

    if config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=config.max_context_length)
    
    print(f"dataset length = {len(dataset)}")
    dl_kwargs = get_dataloader_kwargs(config, dataset, tokenizer, split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.num_workers_dataloader,
        pin_memory=True,
        **dl_kwargs,
    )
    return dataloader

def get_optimizer_scheduler(train_config, model):
    if train_config.use_anyprecision:
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=ANY_PSN_DTYPE[train_config.dtype],
            variance_dtype=ANY_PSN_DTYPE[train_config.dtype],
            use_kahan_summation=train_config.use_kahan_summation,
            weight_decay=train_config.weight_decay,
        )
        
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    
    lr_scheduler =  StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    return optimizer, lr_scheduler


def get_rope_scaling_factor(train_config):
    # Set RoPE scaling factor
    config = AutoConfig.from_pretrained(
        train_config.model_name,
        cache_dir=train_config.output_dir,
    )

    orig_rope_scaling_factor = train_config.rope_scaling
    orig_ctx_len = train_config.context_length
    orig_ctx_len *= orig_rope_scaling_factor
    # expand original context_length to model_max_length
    if train_config.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(train_config.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    return config

def replace_full_to_sparse_attn(train_config):
    # longlora    
    if train_config.use_peft and train_config.peft_method == "longlora":
        # NOTE(LongLoRA): May expand supported model types in the future
        if train_config.model_type == "gpt-neox":
            replace_gpt_neox_attn(train_config.use_flash_attn, train_config.use_full_attn)
        else:
            assert train_config.model_type == "llama", "Only support llama and gpt-neox for now"
            replace_llama_attn(train_config.use_flash_attn, train_config.use_full_attn)

def set_gradient_checkpointing(model, train_config):
    # enable trainable params
    #[p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in train_config.trainable_params.split(",")])]

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    return model

def check_config_dependency(train_config):
    if train_config.use_peft and train_config.peft_method == "longlora":
        assert train_config.model_max_length >= train_config.context_length
        assert train_config.use_full_attn == False
        assert train_config.batching_strategy == "padding"
        assert train_config.run_eval == False #TODO(HJ) need to be fixed
    
    if train_config.dataset in ["alpaca", "alpaca_long"]:
        train_config.run_eval = False
    else:
        train_config.model_max_length = deepcopy(train_config.context_length)

def main(args):
    train_config = TRAIN_CONFIG()
    update_config(train_config, **vars(args))
    check_config_dependency(train_config)    
 
    set_seed(train_config.seed)
 
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=train_config.deepspeed_stage, 
                            gradient_accumulation_steps=train_config.gradient_accumulation_steps) \
                            if train_config.use_deepspeed else None
    mixed_precision =MIXED_PSN_DTYPE[train_config.dtype] if train_config.mixed_precision else None
    fp8_kwargs = [FP8RecipeKwargs(backend="te")] if train_config.dtype == "fp8" else None
    
    accelerator = Accelerator(mixed_precision=mixed_precision,
                              deepspeed_plugin=deepspeed_plugin,
                              kwargs_handlers=fp8_kwargs)
    
    wandb_run = setup_wandb(train_config) if train_config.use_wandb else None
    
    replace_full_to_sparse_attn(train_config) # if use longlora 
    config = get_rope_scaling_factor(train_config)
    
    model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            config=config, 
            load_in_8bit=True if train_config.quantization else None,
            device_map=None, #if int(os.environ["WORLD_SIZE"]) > 1 else "auto",
            #use_cache=True,
            torch_dtype=train_config.dtype,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
                train_config.model_name,
                model_max_length=train_config.model_max_length,
                #padding_side="right", 
                use_fast=True,)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if train_config.model_max_length > train_config.context_length:
        print(f"extend model context_length:{train_config.context_length} into {train_config.model_max_length}")
        # also 
        model.resize_token_embeddings(len(tokenizer))
    
    print_model_size(model, train_config)
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, **vars(args)) 
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
        if wandb_run:
            wandb_run.config.update(peft_config)
    
    train_dataloader = get_dataloader(train_config, tokenizer, "train")
    eval_dataloader = get_dataloader(train_config, tokenizer, "validation") if train_config.run_eval else None


    model = set_gradient_checkpointing(model, train_config) if train_config.peft_method == "longlora" else model        
    model = model.to(accelerator.device)
    optimizer, lr_scheduler = get_optimizer_scheduler(train_config, model)
    
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_config=train_config,
        logger=wandb_run,
        accelerator=accelerator,
    )

    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
    if train_config.use_wandb:
        for k,v in results.items():
            wandb_run.summary[k] = v
 
def get_args():
    parser = argparse.ArgumentParser(description="low precision finetuning")
    parser.add_argument(
        "--mixed_precision", 
        action="store_true",
        help="if False, use normal precision following 'dtype'" 
    )
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
        "--use_anyprecision", 
        action="store_true",
        help="if false, use torch.optim.optimizer"
    )
    
    parser.add_argument(
        "--peft_method",
        type=str, 
        default="lora",
        help="set peft method",
        choices=["lora", "prompt", "adalora", "prefix", "boft", "longlora"], 
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

    parser.add_argument(
        "--model_max_length",
        type=int, 
        default=None,
        help="model max length to extend using longlora",
    )

    parser.add_argument(
        "--debug", 
        action="store_true",
        help="debug mode on. if true then load only a few examples",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    main(args)