import os

import dataclasses
import fire
import random
import numpy as np
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training

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
    init_dict['project'] = train_config.dataset.split("_")[0]
    run = wandb.init(**init_dict)
    run.config.update(train_config)

    # show configurations
    mode = "mixed" if train_config.mixed_precision else "normal"
    dtype = train_config.dtype
    
    run.tags =[mode, dtype]
    run.name = train_config.output_dir

    return run 
        

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)

def get_dataloader(train_config, tokenizer, split="train"):
    dataset_config = generate_dataset_config(train_config)
    dataset = get_preprocessed_dataset(
        tokenizer,
        dataset_config, 
        split=split,
    )
    
    if train_config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)
    print(f"dataset length = {len(dataset)}")

    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_config.num_workers_dataloader,
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


def main(**args):
    train_config = TRAIN_CONFIG()
    update_config(train_config, **args)

    set_seed(train_config.seed)
    
    #if not train_config.enable_fsdp:

    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, 
                            gradient_accumulation_steps=train_config.gradient_accumulation_steps) \
                            if train_config.use_deepspeed else None
    mixed_precision =MIXED_PSN_DTYPE[train_config.dtype] if train_config.mixed_precision else None
    fp8_kwargs = [FP8RecipeKwargs(backend="te")] if train_config.dtype == "fp8" else None
    
    accelerator = Accelerator(mixed_precision=mixed_precision,
                              deepspeed_plugin=deepspeed_plugin,
                              kwargs_handlers=fp8_kwargs)
    
    wandb_run = setup_wandb(train_config) if train_config.use_wandb else None

    model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map=None if int(os.environ["WORLD_SIZE"]) > 1 else "auto",
            use_cache=True,
            torch_dtype=ANY_PSN_DTYPE[train_config.dtype],
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, padding_side='left')
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
    
    print_model_size(model, train_config)

    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, **args)
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
        if wandb_run:
            wandb_run.config.update(peft_config)
    

    train_dataloader = get_dataloader(train_config, tokenizer, "train")
    eval_dataloader = get_dataloader(train_config, tokenizer, "val") if train_config.run_eval else None
    
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

if __name__ == "__main__":

    # args = get_args()
    # main(args)
    fire.Fire(main)