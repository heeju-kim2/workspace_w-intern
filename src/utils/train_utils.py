import os
import time 
import yaml
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm
import json

#from model_checkpointing import save_model_checkpoint, save_optimizer_checkpoint
from torch.nn.utils import clip_grad_norm_
from accelerate.utils import is_xpu_available
import evaluate
import numpy as np
from utils.eval_utils import (
    rouge_for_samsum,
    em_for_gsm8k
)
from peft import get_peft_model_state_dict
# from utils.memory_utils import MemoryTrace
# from utils.flop_utils import FlopMeasure

def set_tokenizer_params(tokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")

def evaluation(model, 
              eval_dataloader, 
              tokenizer, 
              train_config, 
              logger,
              accelerator,
              curr_train_epoch):

    max_eval_step = train_config.max_eval_step

    model.eval()
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    eval_metric = 0
    # metric = evaluate.load("rouge") ##TODO(HJ) need to change

    for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
        total_eval_steps += 1
        if max_eval_step > 0 and total_eval_steps > max_eval_step:
            print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
            break
        
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch = {k :v.to(accelerator.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
        
        # gather all inferences across chips
        accelerator.wait_for_everyone()
    
    if train_config.dataset == "samsum_dataset":
        eval_metric = rouge_for_samsum(train_config, model, tokenizer, accelerator)
    elif train_config.dataset == "gsm8k_dataset":
        eval_metric = em_for_gsm8k(train_config, model, tokenizer, logger, epoch=curr_train_epoch, full=curr_train_epoch+1==train_config.num_epochs)
    
    # Use accelerator.print to print only on the main process.
    accelerator.print(f"epoch {curr_train_epoch}:", eval_metric)

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)

    if logger:
        logger.log({
                    'eval/perplexity': eval_ppl,
                    'eval/loss': eval_epoch_loss,
                    'eval/metric': eval_metric,
                    })

    return eval_ppl, eval_epoch_loss

def train(model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_config,
        logger,
        accelerator,
        
    ):
    # prepares data and models for distributed processing by moving them to chips
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    ## delete
    num_epochs = train_config.num_epochs
    gradient_accumulation_steps = train_config.gradient_accumulation_steps
    run_eval = train_config.run_eval
    max_train_step = train_config.max_train_step
    gradient_clipping = train_config.gradient_clipping
    gradient_clipping_threshold = train_config.gradient_clipping_threshold

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached

    # first eval
    if run_eval:
        eval_ppl, eval_epoch_loss = evaluation(model, eval_dataloader, tokenizer, train_config, logger, accelerator, -1)
        print(f"start score, eval_ppl: {eval_ppl} | eval_epoch_loss: {eval_epoch_loss}")
        if eval_epoch_loss < best_val_loss:
            best_val_loss = eval_epoch_loss
        
        val_loss.append(float(best_val_loss))
        val_prep.append(float(eval_ppl))
    
    # Now we train the model
    for epoch in range(num_epochs):
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_length = len(train_dataloader) // gradient_accumulation_steps
        pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            
        for step, batch in enumerate(train_dataloader):
            total_train_steps += 1
            if max_train_step > 0 and total_train_steps > max_train_step:
                max_steps_reached = True
                break
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch = {k : v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            total_loss += loss.detach().float()
            accelerator.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if gradient_clipping and gradient_clipping_threshold > 0.0:
                    clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
                optimizer.step()
                print(lr_scheduler.get_lr())
                optimizer.zero_grad()
                pbar.update(1)

            if logger:
                logger.log({
                    'train/epoch': epoch + 1,
                    'train/step': epoch * len(train_dataloader) + step,
                    'train/loss': loss.detach().float(),})

            # clear_gpu_cache()
            # accelerator.free_memory()
            pbar.set_description(f"Training Epoch: {epoch+1}/{num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

        lr_scheduler.step()
        pbar.close()
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)

        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if run_eval:
            eval_ppl, eval_epoch_loss = evaluation(model, eval_dataloader, tokenizer, train_config, logger, accelerator, epoch)

            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                accelerator.print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))

            ## save peft layers
            accelerator.wait_for_everyone()
            peft_state_dict = get_peft_model_state_dict(model)
            output_dir = os.path.join(train_config.output_dir, f"epoch_{epoch}")
            model.save_pretrained(output_dir)
            print(f"Epoch {epoch+1}: eval_ppl: {eval_ppl} | eval_epoch_loss: {eval_epoch_loss}")
        accelerator.print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")


    results['avg_train_prep'] = sum(train_prep)/len(train_prep)
    results['avg_train_loss'] = sum(train_loss)/len(train_loss)
    results["avg_epoch_time"] = sum(epoch_times)/ len(epoch_times)
    results["avg_checkpoint_time"] = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    if run_eval:
        results['avg_eval_prep'] = sum(val_prep)/len(val_prep)
        results['avg_eval_loss'] = sum(val_loss)/len(val_loss)
    return results