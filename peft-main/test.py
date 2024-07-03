from src.peft import PrefixTuningConfig, get_peft_model
from src.peft.utils import dataset
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import get_linear_schedule_with_warmup, default_data_collator
from llama_recipes.utils import fsdp_auto_wrap_policy
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
import os
import evaluate
import numpy as np
from rouge_score import rouge_scorer

def get_rouge(eval_preds, eval_labels):
    """
    Calculate the rouge score
    """
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=eval_preds, references=eval_labels)

    return scores

# model_name = "../../models/Llama-2-7b-chat-hf"
model_name = "SalmanFaroz/Llama-2-7b-samsum"
lr = 3e-2
num_epochs = 1
batch_size = 1

# setup()

if torch.distributed.is_initialized():
    clear_gpu_cache(0)
    setup_environ_flags(0)

model = LlamaForCausalLM.from_pretrained(model_name)

# peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
# fsdp_config = FSDP_CONFIG()
# rank = 0

# mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
# my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

# device_id = 0

# model = FSDP(
#     model,
#     auto_wrap_policy= my_auto_wrapping_policy,
#     cpu_offload= None,
#     mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
#     sharding_strategy=fsdp_config.sharding_strategy,
#     device_mesh=None,
#     device_id=device_id,
#     limit_all_gathers=True,
#     sync_module_states=False,
#     param_init_fn=None,
# )

# Load the tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

train_ds = dataset.get_preprocessed_samsum(tokenizer=tokenizer, split="train") # tokenizer, split(train, test)
eval_ds = dataset.get_preprocessed_samsum(tokenizer=tokenizer, split="test") # tokenizer, split(train, test)

train_dataloader = DataLoader(train_ds, shuffle=True, pin_memory=True, collate_fn=default_data_collator, batch_size=batch_size)
eval_dataloader = DataLoader(eval_ds, shuffle=False, pin_memory=True, collate_fn=default_data_collator, batch_size=batch_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

from tqdm import tqdm

device = "cuda"
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    eval_labels = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        labels = batch['summary'].cpu().numpy()
        del batch['summary']

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

        eval_labels.extend(
            # tokenizer.batch_decode(batch['labels'].cpu().numpy(), skip_special_tokens=True)
            tokenizer.batch_decode(labels, skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    # train_epoch_loss = total_loss / len(train_dataloader)
    # train_ppl = torch.exp(train_epoch_loss)
    rouge_score = get_rouge(eval_preds=eval_preds, eval_labels=eval_labels)
    print(rouge_score)
    print(f"{epoch=}: {eval_ppl=} {eval_epoch_loss=}")
    # print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")