# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import torch
import datasets
import functools
from llama_recipes.data.concatenator import ConcatDataset

def apply_prompt_template(sample):
    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    return {
        "prompt": prompt.format(dialog=sample["dialogue"]),
        "summary": sample["summary"],
    }

def tokenize_add_label(sample, tokenizer, split):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        if split == "train":
            sample = {
                "input_ids": prompt + summary,
                "attention_mask" : [1] * (len(prompt) + len(summary)),
                "labels": [-100] * len(prompt) + summary,
            }
        else:
            sample = {
                "input_ids": prompt,
                "attention_mask" : [1] * len(prompt),
                "labels": [-100] * len(prompt),
            }

        return sample

# input only prompts to calculate rouge score
def get_rouge_dataset(config, tokenizer, split="validation"):
    dataset = datasets.load_dataset("samsum", split=split)

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    summaries = [sample["summary"] for sample in dataset]
    dataset = dataset.map(functools.partial(tokenize_add_label, tokenizer=tokenizer, split=split), remove_columns=list(dataset.features))

    if config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=config.context_length)
    
    return dataset, summaries

def get_preprocessed_samsum(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("samsum", split=split)

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(functools.partial(tokenize_add_label, tokenizer=tokenizer, split=split), remove_columns=list(dataset.features))

    return dataset