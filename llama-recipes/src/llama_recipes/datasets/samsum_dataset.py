# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import torch
import datasets
# import functools

def apply_prompt_template(sample):
    prompt = (
        f"Summarize this dialog:\n{{dialogue}}\n---\nSummary:\n"
    )

    return {
        "prompt": prompt.format(dialogue=sample["dialogue"]),
        "summary": sample["summary"],
    }


def get_preprocessed_samsum(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("samsum", split=split)

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample
    
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

def prepare_instructions(dialogues, summaries):
    instructions = []

    prompt = (
        f"Summarize this dialog:\n{{dialogue}}\n---\nSummary:\n"
    )

    for dialogue, summary in zip(dialogues, summaries):
        example = prompt.format(
            dialogue=dialogue,
        )
        instructions.append(example)

    return instructions


def prepare_samsum_data(tokenizer):
    dataset = datasets.load_dataset("samsum")
    val_dataset = dataset["test"]

    dialogues = val_dataset["dialogue"]
    summaries = val_dataset["summary"]

    instructions = prepare_instructions(dialogues, summaries)

    val_dataset = val_dataset.map(apply_prompt_template, remove_columns=list(val_dataset.features))

    def tokenize_add_label(instruction):
        prompt = tokenizer.encode(tokenizer.bos_token + instruction["prompt"], add_special_tokens=False)

        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "labels" : [-100] * len(prompt)
        }

        return sample
    
    val_dataset = val_dataset.map(tokenize_add_label, remove_columns=list(val_dataset.features))

    return val_dataset, instructions, summaries