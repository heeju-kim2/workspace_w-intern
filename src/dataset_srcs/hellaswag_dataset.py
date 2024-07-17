# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }

class HellaSwagDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split="train", num_examples=None):
        self.dataset = load_dataset("Rowan/Hellaswag", split=split)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }


def get_preprocessed_hellaswag(dataset_config, tokenizer, split, num_examples=None): 
    if num_examples:
        split += f"[:{num_examples}]"
    dataset = load_dataset("Rowan/Hellaswag", split=split)
    print(dataset)
    

    def tokenize_add_label(data):
        prompt = tokenizer.encode(tokenizer.bos_token + data["ctx"], add_special_tokens=False)
        samples = list()
        for i in range(4):
            summary = tokenizer.encode(data["endings"][i] +  tokenizer.eos_token, add_special_tokens=False)
            sample = {
                "input_ids": prompt + summary,
                "attention_mask" : [1] * (len(prompt) + len(summary)),
                "labels": [-100] * len(prompt) + summary,
                }
            samples.append(sample)
        return samples

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset


if __name__ == "__main__" :
    from transformers import AutoTokenizer
    dataset_config = None
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    dataset = get_preprocessed_hellaswag(dataset_config, tokenizer, "train", 10)
    print(dataset)
    