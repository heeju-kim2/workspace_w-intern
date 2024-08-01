import copy
import torch
import datasets
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq
import random
from itertools import islice

import numpy as np
import torch


def get_preprocessed_redpajama_dataset(
    dataset_config, 
    tokenizer, 
    split, 
    num_examples=None, 
    do_eval=False):
    
    if num_examples:
        split += f"[:{num_examples}]"
    
    dataset = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split, trust_remote_code=True)
    
    def apply_tokenizer(example):
        context_length = tokenizer.model_max_length
        outputs = tokenizer(
                example["text"] ,
                truncation=False,
                return_tensors="pt",
                pad_to_multiple_of=context_length+1,
                padding=True,
            )
        input_ids = outputs["input_ids"].view(-1, context_length+1)[0]
        #return {"input_ids":input_ids}
        return {"input_ids": input_ids[:-1], "labels": input_ids[1:] }

    dataset = dataset.map(apply_tokenizer, batched=False, num_proc=128, remove_columns=["text", "meta"])
    return dataset

if __name__ == "__main__":

    from transformers import AutoTokenizer
    import torch
    
    model_name="meta-llama/Llama-2-7b-chat-hf"
    model_max_length=8192

    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=model_max_length,
            #padding_side="right", 
            use_fast=True,)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = get_preprocessed_redpajama_dataset(None, tokenizer, "train", 10)
    print(len(dataset[0]['input_ids']))
    print(len(dataset[0]['labels']))
    print(dataset[0]['labels'])
    print(dataset[0]['input_ids'][:10])
    print(dataset[0]['labels'][:10])
    # print(dataset[0].keys())
    # print(dataset[0]['input_ids'])
    # print(dataset)
    # batch_size=2
    # batch_sampler = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=True)
    # collate_fn = DataCollatorForSeq2Seq(tokenizer)
    # dataloader = torch.utils.data.DataLoader(dataset, num_workers=2, collate_fn=collate_fn)
    

    # for batch in dataloader:
    #     print(batch)
    #     exit(0)    