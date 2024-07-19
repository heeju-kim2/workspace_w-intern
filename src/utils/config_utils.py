from dataclasses import asdict
from torch.utils.data import DistributedSampler
from peft import LoraConfig
import inspect

import torch.distributed as dist
from torch.utils.data import DistributedSampler

from configs import (
    datasets, 
    lora_config, 
    llama_adapter_config, 
    prefix_config, 
    train_config, 
    prompt_config, 
    adalora_config,
    boft_config,)

from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    AdaLoraConfig,
    BOFTConfig,
)
#from pyreft import ReftConfig
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq

from data.sampler import LengthBasedBatchSampler
from utils.dataset_utils import DATASET_PREPROC

def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)


def generate_peft_config(train_config, **kwargs):
    configs = (
        lora_config, 
        llama_adapter_config, 
        prefix_config, 
        prompt_config, 
        adalora_config,
        boft_config,
        #reft_config,
        )
    peft_configs = (
        LoraConfig, 
        AdaptionPromptConfig, 
        PrefixTuningConfig, 
        PromptTuningConfig, 
        AdaLoraConfig,
        BOFTConfig,
        #ReftConfig, 
        )
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    if train_config.peft_method not in names:
        raise RuntimeError(f"Peft config not found: {train_config.peft_method}")


    config = configs[names.index(train_config.peft_method)]()

    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)

    return peft_config


def generate_dataset_config(train_config):
    names = tuple(DATASET_PREPROC.keys())

    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"
    dataset_config = {k:v for k, v in inspect.getmembers(datasets)}[train_config.dataset]()
    dataset_config.few_shot = train_config.few_shot

    return  dataset_config


def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
        kwargs = {}
        batch_size = train_config.batch_size_training if mode=="train" else train_config.eval_batch_size
        if train_config.batching_strategy == "padding":
            kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
            kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        elif train_config.batching_strategy == "packing":
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
            kwargs["collate_fn"] = default_data_collator
        else:
            raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")

        return kwargs