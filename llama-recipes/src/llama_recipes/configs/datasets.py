# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    

@dataclass
class gsm8k_dataset:
    dataset: str = "gsm8k_dataset"
    train_split: str = "train"
    test_split: str = "test"
    few_shot: str = "none" # EM,test,train few_shot in split으로 few_shot 추가 여부 결정
    data_path: str = "src/llama_recipes/datasets/gsm8k/data"


@dataclass
class hella_dataset:
    dataset: str = "hella_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"