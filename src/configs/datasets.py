from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = "tatsu-lab/alpaca"
    #data_path: str = "alpaca_dataset.json"
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class gsm8k_dataset:
    dataset: str = "gsm8k_dataset"
    train_split: str = "train"
    test_split: str = "test"
    few_shot: str = "none" # EM, test, train in few_shot으로 각각 few_shot 여부 결정