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
    data_path: str = "raw_datasets/alpaca_dataset.json" #"tatsu-lab/alpaca"
    #data_path: str = "alpaca_dataset.json"

@dataclass
class alpaca_long_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = "raw_datasets/alpaca_long_dataset.json" #"tatsu-lab/alpaca"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class redpajama_dataset:
    dataset: str = "redpajama_dataset"
    train_split: str = "train"
    test_split: str = "validation"

## asr_dataset
@dataclass
class common_voice_dataset:
    dataset: str = "common_voice_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_path: str = "mozilla-foundation/common_voice_11_0"
    language: str="hi" # language to use hindi
    task: str="transcribe" # task to use for training
    model_name: str = "openai/whisper-small" # need config update 
    do_lower_case : bool = False
    do_remove_punctuation : bool = False 
