from dataclasses import dataclass

@dataclass
class eval_config:
    # model 
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None

    # peft
    use_peft: bool =False
    peft_path: str = None
    dtype: str = "float16"

    #dataloader
    dataset: str = "samsum_dataset" #"samsum_dataset" # alternative : passkey_retrieval, 
    num_workers_dataloader: int=2
    batching_strategy: str = "padding" #alternative : packing, padding
    eval_batch_size: int=1
    # eval
    metric: str = "rouge" # rouge for samsum, mt-bench, alpaca_eval for alpaca,
    seed: int = 12345    


    # decode
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 2048

    # save
    save_inference: bool = True
    save_metrics: bool = True
    output_dir: str = "PATH/to/PEFT"
    
    # acceleration
    use_fast_kernels: bool = True
    quantization: bool = False

    
    # debug
    debug_n_example: int = 5

