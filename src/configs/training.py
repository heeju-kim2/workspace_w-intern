from dataclasses import dataclass

@dataclass
class train_config:
    # model 
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None

    # low precision training 
    mixed_precision: bool=False # 
    dtype: str="bfloat16" #alternative: fp32, fp16, bf16, fp8 w/mp, fp32, bf16 w/o mp
    
    # log and save
    output_dir: str = "prompt_tuning"
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting

    # eval 
    run_eval: bool=True
    eval_batch_size: int=1
    max_eval_step: int=0

    #dataloader
    dataset = "samsum_dataset" # alternative : alpaca_dataset
    num_workers_dataloader: int=2

    #train
    batch_size_training: int=1
    batching_strategy: str="padding" #alternative : packing, padding
    context_length: int=4096 # model context length 
    gradient_accumulation_steps: int=4
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=5
    max_train_step: int=0
    num_train_exmaples: str = None # for debug
    
    use_anyprecision: bool = False 
    use_kahan_summation: bool = True # use if optimizer == "Anyprecison"
    lr: float=1e-4
    weight_decay: float=0.1
    gamma: float= 0.85
    seed: int=42
    
    
    # peft
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=True
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    qlora: bool=False
    r:int = 64 # lora rank
    target_r:int = 64
    from_peft_checkpoint:str = ""

    # quantization 
    quantization: bool = False

    # distribution training or acceleration
    #enable_fsdp: bool=False
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_deepspeed: bool=False
    
    # profile
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
     
    # gsm8k
    few_shot: str = "none" # few_shot will be activated if EM, test, train in few_shot
    no_prompt: bool = False # give no instruction prompt