from dataclasses import dataclass, field
from typing import List
#from pyreft import LoreftIntervention

@dataclass
class lora_config:
     r: int=64
     lora_alpha: int=64
     # target_modules: List[str] = field(default_factory=lambda: ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"])
     target_modules: List[str] = field(default_factory=lambda: ["v_proj", "q_proj"])
     bias= "none"
     task_type: str= "CAUSAL_LM"
     lora_dropout: float=0.05
     inference_mode: bool = False

@dataclass
class llama_adapter_config:
     adapter_len: int= 10
     adapter_layers: int= 30
     task_type: str= "CAUSAL_LM"

#CAUTION prefix tuning is currently not supported
@dataclass
class prefix_config:
     encoder_hidden_size: int=768
     prefix_projection: bool=True
     num_virtual_tokens: int=30
     task_type: str= "CAUSAL_LM"


@dataclass
class prompt_config:
     task_type: str = "CAUSAL_LM"
     num_virtual_tokens: int=32
     prompt_tuning_init_text: str = "Please summarize the following conversation."
     prompt_tuning_init: str = "TEXT"
     tokenizer_name_or_path: str = "models/Llama-2-7b-chat-hf"

@dataclass
class adalora_config:
     task_type: str = "CAUSAL_LM"
     target_r: int = 64
     init_r: int = 96
     tinit: int = 1
     tfinal: int = 190
     deltaT: int = 1
     lora_alpha: int=256
     beta1: float=0.85
     beta2: float=0.85
     orth_reg_weight: float=0.1
     target_modules: List[str] = field(default_factory=lambda: ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],)
     lora_dropout: float=0.05
     bias= "none"
     inference_mode: bool=False
     
@dataclass
class boft_config:
     task_type: str = "CAUSAL_LM"
     boft_block_size: int = 4
     boft_n_butterfly_factor: int = 2
     target_modules: List[str] = field(default_factory=lambda : ["q_proj", "v_proj", "k_proj", "o_proj"])
     boft_dropout : float = 0.1
     bias: str = "boft_only"

# @dataclass
# class reft_config:
#      intervention = LoreftIntervention(embed_dim=4096 , low_rank_dimension=4) # llama-2-7b hidden_dim
#      layer: int = 15
#      component : str = "block_output"
#      low_rank_dimension : int = 4

     
