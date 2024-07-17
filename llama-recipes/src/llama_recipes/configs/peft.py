# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import List

@dataclass
class lora_config:
     r: int=8
     lora_alpha: int=32
     target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
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
     num_virtual_tokens: int=30
     task_type: str= "CAUSAL_LM"

@dataclass
class prompt_config:
     task_type: str = "CAUSAL_LM"
     num_virtual_tokens: int=32
     prompt_tuning_init_text: str = "Please summarize the following conversation."
     #tokenizer_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"

@dataclass
class adalora_config:
     task_type: str = "CAUSAL_LM"
     r: int = 8
     lora_alpha: int=32
     target_modules: List[int] = field(default_factory=lambda: ['q_proj', 'v_proj'])
     lora_dropout: float=0.01