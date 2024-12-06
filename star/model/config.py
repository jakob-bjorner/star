from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Any, List, Optional

# @dataclass
# class BaseModelHandlerConfig:
#     _target_: str = MISSING

@dataclass
class UnslothModelHandlerConfig: # (BaseModelHandlerConfig):
    defaults: List[Any] = field(default_factory=lambda: [
        "_self_",
        ])
    _target_: str = "star.model.model.UnslothModelHandler"
    # model_name: str = MISSING
    max_seq_length: int = 4096
    # checkpoint base is for the tokenizer to be consistent.
    checkpoint_base: str = "unsloth/Meta-Llama-3.1-8B-Instruct" # "unsloth/Llama-3.2-1B-Instruct" # 
    # things related to unsloth should go here, but might be complicated like the peft that is being used
    # I might do Peft support later???

    # any things related to optimization aswell
    # epochs: int = 1
    # max_steps: int = -1
    warmup: int = 5
    scheduler_str: str = "lambda" # can be const
    optimizer_partial: Any = MISSING
    lora_r: int = 16
# TODO: look at adamw_8bit optimizer 
# https://huggingface.co/blog/mlabonne/sft-llama3
# https://huggingface.co/docs/bitsandbytes/main/en/optimizers
# likely need to see how unsloth does this by default as parameters seem to need to be changed to stable versions...

ConfigStore.instance().store(name="AdamW", node={"_target_":"torch.optim.adamw.AdamW", 
                                                                     "lr": 0.0002,
                                                                     "weight_decay": 0.01,
                                                                     "betas": (0.9, 0.95),
                                                                     "_partial_": True}, group="optimizer_partial")

ConfigStore.instance().store(name="UnslothModelHandlerConfig", node=UnslothModelHandlerConfig(defaults=[{"/optimizer_partial": "AdamW"},
                                                                                                        "_self_",
                                                                                                        ],
                                                                                            #   model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
                                                                                             ), group="model_handler")
