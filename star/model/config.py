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
    model_name: str = MISSING
    max_seq_length: int = 2048
    # things related to unsloth should go here, but might be complicated like the peft that is being used
    # I might do Peft support later???

    # any things related to optimization aswell
    epochs: int = 1
    max_steps: int = -1
    warmup: int = 5
    scheduler_str: str = "lambda"
    optimizer_partial: Any = MISSING

ConfigStore.instance().store(name="AdamW", node={"_target_":"torch.optim.adamw.AdamW", 
                                                                     "lr": 0.0002,
                                                                     "weight_decay": 0.01,
                                                                     "_partial_": True}, group="optimizer_partial")

ConfigStore.instance().store(name="UnslothModelHandlerConfig", node=UnslothModelHandlerConfig(defaults=[{"/optimizer_partial": "AdamW"},
                                                                                                        "_self_",
                                                                                                        ],
                                                                                              model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
                                                                                             ), group="model_handler")
