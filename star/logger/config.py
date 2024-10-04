from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Any, List, Optional

@dataclass
class BaseLoggerConfig:
    name: Optional[str] = None
    # could but the logger name init in the wandb init function, so that the logic is only stored there. 
    # - slightly more irritating to change,
    # - the print logger doesn't get the benefit of having the same stuff logged to it.
    # - this is really just a wandb specific convention, but it is a nice convention to extend to all other loggers I think.
    _target_: str = MISSING

@dataclass
class WandBLoggerConfig(BaseLoggerConfig):
    # in the future might want to distinguish between what I pass into the __init__ of the logger and what I pass into the init() of wandb. 
    # Something like higher level options to resume runs or add data to old runs...
    project: str = "star"
    dir: str = "${hydra:runtime.output_dir}"
    _target_: str = "star.logger.logger.WandBLogger"


ConfigStore.instance().store(name="WandBLoggerConfig", node=WandBLoggerConfig, group="logger")
ConfigStore.instance().store(name="CustomWandBLogger", node=WandBLoggerConfig(_target_="star.logger.logger.CustomWandBLogger"), group="logger")
ConfigStore.instance().store(name="BasicPrintLoggerConfig", node={"name": None,
                                                                  "_target_": "star.logger.logger.BasicPrintLogger",
                                                                  "output_dir": "${hydra:runtime.output_dir}"
                                                                  }, group="logger")
