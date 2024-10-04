from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Any, List
from star.logger.config import BaseLoggerConfig
from star.agent.config import AgentConfig
@dataclass
class TrainerConfig:
    defaults: List[Any] = field(default_factory=lambda : [
        {"/agent": "AgentConfig"},
        "_self_",
        ])
    _target_: str = "star.trainer.trainer.Trainer"
    logger: BaseLoggerConfig = MISSING
    agent: AgentConfig = MISSING
    device: str = "${device}"
    output_dir: str = "${hydra:runtime.output_dir}"
    gen_batch_size: int = 128
    single_batch: bool = False
    reinforce_batch_size: int = 16
    num_workers: int = 3
    dataset_name: str = "mmlu_YO-NG"
    _partial_: bool = True
    # seed: int = 0 # for numbers 0 or lower, the seed is random. This is for easy testing.
ConfigStore.instance().store(name="TrainerConfig", node=TrainerConfig, group="trainer")


