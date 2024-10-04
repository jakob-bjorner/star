from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Any, List, Optional
from star.model.config import UnslothModelHandlerConfig

@dataclass
class AgentConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        {"/model_handler": "UnslothModelHandlerConfig"},
        "_self_",
    ])
    _target_: str = "star.agent.agent.Agent"
    model_handler: UnslothModelHandlerConfig = MISSING

ConfigStore.instance().store(name="AgentConfig", node=AgentConfig, group="agent")
