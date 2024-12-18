from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict
import os

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from omegaconf import MISSING, OmegaConf
from hydra.core.config_store import ConfigStore
OmegaConf.register_new_resolver("eval", eval)
cs = ConfigStore.instance()

from star.trainer.config import TrainerConfig
from star.init_configs import init_configs

init_configs()

@dataclass
class CustomKargoLauncherConfig(SlurmQueueConf): 
    """ https://hydra.cc/docs/1.3/plugins/submitit_launcher/ then go to github and look at config.py this is what I extend.
        to run things locally, use the option on launch `python run.py hydra/launcher=submitit_local`, 
        or in this case, without -m it launches things locally.
    """
    # submitit_folder: str = 
    # the default submitit_folder = "${hydra.sweep.dir}/.submitit/%j"
    # so reasonable and can't make it anything more reasonable it seems, because 
    # they launch with map_executor on the backend, which is the best for my 
    # parallel jobs, but prevents nicely putting the submitit loggs into more 
    # careful foldering. Perhaps in the future I can follow a per experiment 
    # foldering, and the specificity of the sweep.dir folder will be more useful 
    # to me.
    timeout_min: int = 2880 # 60 * 24 * 2
    # cpus_per_task: int|None = 6 # type: ignore
    gpus_per_node: int|None = None
    tasks_per_node: int =  1
    mem_gb: int|None =  None
    nodes: int = 1
    _target_: str = "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    partition: str|None = "kargo-lab" # overcap
    qos: str|None = "short"
    exclude: str|None = "major,crushinator,nestor,voltron,xaea-12,samantha"
    additional_parameters: Dict[str, Any] = field(default_factory=lambda: {"cpus-per-gpu": 6, "gpus-per-node": "a40:1", "requeue": True})
    array_parallelism: int = 20
cs.store(name="custom_kargo_submitit", node=CustomKargoLauncherConfig, group="hydra/launcher")

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        {"logger": "BasicPrintLoggerConfig"},
        {"trainer": "TrainerConfig"},
        {"override hydra/launcher": os.getenv("GREEK_LAUNCHER", "custom_kargo_submitit")},
        # {"override hydra/sweeper": "optuna"}, # https://hydra.cc/docs/plugins/optuna_sweeper/
        # {"override hydra/sweeper/sampler": "random"}, 
        "_self_",
        ])
    logger: Any = MISSING
    run_type: str = "train"
    node_name: str = MISSING
    device: str = os.getenv('device', "cuda")
    seed: int = 2
    trainer: TrainerConfig = MISSING
    # batch_size: int = 128
    # known issue: the multirun.yaml is saved to the sweep dir, and not the subdirs, so it is not saved! (don't think I will need this to be saved tho, and makes folders easier to read)
    hydra: Any = field(default_factory=lambda: {
        "sweep":{"dir": "star_runs", 
                 "subdir": "${run_type}_${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.num}" },
        "run":{"dir":  "star_runs/${run_type}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
        # "sweeper": {"sampler": "random"},
    })
cs.store(name="RunConfig", node=RunConfig)



def get_run_name_from_cfg(cfg: RunConfig):
    name = f"{cfg.run_type}"
    if "prompt_perf" in cfg.run_type:
        name += f"_prompt={cfg.trainer.meta_prompt_to_preamble}_qs={cfg.trainer.num_train_questions}"
    if 'dpo' in cfg.trainer.train_strategy:
        name += f"_beta={cfg.trainer.dpo_beta}_samples={cfg.trainer.samples_per_train_question}"
    name += f"_ts={cfg.trainer.train_strategy}_ds={cfg.trainer.dataset_name}_lr={cfg.trainer.agent.model_handler.optimizer_partial.lr}_wd={cfg.trainer.agent.model_handler.optimizer_partial.weight_decay}_lora={cfg.trainer.agent.model_handler.lora_r}_gaccum={cfg.trainer.train_gradient_accumulation_steps}_warmup={cfg.trainer.agent.model_handler.warmup}_gnorm={cfg.trainer.max_grad_norm}_shuff={cfg.trainer.shuffle_dataloader}_seed={cfg.seed}"
    if cfg.trainer.max_epochs > 1:
        name += f"_resetModel={cfg.trainer.reset_model_every_star_epoch}"
    if cfg.run_type.startswith("eval"):
        # replace the name with the model and dataset?
        name = f"{cfg.run_type}_lr={cfg.trainer.agent.model_handler.optimizer_partial.lr}_wd={cfg.trainer.agent.model_handler.optimizer_partial.weight_decay}_lora={cfg.trainer.agent.model_handler.lora_r}"
        if 'dpo' in cfg.trainer.train_strategy:
            name += f"_beta={cfg.trainer.dpo_beta}_samples={cfg.trainer.samples_per_train_question}"
        name += f"_name={cfg.trainer.model_name}"
        
    return name
