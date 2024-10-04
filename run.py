from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from omegaconf import OmegaConf, MISSING
from hydra import main as hydra_main
from hydra.utils import instantiate
import os
from typing import Optional, Any, List, Dict
from dataclasses import dataclass, field
import random
import numpy as np
from  star.trainer.config import TrainerConfig
OmegaConf.register_new_resolver("eval", eval)
cs = ConfigStore.instance()
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
    cpus_per_task: int|None = 6 # type: ignore
    gpus_per_node: int|None = None
    tasks_per_node: int =  1
    mem_gb: int|None =  None
    nodes: int = 1
    _target_: str = "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    partition: str|None = "kargo-lab" # overcap
    qos: str|None = "short"
    exclude: str|None = "major,crushinator,nestor,voltron,xaea-12,samantha"
    additional_parameters: Dict[str, Any] = field(default_factory=lambda: {"gpus": "a40:1", "requeue": True})
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
    run_type: str|None = "train"
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



@hydra_main(version_base=None, config_name='RunConfig')
def my_app(cfg: RunConfig) -> None:
    # for running different system prompt configurations?
    import torch
    # lazy import for fast hydra command line utility.

    # if cfg.device == "mps":
    #     assert torch.backends.mps.is_available(), "mps must be available for mps device spec"
    # elif cfg.device == "cuda":
    #     assert torch.cuda.is_available(), "cuda must be available for cuda device"
    # else:
    #     raise Exception(f"device {cfg.device} cannot be specified. No cpu because don't like slow on accident.")

    cfg.node_name = os.getenv("SLURMD_NODENAME", "NO_NODE_NAME_FOUND")
    # ppconfig = cfg.preprocessing
    cfg.logger.name = f"{cfg.run_type}"
    # if cfg.trainer.model.mlm_probability != 0.15 or cfg.trainer.model.word_masking:
    #     cfg.logger.name += f"_mratio={cfg.trainer.model.mlm_probability}_wmask={cfg.trainer.model.word_masking}"
    # if hasattr(cfg.trainer.model.maskedlm, "combine_layer") and cfg.trainer.model.maskedlm.combine_layer >= 0:  # type: ignore
    #     cfg.logger.name += f"_comb_exp={(cfg.trainer.model.maskedlm.combine_layer, cfg.trainer.model.maskedlm.expand_layer)}" # type: ignore
    # if ppconfig.prob_combine != 0.0:
    #     cfg.logger.name += f"_pp_cdms={(ppconfig.prob_combine, ppconfig.prob_delete, ppconfig.prob_move, ppconfig.prob_swap)}"
    # if cfg.trainer.model.supervised_weight != 1 or cfg.trainer.model.so_weight != 1 or cfg.trainer.model.mlm_weight != 1 or cfg.trainer.model.tlm_weight != 1 or cfg.trainer.model.psi_weight != 1:
    #     cfg.logger.name += f"_ws=({cfg.trainer.model.supervised_weight:.3},{cfg.trainer.model.so_weight:.3},{cfg.trainer.model.mlm_weight:.3},{cfg.trainer.model.tlm_weight:.3},{cfg.trainer.model.psi_weight:.3})"
    # if "max_softmax" in cfg.trainer.model.coverage_encouragement_type:
    #     cfg.logger.name += f"_maxCvgTempStartEnd={(cfg.trainer.model.max_softmax_temperature_start, cfg.trainer.model.max_softmax_temperature_end)}_cvgW={cfg.trainer.model.coverage_weight}_cvgType={cfg.trainer.model.coverage_encouragement_type}"
    cfg.logger.name += f"_ds={cfg.trainer.dataset_name}_lr={cfg.trainer.agent.model_handler.optimizer_partial.lr}_seed={cfg.seed}"
    # _cosSim={cfg.trainer.model.cosine_sim}_simTemp={cfg.trainer.model.sim_func_temp}_thresh={cfg.trainer.model.threshold}_divByLen={cfg.trainer.model.div_by_len}_entropyLoss={cfg.trainer.model.entropy_loss}
    # import ipdb; ipdb.set_trace()

    isMultirun = "num" in HydraConfig.get().job # type: ignore # for implicit debugging when launching a job without -m.  
    # cfg.datasetloaders.num_workers =  3 if not isMultirun else HydraConfig.get().launcher.cpus_per_task - 3
    cfg_for_logging = OmegaConf.to_container(cfg)
    seed = cfg.seed # the preprocessing that occurs in the dataset objects for corruptions need the seeds set for consistency.
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger = instantiate(cfg.logger)
    # import ipdb; ipdb.set_trace()
    partial_trainer = instantiate(cfg.trainer)
    trainer = partial_trainer(logger=logger)
    # using with wandb.init is a hack to get wandb to create a new run for every -m sweep. otherwise it concats them to one run.
    # if cfg.run_type == "val":
    #     # python run.py run_type=val trainer.gen_batch_size=32 "trainer.dataset_name=mmlu_YO-NG" "+current_global_step=1" "+model_checkpoint_to_eval=/nethome/jbjorner3/dev/hallucination-fun/star/star_runs/train_2024-10-01/03-06-34_0/checkpoint_1"
    #     model_checkpoint_to_eval = cfg.model_checkpoint_to_eval
    #     current_global_step = cfg.current_global_step 
    #     trainer.val(cfg, current_global_step, model_checkpoint_to_eval)
    #     # trainer.val(cfg.model_checkpoint_to_eval, asynchronous=False)
    #     return
    with trainer.logger.init(config=cfg_for_logging) as run:
        trainer.fit(cfg)
    # if isinstance(ret, dict) and "eval_jaen_AER" in ret:
    #     return ret["eval_jaen_AER"]



if __name__ == "__main__":
    my_app()