from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from hydra import main as hydra_main
from hydra.utils import instantiate
import os
import random
import numpy as np
from config import RunConfig, get_run_name_from_cfg
from dotenv import load_dotenv

# Load the environment variables from .env
load_dotenv()

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
    cfg.logger.name = get_run_name_from_cfg(cfg)
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
    # if 'dpo' in cfg.trainer.train_strategy:
    #     cfg.logger.name += f"_beta={cfg.trainer.dpo_beta}_samples={cfg.trainer.samples_per_train_question}"
    
    # cfg.logger.name += f"_ts={cfg.trainer.train_strategy}_ds={cfg.trainer.dataset_name}_lr={cfg.trainer.agent.model_handler.optimizer_partial.lr}_wd={cfg.trainer.agent.model_handler.optimizer_partial.weight_decay}_lora={cfg.trainer.agent.model_handler.lora_r}_gaccum={cfg.trainer.train_gradient_accumulation_steps}_warmup={cfg.trainer.agent.model_handler.warmup}_gnorm={cfg.trainer.max_grad_norm}_seed={cfg.seed}"
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