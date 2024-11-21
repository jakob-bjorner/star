from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Any, List, Optional
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
    train_batch_size: int = 16
    train_gradient_accumulation_steps: int = 1
    num_workers: int = 3
    dataset_name: str = "mmlu_YO-NG" # mmlu
    system_message: str = "You are a helpful assistant"
    preamble_to_question: str = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
    samples_per_train_question: int = 1 # must be at least 2 for dpo 
    # I can decide to create two training bodies or I can decide to keep only one trainer body? benefit if I keep just one is that I only have to make the changes one place, and don't have to port them around, but can make errors...
    # I think I'll keep just one for now, because I just need to integrate two implementations, and doesn't seem so hard. DPO and SFT Reinforcement. In the future, I will want to try different inference techniques (like Beamsearch maybe MCTS?), and possibly also meta prompts.
    # beam search option should just be passed into the generation function, and should be agnostic to dpo or reinforce,
    # MCTS could involve an architectural modification, but then would just be inference modification? Not exactly, the training would be modified, and the return of the generation function would be substantially changed.
    # meta prompts would involve a different inference technique as well, but nothing related to training, so just fine?
    # PPO would involve architectural modification, and needing to have some sort of training for the value head, and other things like this?
    # other algorithms like many samplings idea for finding optimal probability sequence  would modify inference, with substantial communication between the loss function, and the generated samples?
    train_strategy: str = "reinforce" # either dpo or reinforce # I could change this to an object which contains the specs necessary for the strategy to be executed, but whatever for now.
    dpo_beta: float = 0.1
    skip_first_validation_loop: bool = False
    interal_epochs: int = 1
    preload_data_for_training: Optional[str] = None
    preload_data_for_val: Optional[str] = None
    log_every_n_steps: int = 1
    val_every_n_steps: int = 60
    max_epochs: int = 10
    max_grad_norm: float = 1.0
    do_eval: bool = True
    _partial_: bool = True
    # seed: int = 0 # for numbers 0 or lower, the seed is random. This is for easy testing.
ConfigStore.instance().store(name="TrainerConfig", node=TrainerConfig, group="trainer")



# eval gen acc 13.0, train 15.6  for bellow prompt: and you get 65 for train and similar for eval with english setting.
# Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.
# eval gen acc 15.8, train 16.8  for bellow prompt:
# Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step in English before answering.
# eval gen acc 22.2, train 21.9
# Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Translate the question and answers to English to begin with, and think step by step before answering.