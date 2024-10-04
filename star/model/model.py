import torch
from torch.optim import Optimizer # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
import transformers
from unsloth import FastLanguageModel
from typing import Tuple, Any
from abc import ABC, abstractmethod

class BaseModelHandler(ABC):
    @abstractmethod
    def get_model_tokenizer(self, checkpoint=None) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerFast]:
        ...
    @abstractmethod
    def get_optimizer_scheduler_max_steps(self, model, steps_per_epoch) -> Tuple[Optimizer, LRScheduler, int]:
        ...
    @abstractmethod
    def prepare_for_inference(self, model) -> transformers.PreTrainedModel:
        ...
    @abstractmethod
    def prepare_for_training(self, model) -> transformers.PreTrainedModel:
        ...
    @abstractmethod
    def get_tokenizer(self, checkpoint) -> transformers.PreTrainedTokenizerFast:
        ...
    @abstractmethod
    def save(self, model, checkpoint):
        ...

class UnslothModelHandler(BaseModelHandler):
    def __init__(self, model_name, 
                 max_seq_length,
                 epochs,
                 max_steps,
                 warmup,
                 scheduler_str: str,
                 optimizer_partial: Any):
        assert max_steps != -1 or epochs != -1, f"one of max_steps or epochs must be non negative, but both are -1"
        self.epochs = epochs
        self.max_steps = max_steps
        self.warmup = warmup
        self.scheduler_str = scheduler_str
        self.optimizer_partial = optimizer_partial
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        assert self.model_name == "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct is the only supported model until I get the peft working for others. I can special case it or I could do something with accepting lists in the configs"
    def get_model_tokenizer(self, checkpoint=None) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerFast]:
        
        peft_function = lambda x: x
        if checkpoint is None:
            checkpoint = self.model_name
            peft_function = lambda model: FastLanguageModel.get_peft_model(
                model,
                r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 16,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                # use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=self.max_seq_length,
            dtype=torch.bfloat16,
        )
        model = peft_function(model)
        return model, tokenizer
    
    def get_optimizer_scheduler_max_steps(self, model, steps_per_epoch) -> Tuple[Optimizer, LRScheduler, int]:
        if self.max_steps == -1:
            max_steps = self.epochs * steps_per_epoch
        else:
            max_steps = self.max_steps
        warmup = self.warmup
        optimizer = self.optimizer_partial(model.parameters())
        if self.scheduler_str == "lambda":
            scheduler = LambdaLR(optimizer, lambda step: step/warmup if warmup > step else 1 - (step-warmup)/ (max_steps-warmup))
        else:
            raise Exception(f"scheduler {self.scheduler_str} is not implemented yet in model.py")
        return optimizer, scheduler, max_steps
    def prepare_for_inference(self, model):
        return FastLanguageModel.for_inference(model)
    def prepare_for_training(self, model):
        return FastLanguageModel.for_training(model)
    def get_tokenizer(self, checkpoint) -> transformers.PreTrainedTokenizerFast:
        if checkpoint is None:
            checkpoint = self.model_name
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        return tokenizer
    def save(self, model: transformers.PreTrainedModel, checkpoint):
        model.save_pretrained(checkpoint)