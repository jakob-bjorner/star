import torch
from torch.optim import Optimizer # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
import transformers
from unsloth import FastLanguageModel
from typing import Tuple, Any
from abc import ABC, abstractmethod
import os

class BaseModelHandler(ABC):
    @abstractmethod
    def get_model_tokenizer(self, checkpoint: str) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerFast]:
        ...
    @abstractmethod
    def get_optimizer_scheduler_max_steps(self, model, max_steps) -> Tuple[Optimizer, LRScheduler]:
        ...
    @abstractmethod
    def prepare_for_inference(self, model) -> transformers.PreTrainedModel:
        ...
    @abstractmethod
    def prepare_for_training(self, model) -> transformers.PreTrainedModel:
        ...
    @abstractmethod
    def get_tokenizer(self, checkpoint:str) -> transformers.PreTrainedTokenizerFast:
        ...
    @abstractmethod
    def save(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizerFast,  checkpoint):
        ...

class UnslothModelHandler(BaseModelHandler):
    def __init__(self,
                 max_seq_length,
                 checkpoint_base,
                #  epochs,
                #  max_steps,
                 warmup,
                 scheduler_str: str,
                 optimizer_partial: Any,
                 lora_r: int):
        
        self.warmup = warmup
        self.scheduler_str = scheduler_str
        self.optimizer_partial = optimizer_partial
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.checkpoint_base = checkpoint_base
        # assert self.model_name == "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct is the only supported model until I get the peft working for others. I can special case it or I could do something with accepting lists in the configs"
    def get_model_tokenizer(self, checkpoint: str) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerFast]:
        peft_function = lambda x: x
        if not os.path.exists(checkpoint) and checkpoint == "unsloth/Meta-Llama-3.1-8B-Instruct":
            peft_function = lambda model: FastLanguageModel.get_peft_model(
                model,
                r = self.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = self.lora_r,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                # use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
        model, _ = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=self.max_seq_length,
            dtype=torch.bfloat16,
        )
        model = peft_function(model)
        tokenizer = self.get_tokenizer("")
        return model, tokenizer
    
    def get_optimizer_scheduler_max_steps(self, model, max_steps) -> Tuple[Optimizer, LRScheduler]:
        warmup = self.warmup
        optimizer = self.optimizer_partial(model.parameters())
        if self.scheduler_str == "lambda":
            if max_steps == 0:
                scheduler = LambdaLR(optimizer, lambda step: 1) # because no steps will be taken, we just return constant.
            elif warmup == 0:
                scheduler = LambdaLR(optimizer, lambda step: 1 - (step / max_steps))
            else:
                scheduler = LambdaLR(optimizer, lambda step: (step+1)/warmup if warmup >= (step+1) else 1 - (step+1-warmup) / (max_steps-warmup))
        elif self.scheduler_str == "const":
            scheduler = LambdaLR(optimizer, lambda step: 1)
        else:
            raise Exception(f"scheduler {self.scheduler_str} is not implemented yet in model.py")
        return optimizer, scheduler
    def prepare_for_inference(self, model):
        return FastLanguageModel.for_inference(model)
    def prepare_for_training(self, model):
        return FastLanguageModel.for_training(model)
    def get_tokenizer(self, checkpoint: str) -> transformers.PreTrainedTokenizerFast:
        checkpoint = self.checkpoint_base
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        return tokenizer
    def save(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizerFast, checkpoint):
        if self.checkpoint_base == "unsloth/Meta-Llama-3.1-8B-Instruct": # check if peft:
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
        else:
            model.save_pretrained(checkpoint, safe_serialization=False)
            tokenizer.save_pretrained(checkpoint)
        