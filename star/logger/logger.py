from abc import ABC, abstractmethod
from pprint import pprint
import wandb
from datetime import datetime
from typing import Dict, Any
import contextlib
import os

class BaseLogger(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError()
    @abstractmethod
    def init(self, config: Dict, **kwargs) -> contextlib.AbstractContextManager: 
        # returning this type of object allows for nice cleanup in multirun cases for wandb, but requires to run: with init(): \n\t body...
        raise NotImplementedError()
    @abstractmethod
    def log(self, log_dict, **kwargs):
        raise NotImplementedError()
    def watch(self, *args, **kwargs) -> Any:
        pass
    def Image(self, *args, **kwargs) -> Any:
        pass


class BasicPrintLogger(BaseLogger):
    def __init__(self, **kwargs):
        print(f"__init__ {datetime.now()}", end=' ')
        self.output_file = kwargs.get("output_dir", None) 
        if self.output_file is not None:
            self.output_file = os.path.join(self.output_file, "logger.txt")
        self.output_file
        pprint(kwargs)
    def init(self, config: Dict, **kwargs) -> contextlib.AbstractContextManager:
        print(f"init: {datetime.now()}", end=' ')
        if self.output_file is not None:
            with open(self.output_file, "a") as fout:
                fout.write(f"init: {datetime.now()} ({config}, {kwargs})\n")
        pprint((config, kwargs))
        return contextlib.nullcontext()
    def log(self, *args, **kwargs):
        print(f"log: {datetime.now()}", end=' ')
        if self.output_file is not None:
            with open(self.output_file, "a") as fout:
                fout.write(f"log: {datetime.now()} ({args}, {kwargs})\n")
        pprint((args, kwargs))
    def Image(self, *args, **kwargs):
        return (args, kwargs)


class WandBLogger(BaseLogger):
    # TODO: if ever I need to resume experiment look into adding id as a kwarg, in wandb: https://github.com/ashleve/lightning-hydra-template/blob/main/configs/logger/wandb.yaml
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.wandb = wandb
    def init(self, config: Dict, **kwargs) -> contextlib.AbstractContextManager:
        self.kwargs.update(kwargs)
        pprint((config, kwargs))
        self.run = self.wandb.init(**self.kwargs, config=config)
        assert isinstance(self.run, contextlib.AbstractContextManager)
        return self.run
    def log(self, *args, **kwargs):
        return self.wandb.log(*args, **kwargs)
    def watch(self, *args, **kwargs):
        return self.wandb.watch(*args, **kwargs)
    def Image(self, *args, **kwargs):
        return self.wandb.Image(*args, **kwargs)



class CustomWandBLogger(WandBLogger):
    # allows for asynchronous logging of elements when they may take longer than normal to generate.
    def init(self, config: Dict, **kwargs):
        self.run = super().init(config, **kwargs)
        self.wandb.define_metric("custom_step")
        self.wandb.define_metric("evaluate_*", step_metric='custom_step') # allows for asynchronous logging of eval events.
        return self.run



