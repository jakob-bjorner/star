from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from star.logger.logger import BaseLogger
from typing import Any, Callable, Iterable
import random
import numpy as np
from torch.optim.lr_scheduler import LRScheduler
import tqdm
import os
from star.dataset.dataset import RationalDataset, get_reinforce_collate_fn, get_generation_collate_fn, get_train_val_test_dataset
from math import ceil
from torch.utils.data import DataLoader
from star.agent.agent import Agent, generate_async_sample_jobs_distributed, wait_for_submitted_jobs, GeneratedSamples
import gc
def acc(iterator):
    return sum(iterator) / len(iterator) 

class Trainer:
    ''' AwesomeAlignTrainer for replicating awesomealign. Then will want to extend this base class for other purposes and experiments. '''
    def __init__(self, 
                 agent: Agent,
                 logger: BaseLogger, 
                 device: str,
                 output_dir: str,
                 gen_batch_size: int,
                 single_batch: bool,
                 reinforce_batch_size: int,
                 num_workers: int,
                 dataset_name: str
                 ):
        self.agent = agent
        self.logger = logger
        self.device = device
        self.output_dir = output_dir
        self.gen_batch_size = gen_batch_size
        self.single_batch = single_batch
        self.reinforce_batch_size = reinforce_batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name

        self.log_every_n_steps = 1
        self.val_every_n_steps = 30
        self.max_epochs = 10
        self.generation_temperature = 1.0 # validation temperature is always 0.0. (could use a seed for consistency? the temperature is for sampling non argmax actions in the random case, so sampling with temperature would make sense for the val in this case, but we should sample with temp=0 to be comparable to other methods which do so.)

    def initialize_train(self, config):

        self.config = config
        self.current_star_iteration = 0
        self.current_global_step = 0
        
        gen_batch_size = self.gen_batch_size
        
        train_dataset, val_dataset, test_datset = get_train_val_test_dataset(self.dataset_name)
        self.generation_train_dataloader = DataLoader(train_dataset, batch_size=gen_batch_size, collate_fn=get_generation_collate_fn(self.agent.model_handler.get_tokenizer(None)), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore
        self.generation_val_dataloader = DataLoader(val_dataset, batch_size=gen_batch_size, collate_fn=get_generation_collate_fn(self.agent.model_handler.get_tokenizer(None)), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore
        self.latest_checkpoint = None
        self.async_job_info = None

    def train_loop(self, progress_bar: tqdm.tqdm, formatted_string_postfix: str):
        # for star, this is a very simple training loop, we iterate through every correct sample performing sft on them.
        # for other RL methods like MCTS variations, there can be more complicated ideas of a training loop. some times you want to sample online data though, and this can be done in the training loop in the general case.
        # perform sampling every couple steps. For STaR, this is every step.
        gc.collect()
        torch.cuda.empty_cache() # don't want to accidently over allocate memory.
        
        log_dict = {}
        generation_output_dir = os.path.join(self.output_dir, f"train_{self.current_global_step}")
        jobs = generate_async_sample_jobs_distributed(output_dir=generation_output_dir, 
                                                      dataloader=self.generation_train_dataloader, 
                                                      model_handler=self.agent.model_handler, 
                                                      checkpoint=self.latest_checkpoint,
                                                      temperature=self.generation_temperature, 
                                                      single_batch=self.single_batch)
        sampled_completitions = wait_for_submitted_jobs(jobs, generation_output_dir)
        self.sync_val_jobs() # should be done generating vals if finish generating the train.

        generation_acc = sum(sampled_completitions.scores) / len(sampled_completitions.scores)
        log_dict["train_gen_acc"] = generation_acc
        log_dict.update({f"train_gen_{subject}_acc": acc(sampled_completitions.get_subset_by_subject(subject).scores) for subject in sampled_completitions.subjects})
        log_dict["custom_step"] = self.current_global_step
        print(log_dict)
        self.logger.log(log_dict)
        log_dict = dict()
        # 2  gigs, then to 6.4 on the model optimizer load
        steps_per_epoch = ceil(sum(sampled_completitions.scores) / self.reinforce_batch_size)
        model, tokenizer, optimizer, scheduler, local_max_steps = self.agent.get_new_model_tokenizer_optimizer_scheduler_max_steps(steps_per_epoch)
        reinforce_dataset = RationalDataset(sampled_completitions.raw_responses, sampled_completitions.scores, sampled_completitions.prompted_questions)
        dl_reinforce = DataLoader(reinforce_dataset, batch_size=self.reinforce_batch_size, collate_fn=get_reinforce_collate_fn(tokenizer), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda")
        # peform gradient step(s) on the data. For STaR, you filter out only the ones you got right in the last generation step, and train on those.
        #   STaR also has you train on the base model rather than iterate through the model. Kind of similar to a KL constraint.
        # typically the optimizer would be stored in the trainer, and 
        progress_bar.set_postfix_str(formatted_string_postfix.format_map({"acc": generation_acc}))
        
        def fixed_steps_iterator(iterator, total):
            i = 0
            while i < total:
                for data in iterator:
                    yield data
                    i += 1
                    if i == total:
                        return

        for data_i, data in enumerate(tqdm.tqdm(fixed_steps_iterator(dl_reinforce, local_max_steps), total=local_max_steps)):
            optimizer.zero_grad()
            losses = self.agent.train_step(model, data)
            loss = losses.loss
            loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()
            scheduler.step()
            last_lr = scheduler.get_last_lr()[-1]
            if self.current_global_step % self.log_every_n_steps == 0 or data_i == local_max_steps-1 or data_i == 0:
                log_dict["loss"] = loss.item()
                log_dict["lr"] = last_lr
                log_dict = {("train_" + key): value for key, value in log_dict.items()}
                log_dict["custom_step"] = self.current_global_step
                print(log_dict)
                self.logger.log(log_dict)
                log_dict = dict()
            self.current_global_step += 1
            if self.current_global_step % self.val_every_n_steps == 0 or data_i == local_max_steps - 1:
                self.sync_val_jobs()
                # must not be technically possible for this eval to be called before the other has finished
                generation_output_dir = os.path.join(self.output_dir, f"val_{self.current_global_step}")
                eval_model_checkpoint = os.path.join(generation_output_dir, f"eval_model_checkpoint")
                self.agent.save_model(model, eval_model_checkpoint)
                self.validation_loop(eval_model_checkpoint)
            if self.single_batch:
                break
        self.latest_checkpoint = os.path.join(self.output_dir, f"checkpoint_{self.current_star_iteration}")
        self.agent.save_model(model, self.latest_checkpoint)
        progress_bar.update()

    def fit(self, config):
        self.initialize_train(config)
        self.validation_loop(None)
        formatted_string_postfix = "acc: {acc:3.5f}"
        progress_bar = tqdm.trange(self.max_epochs, postfix=formatted_string_postfix.format_map({"acc": 0}))
        while self.current_star_iteration < self.max_epochs:
            self.current_star_iteration += 1
            self.train_loop(progress_bar, formatted_string_postfix)
        self.sync_val_jobs()
        self.test_loop()
        # self.save_checkpoint("post_train.pt")
        # return self.last_eval_log_dict
    def sync_val_jobs(self):
        if self.async_job_info:
            async_jobs, async_generation_output_dir, async_global_step, asynchronous_report = self.async_job_info
            self.async_job_info = None
            sampled_completitions = wait_for_submitted_jobs(async_jobs, async_generation_output_dir)
            asynchronous_report(sampled_completitions, async_global_step)
    def validation_loop(self, model_checkpoint_to_eval, asynchronous=True):
        with torch.no_grad():
            def asynchronous_report(sampled_completitions: GeneratedSamples, custom_step):
                log_dict_total = dict()
                # MUST HAVE evaluate_* for logging with custom_step!
                log_dict_total["evaluate_gen_acc"] = sum(sampled_completitions.scores) / len(sampled_completitions.scores)
                log_dict_total.update({f"evaluate_gen_{subject}_acc": acc(sampled_completitions.get_subset_by_subject(subject).scores) for subject in sampled_completitions.subjects})
                log_dict_total["custom_step"] = custom_step
                print({k: v for k, v in log_dict_total.items() if "plotted" not in k}, self.current_global_step)
                # self.last_eval_log_dict = log_dict_total
                self.logger.log(log_dict_total)

            # sampling step from star.
            generation_output_dir = os.path.join(self.output_dir, f"val_{self.current_global_step}")
            jobs = generate_async_sample_jobs_distributed(output_dir=generation_output_dir, 
                                                          dataloader=self.generation_val_dataloader, 
                                                          model_handler=self.agent.model_handler, 
                                                          checkpoint=model_checkpoint_to_eval,
                                                          temperature=0.0, 
                                                          single_batch=self.single_batch)
            self.sync_val_jobs()
            if asynchronous:
                self.async_job_info = (jobs, generation_output_dir, self.current_global_step, asynchronous_report)
            else:
                sampled_completitions = wait_for_submitted_jobs(jobs, generation_output_dir)
                asynchronous_report(sampled_completitions, self.current_global_step)

    def test_loop(self):
        # should run alignment for all languages on the test sets.
        pass
    
    def val(self, config, current_global_step, model_checkpoint):
        self.initialize_train(config)

        # self.config = config
        # self.current_star_iteration = 0
        self.current_global_step = current_global_step
        self.validation_loop(model_checkpoint, asynchronous=False)
        
        # gen_batch_size = self.gen_batch_size
        
        # train_dataset, val_dataset, test_datset = get_train_val_test_dataset(self.dataset_name)
        # self.generation_train_dataloader = DataLoader(train_dataset, batch_size=gen_batch_size, collate_fn=get_generation_collate_fn(self.agent.model_handler.get_tokenizer(None)), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore
        # self.generation_val_dataloader = DataLoader(val_dataset, batch_size=gen_batch_size, collate_fn=get_generation_collate_fn(self.agent.model_handler.get_tokenizer(None)), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore
        # self.latest_checkpoint = None
        # self.async_job_info = None
        

