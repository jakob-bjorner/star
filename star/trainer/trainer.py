from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from star.logger.logger import BaseLogger
from typing import Any, Callable, Iterable, Optional
import random
import numpy as np
from torch.optim.lr_scheduler import LRScheduler
import tqdm
import os
from star.dataset.dataset import RationalDataset, get_reinforce_collate_fn, get_generation_collate_fn, get_train_val_test_dataset, DPODataset, get_dpo_collate_fn
from math import ceil
from torch.utils.data import DataLoader
from star.agent.agent import Agent, generate_async_sample_jobs_distributed, generate_async_nll_jobs_distributed, wait_for_submitted_generation_jobs, wait_for_submitted_nll_jobs, GeneratedSamples
import pandas as pd
import gc
def acc(iterator):
    return sum(iterator) / len(iterator) 
def get_sample_completitions_from_csv(csv_file):
    data_in_lists = pd.read_csv(csv_file).replace(np.nan, None).to_dict('list')
    data_in_lists = {str(k): v for k, v in data_in_lists.items()}
    sampled_completitions = GeneratedSamples(**data_in_lists) 
    return sampled_completitions
class Trainer:
    ''' AwesomeAlignTrainer for replicating awesomealign. Then will want to extend this base class for other purposes and experiments. '''
    def __init__(self, 
                 agent: Agent,
                 logger: BaseLogger, 
                 device: str,
                 output_dir: str,
                 gen_batch_size: int,
                 single_batch: bool,
                 train_batch_size: int,
                 train_gradient_accumulation_steps: int,
                 num_workers: int,
                 dataset_name: str,
                 system_message: str,
                 preamble_to_question: str,
                 meta_prompt_to_preamble: str,
                 samples_per_train_question: int,
                 train_strategy: str,
                 dpo_beta: float,
                 skip_first_validation_loop: bool,
                 interal_epochs: int, 
                 preload_data_for_training: Optional[str],
                 preload_data_for_val: Optional[str],
                #  log_every_n_steps: int,
                 val_every_n_steps: int, 
                 max_epochs: int,
                 max_grad_norm: float,
                 do_eval: bool,
                 do_acc_eval: bool,
                 model_name: str, 
                 ref_model_name: str,
                 shuffle_dataloader: bool,
                 debug_eval: bool,
                 num_train_questions: int,
                 reset_model_every_star_epoch: bool,
                 ):
        self.agent = agent
        self.logger = logger
        self.device = device
        self.output_dir = output_dir
        self.gen_batch_size = gen_batch_size
        self.single_batch = single_batch
        self.train_batch_size = train_batch_size
        self.train_gradient_accumulation_steps = train_gradient_accumulation_steps
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.system_message = system_message
        self.preamble_to_question = preamble_to_question + (" " + meta_prompt_to_preamble if len(meta_prompt_to_preamble) > 0 else "")
        self.samples_per_train_question = samples_per_train_question
        self.train_strategy = train_strategy
        self.dpo_beta = dpo_beta
        self.skip_first_validation_loop = skip_first_validation_loop
        self.preload_data_for_training = preload_data_for_training
        self.preload_data_for_val = preload_data_for_val

        # assert max_steps != -1 or epochs != -1, f"one of max_steps or epochs must be non negative, but both are -1"
        self.interal_epochs = interal_epochs
        # self.max_steps = max_steps

        # self.log_every_n_steps = log_every_n_steps
        self.val_every_n_steps = val_every_n_steps
        self.max_epochs = max_epochs
        self.max_grad_norm = max_grad_norm
        self.do_eval = do_eval
        self.do_acc_eval = do_acc_eval
        self.model_name = model_name
        self.ref_model_name = ref_model_name
        self.shuffle_dataloader = shuffle_dataloader
        self.debug_eval = debug_eval
        self.num_train_questions = num_train_questions
        self.reset_model_every_star_epoch = reset_model_every_star_epoch
        self.generation_temperature = 1.0 
        # self.subset_mmlu = 
        # validation temperature is always 0.0. (could use a seed for consistency? the temperature is for sampling non argmax actions in the random case, so sampling with temperature would make sense for the val in this case, but we should sample with temp=0 to be comparable to other methods which do so.)
        # the policy we are training might explicitly look at the expected reward under the sampling policy rather than this argmax policy, so maybe temperature, but definitely temperature if we sample more than one time.
        # assert self.log_every_n_steps % self.train_gradient_accumulation_steps == 0
        if self.do_acc_eval:
            assert self.do_eval, "do_eval must be true if do acc eval is true."
        if 'dpo' in self.train_strategy and self.interal_epochs > 0:
            assert self.samples_per_train_question > 1, "number of samples must be larger than 1 for DPO training if you are actually trying to train ie internal epochs > 0."
        if self.samples_per_train_question > 1:
            assert self.generation_temperature != 0.0, "generation temperature should be greater than 0.0 if you have to sample multiple responses per question."
        
    def initialize_train(self, config):
        self.config = config
        self.current_star_iteration = 0
        self.current_global_step = 0
        
        gen_batch_size = self.gen_batch_size
        
        train_dataset, val_dataset, test_datset = get_train_val_test_dataset(self.dataset_name, self.num_train_questions)
        self.generation_train_dataloader = DataLoader(train_dataset, batch_size=gen_batch_size, collate_fn=get_generation_collate_fn(self.agent.model_handler.get_tokenizer(self.model_name), system_message=self.system_message, preamble_to_question=self.preamble_to_question), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore
        self.generation_val_dataloader = DataLoader(val_dataset, batch_size=gen_batch_size, collate_fn=get_generation_collate_fn(self.agent.model_handler.get_tokenizer(self.model_name), system_message=self.system_message, preamble_to_question=self.preamble_to_question), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore
        self.latest_checkpoint = self.model_name
        self.async_job_info = None
        self.dataloader_to_measure_loss = None
    def get_dataloader(self, sampled_completitions: GeneratedSamples, is_training: bool):
        if 'dpo' in self.train_strategy:
            dataset = DPODataset(sampled_completitions.subjects, sampled_completitions.indices, sampled_completitions.prompted_questions, sampled_completitions.model_answers, sampled_completitions.raw_responses, sampled_completitions.scores)
            get_collate_fn = get_dpo_collate_fn
        elif 'reinforce' in self.train_strategy:
            dataset = RationalDataset(sampled_completitions.raw_responses, sampled_completitions.scores, sampled_completitions.prompted_questions, ("unique" in self.train_strategy))
            get_collate_fn = get_reinforce_collate_fn
        else:
            raise Exception(f"Invalid option for train_strategy in Trainer {self.train_strategy}")
        if is_training:
            batches_in_dataloader = ceil(len(dataset) / self.train_batch_size)
            return batches_in_dataloader, DataLoader(dataset, shuffle=self.shuffle_dataloader, batch_size=self.train_batch_size, collate_fn=get_collate_fn(self.agent.model_handler.get_tokenizer(self.model_name)), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda")
        else:
            return DataLoader(dataset, batch_size=ceil(self.gen_batch_size / 6), collate_fn=get_collate_fn(self.agent.model_handler.get_tokenizer(self.model_name)), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda")
    def train_loop(self, progress_bar: tqdm.tqdm, formatted_string_postfix: str):
        # for star, this is a very simple training loop, we iterate through every correct sample performing sft on them.
        # for other RL methods like MCTS variations, there can be more complicated ideas of a training loop. some times you want to sample online data though, and this can be done in the training loop in the general case.
        # perform sampling every couple steps. For STaR, this is every step.
        gc.collect()
        torch.cuda.empty_cache() # don't want to accidently over allocate memory.

        log_dict = {}
        generation_output_dir = os.path.join(self.output_dir, f"train_{self.current_global_step}")
        jobs_total = []
        if self.preload_data_for_training is not None and self.current_global_step == 0:
            sampled_completitions = get_sample_completitions_from_csv(self.preload_data_for_training)
            generation_acc = 0.0
        else:
            for _ in range(self.samples_per_train_question):
                jobs = generate_async_sample_jobs_distributed(output_dir=generation_output_dir, 
                                                            dataloader=self.generation_train_dataloader, 
                                                            model_handler=self.agent.model_handler, 
                                                            checkpoint=self.latest_checkpoint,
                                                            temperature=self.generation_temperature, 
                                                            single_batch=self.single_batch,
                                                            debug_eval=False) # not eval, but if I want to debug train, then should create another flag.
                jobs_total.extend(jobs)
            sampled_completitions = wait_for_submitted_generation_jobs(jobs_total, generation_output_dir)

            generation_acc = sum(sampled_completitions.scores) / len(sampled_completitions.scores)
            log_dict["train_gen_acc"] = generation_acc
            log_dict.update({f"train_gen_{subject}_acc": subject_acc for subject, subject_acc in sampled_completitions.get_accuracy_by_subject().items()})
            log_dict["custom_step"] = self.current_global_step
            print(log_dict)
            self.logger.log(log_dict)
            log_dict = dict()
        
        self.sync_val_jobs() # should be done generating vals if finish generating the train.
        
        # 2 gigs, then to 6.4 on the model optimizer load
        batches_in_dataloader, train_dataloader = self.get_dataloader(sampled_completitions, is_training=True)
        interal_max_steps = self.interal_epochs * batches_in_dataloader 
        
        model, tokenizer, optimizer, scheduler = self.agent.get_new_model_tokenizer_optimizer_scheduler_max_steps(interal_max_steps // self.train_gradient_accumulation_steps, 
                                                                                                                  self.model_name if (self.reset_model_every_star_epoch or self.latest_checkpoint is None) else self.latest_checkpoint)
        
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
            return

        accum_losses = []
        grad_norm = None
        last_lr = None
        for data_i, data in enumerate(tqdm.tqdm(fixed_steps_iterator(train_dataloader, interal_max_steps), total=interal_max_steps)):
            data.to('cuda')
            with torch.amp.autocast('cuda', dtype = torch.bfloat16): # type: ignore
                if "dpo" in self.train_strategy:
                    if "ref_model" not in locals():
                        ref_model, _ = self.agent.model_handler.get_model_tokenizer(self.ref_model_name)
                        ref_model = self.agent.model_handler.prepare_for_training(ref_model)
                    else:
                        ref_model = locals()['ref_model']
                    losses = self.agent.dpo_train_step(model, ref_model, data, self.dpo_beta) # 

                elif "reinforce" in self.train_strategy:
                    losses = self.agent.reinforce_train_step(model, data)
                else:
                    raise Exception(f"invalid option for train_strategy in Trainer {self.train_strategy}")
                accum_losses.append({k: v.detach().item() for k, v in losses.__dict__.items()})
                loss = losses.loss / self.train_gradient_accumulation_steps
                loss.backward()

                if (self.train_gradient_accumulation_steps == 1 or data_i != 0) and data_i % self.train_gradient_accumulation_steps == 0 :
                    if self.max_grad_norm > 0.0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    last_lr = scheduler.get_last_lr()[-1]
                    scheduler.step()
                    log_dict.update({k: np.mean([accum_loss_el[k] for accum_loss_el in accum_losses]).item() for k in accum_losses[0].keys()})
                    if grad_norm is None:
                        grad_norm = torch.linalg.vector_norm(torch.concat([torch.linalg.vector_norm(g.grad).flatten() for g in model.parameters() if hasattr(g, "grad") and g.grad is not None]))
                    log_dict['grad_norm'] = grad_norm.item()
                    log_dict["lr"] = last_lr
                    log_dict = {("train_" + key): value for key, value in log_dict.items()}
                    log_dict["custom_step"] = self.current_global_step
                    print(log_dict)
                    self.logger.log(log_dict)
                    log_dict = dict()
                    accum_losses = []
                    self.current_global_step += 1
                    if self.current_global_step % self.val_every_n_steps == 0 or data_i == interal_max_steps-1 or self.single_batch:
                        self.sync_val_jobs()
                        # must not be technically possible for this eval to be called before the other has finished
                        generation_output_dir = os.path.join(self.output_dir, f"val_{self.current_global_step}")
                        eval_model_checkpoint = os.path.join(generation_output_dir, f"eval_model_checkpoint")
                        self.agent.save_model(model, tokenizer, eval_model_checkpoint)
                        if interal_max_steps - 1 == data_i or self.single_batch:
                            self.validation_loop(eval_model_checkpoint, True)
                        else:
                            self.validation_loop(eval_model_checkpoint, False)
                if self.single_batch:
                    break
        self.latest_checkpoint = os.path.join(self.output_dir, f"checkpoint_{self.current_star_iteration}")
        self.agent.save_model(model, tokenizer, self.latest_checkpoint)
        progress_bar.update()

    def fit(self, config):
        self.initialize_train(config)
        if self.preload_data_for_val:
            sampled_completitions = get_sample_completitions_from_csv(csv_file=self.preload_data_for_val)
            dataloader = self.get_dataloader(sampled_completitions, is_training=False) 
            assert isinstance(dataloader, DataLoader)
            self.dataloader_to_measure_loss = dataloader
        if not self.skip_first_validation_loop:
            if self.preload_data_for_val:
                self.validation_loop(self.model_name, False)
            else:
                self.validation_loop(self.model_name, True)
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
            async_nll_jobs, save_dataset, async_generation_jobs, subset_indices_in_eval_dpo, async_generation_output_dir, async_global_step, asynchronous_report = self.async_job_info
            self.async_job_info = None
            if self.do_acc_eval:
                sampled_completitions = wait_for_submitted_generation_jobs(async_generation_jobs, async_generation_output_dir)
                if save_dataset:
                    dataloader_to_measure_loss = self.get_dataloader(sampled_completitions, is_training=False)
                    assert isinstance(dataloader_to_measure_loss, DataLoader)
                    self.dataloader_to_measure_loss = dataloader_to_measure_loss
            else:
                sampled_completitions = None
            if async_nll_jobs is not None:
                avg_nll_val = wait_for_submitted_nll_jobs(async_nll_jobs)
            else:
                avg_nll_val = None
                
            asynchronous_report(sampled_completitions, async_global_step, avg_nll_val, subset_indices_in_eval_dpo)
    def validation_loop(self, model_checkpoint_to_eval, save_gen_data_as_new_val_loss_data, asynchronous=True):
        """
        model_checkpoint_to_eval this can be None and if it is, we will just eval with the default model.
        save_gen_data_as_new_val_loss_data is a bool which says whether to change self.dataloader_to_measure_loss after the val dataset has been generated.
        """
        if self.do_eval is False:
            return
        with torch.no_grad():
            def asynchronous_report(sampled_completitions: Optional[GeneratedSamples], custom_step, avg_nll_val, subset_indices_in_eval_dpo):
                log_dict_total = dict()
                # MUST HAVE evaluate_* for logging with custom_step!
                if avg_nll_val is not None:
                    log_dict_total.update({f"evaluate_{key}": val for key, val in avg_nll_val.items()})
                if sampled_completitions is not None:
                    log_dict_total["evaluate_gen_acc"] = sum(sampled_completitions.scores) / len(sampled_completitions.scores)
                    log_dict_total.update({f"evaluate_gen_{subject}_acc": subject_acc for subject, subject_acc in sampled_completitions.get_accuracy_by_subject().items()})
                    if subset_indices_in_eval_dpo is not None:
                        subset_sampled_completitions = sampled_completitions.get_subset_by_indices(subset_indices_in_eval_dpo)
                        log_dict_total["evaluate_gen_acc_subset_in_loss"] = 0 if len(subset_sampled_completitions.scores) == 0 else sum(subset_sampled_completitions.scores) / len(subset_sampled_completitions.scores)
                        # log_dict_total.update({f"evaluate_gen_{subject}_acc": subject_acc for subject, subject_acc in sampled_completitions.get_accuracy_by_subject().items()})
                    
                log_dict_total["custom_step"] = custom_step
                print({k: v for k, v in log_dict_total.items() if "plotted" not in k}, custom_step)
                # self.last_eval_log_dict = log_dict_total
                self.logger.log(log_dict_total)

            # sampling step from star.
            generation_output_dir = os.path.join(self.output_dir, f"val_{self.current_global_step}")
            total_async_generation_jobs = []
            if self.do_acc_eval:
                if save_gen_data_as_new_val_loss_data:
                    for _ in range(self.samples_per_train_question): # this is especially applicable in the DPO case.
                        async_generation_jobs = generate_async_sample_jobs_distributed(output_dir=generation_output_dir, 
                                                                dataloader=self.generation_val_dataloader, 
                                                                model_handler=self.agent.model_handler, 
                                                                checkpoint=model_checkpoint_to_eval,
                                                                temperature=self.generation_temperature,  # Note this is different because we need responses to be different if we want to measure dpo loss on val set.
                                                                single_batch=self.single_batch,
                                                                debug_eval=self.debug_eval)
                        total_async_generation_jobs.extend(async_generation_jobs)
                else:
                    # no need to loop, they will be the same always.
                    async_generation_jobs = generate_async_sample_jobs_distributed(output_dir=generation_output_dir, 
                                                                dataloader=self.generation_val_dataloader, 
                                                                model_handler=self.agent.model_handler, 
                                                                checkpoint=model_checkpoint_to_eval,
                                                                temperature=0.0,  # Note this is different because we need responses to be different if we want to measure dpo loss on val set.
                                                                single_batch=self.single_batch,
                                                                debug_eval=self.debug_eval)
                    total_async_generation_jobs.extend(async_generation_jobs)
            subset_indices_in_eval_dpo = None
            if self.dataloader_to_measure_loss is not None:
                async_nll_jobs = generate_async_nll_jobs_distributed(output_dir=generation_output_dir, 
                                                                 dataloader=self.dataloader_to_measure_loss,
                                                                 agent=self.agent, 
                                                                 train_strategy=self.train_strategy,
                                                                 dpo_beta=self.dpo_beta,
                                                                 checkpoint=model_checkpoint_to_eval,
                                                                 single_batch=self.single_batch,
                                                                 ref_model_name=self.ref_model_name,
                                                                 debug_eval=self.debug_eval)
                if self.train_strategy == "dpo":
                    assert isinstance(self.dataloader_to_measure_loss.dataset, DPODataset)
                    subset_indices_in_eval_dpo = set(i[0] for i in self.dataloader_to_measure_loss.dataset.internal_data)
                # elif self.train_strategy == "reinforce": implicitly set to None because it is commented out.
                    # subset_indices_in_eval_dpo = set(self.dataloader_to_measure_loss.dataset.) # this isn't possible currently, and don't think it's worth doing right now as STAR's setting isn't the focus.

            else:
                async_nll_jobs = None

            self.sync_val_jobs()
            if asynchronous:
                self.async_job_info = (async_nll_jobs, save_gen_data_as_new_val_loss_data, total_async_generation_jobs, subset_indices_in_eval_dpo, generation_output_dir, self.current_global_step, asynchronous_report)
            else:
                if self.do_acc_eval:
                    sampled_completitions = wait_for_submitted_generation_jobs(total_async_generation_jobs, generation_output_dir)
                else:
                    sampled_completitions = None
                avg_nll_val = wait_for_submitted_nll_jobs(async_nll_jobs) if async_nll_jobs is not None else None
                asynchronous_report(sampled_completitions, self.current_global_step, avg_nll_val, subset_indices_in_eval_dpo)

    def test_loop(self):
        # should run alignment for all languages on the test sets.
        pass
    
    # def val(self, config, current_global_step, model_checkpoint):
    #     self.initialize_train(config)

    #     # self.config = config
    #     # self.current_star_iteration = 0
    #     self.current_global_step = current_global_step
    #     self.validation_loop(model_checkpoint, asynchronous=False)
        
        # gen_batch_size = self.gen_batch_size
        
        # train_dataset, val_dataset, test_datset = get_train_val_test_dataset(self.dataset_name)
        # self.generation_train_dataloader = DataLoader(train_dataset, batch_size=gen_batch_size, collate_fn=get_generation_collate_fn(self.agent.model_handler.get_tokenizer(self.model_name)), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore
        # self.generation_val_dataloader = DataLoader(val_dataset, batch_size=gen_batch_size, collate_fn=get_generation_collate_fn(self.agent.model_handler.get_tokenizer(self.model_name)), num_workers=self.num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore
        # self.latest_checkpoint = None
        # self.async_job_info = None
        

