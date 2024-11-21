import os
import torch
from star.model.model import BaseModelHandler
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List, Any
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
import tqdm
from star.dataset.dataset import GenerationCollateReturn, DPOCollateReturn, ReinforceCollateReturn
import pandas as pd
from star.utils.utils import GeneratedSamples, normalize_response, MULTILINGUAL_ANSWER_REGEXES, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE, normalize_extracted_answer
import re
@dataclass
class DPOStepOutput:
    loss: torch.Tensor
    reward_winning: torch.Tensor
    reward_losing: torch.Tensor

class Agent:
    def __init__(self, model_handler: BaseModelHandler): 
        # should the agent have a model? 
        # the model should restart for every generation, so potentially, 
        # I could just have a get model function or something of that sort passed in.
        # this should have an ability to:
        #   get the base model for training, to save, and for sampling from a saved checkpoint.
        self.model_handler = model_handler
        self.async_job_handlers = dict()
    def get_new_model_tokenizer_optimizer_scheduler_max_steps(self, gradient_steps_for_scheduler):
        model, tokenizer = self.model_handler.get_model_tokenizer(None)
        self.model_handler.prepare_for_training(model)
        optimizer, scheduler = self.model_handler.get_optimizer_scheduler_max_steps(model, gradient_steps_for_scheduler)
        return model, tokenizer, optimizer, scheduler
    def save_model(self, model, checkpoint):
        self.model_handler.save(model, checkpoint)
    def reinforce_train_step(self, model, data):
        # with torch.autocast('cuda', dtype = torch.bfloat16):
        return model(**data.__dict__)
    def dpo_train_step(self, model, ref_model, data, beta):
        # TODO: modify the DPO masking implementation to exclude the <eos> token as recommended by DPO paper?
        def get_nlls_per_batch_element(model, input_ids, attention_mask, labels):
            model_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # need to pass in labels else the computation is wrong???
            logits = model_output.logits

            nlls_per_token = torch.nn.functional.cross_entropy(logits.transpose(-1,-2)[:,:,:-1], labels[:, 1:], reduction='none')
            # if True:
            #     print("debugging if logits are slightly off numerically. They do seem to be. not sure what could be different?")
            #     labels_temp = labels[:, 1:].clone()
            #     logits_temp = logits[:, :-1, :]
            #     loss_mask = labels_temp != -100

            #     # dummy token; we'll ignore the losses on these tokens later
            #     labels_temp[labels_temp == -100] = 0

            #     per_token_logps = torch.gather(logits_temp.log_softmax(-1), dim=2, index=labels_temp.unsqueeze(2)).squeeze(2)

            #     all_logps = (per_token_logps * loss_mask).sum(-1)
            #     print(f"{all_logps=}")
            #     print(f"{nlls_per_token.flatten(1).sum(1)=}")
            # import ipdb; ipdb.set_trace()
            return nlls_per_token.flatten(1).sum(1)
        input_ids = data.input_ids
        attention_mask = data.attention_mask
        labels = data.labels
    
        with torch.no_grad():
            nll_ref_sum = get_nlls_per_batch_element(ref_model, input_ids, attention_mask, labels)
        nll_model_sum = get_nlls_per_batch_element(model, input_ids, attention_mask, labels)
        log_ratios = - (nll_model_sum - nll_ref_sum)
        per_example_loss = - torch.nn.functional.logsigmoid(beta * (log_ratios[0::2] - log_ratios[1::2]))
        loss = per_example_loss.mean()
        with torch.no_grad():
            reward_winning = beta * log_ratios[0::2].mean()
            reward_losing = beta * log_ratios[1::2].mean()
        return DPOStepOutput(loss=loss, reward_winning=reward_winning, reward_losing=reward_losing)
    # def dpo_train_step_alternative(self, model, ref_model, data, beta):
    #     input_ids = data.input_ids
    #     attention_mask = data.attention_mask
    #     labels = data.labels
    #     labels = labels[:, 1:].clone()
    #     outputs = model(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         use_cache=False,
    #     )
    #     logits = outputs.logits
    #     logits = logits[:, :-1, :]
    #     loss_mask = labels != label_pad_token_id

    #     # dummy token; we'll ignore the losses on these tokens later
    #     labels[labels == label_pad_token_id] = 0

    #     per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    #     all_logps = (per_token_logps * loss_mask).sum(-1)
    #     # return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)
    

    #     pi_logratios = policy_chosen_logps - policy_rejected_logps
    #     ref_logratios = reference_chosen_logps - reference_rejected_logps

    #     pi_logratios = pi_logratios
    #     ref_logratios = ref_logratios
    #     logits = pi_logratios - ref_logratios

    #     chosen_rewards = (
    #         beta * (policy_chosen_logps - reference_chosen_logps).detach()
    #     )
    #     rejected_rewards = (
    #         beta * (policy_rejected_logps - reference_rejected_logps).detach()
    #     )

    #     losses = (
    #         -torch.nn.functional.logsigmoid(beta * logits) 
    #     )
    #     return DPOStepOutput(loss=losses, reward_winning=, reward_losing=)
    
def get_executor(output_dir: str, single_batch: bool):
    import time
    start_time = time.time()
    import submitit
    executor = submitit.AutoExecutor(folder=os.path.join(output_dir, 'submitit_logs/%j'))
    os.unsetenv("SLURM_CPU_BIND")
    # single batch implies we run debug implies we just launch on kargo partition.
    
    partition = 'kargo-lab' if single_batch else 'overcap'
    partition = 'overcap' # this for when not enough lab compute is available just uncomment this.
    cpus_per_task = 6 if partition == 'kargo-lab' else 14
    slurm_array_parallelism = 4 if partition == "kargo-lab" else 55
    executor.update_parameters(
        # set to time out after 2 days
        timeout_min=60 * 24 * 2,
        # set to short session
        # slurm_partition="kargo-lab", # slurm_partition="overcap", slurm_account="overcap", is also required with overcap
        # slurm_partition="overcap",
        slurm_partition=partition,
        # slurm_account="overcap",
        slurm_qos="short",
        slurm_exclude="major,crushinator,nestor,chappie,voltron",
        cpus_per_task=cpus_per_task, # 6 or 14.
        slurm_array_parallelism=slurm_array_parallelism, # only about 11 get scheduled at once, and with batch of 128 gets 08:44 mins
        slurm_additional_parameters= {
            "gpus":"a40:1"
        }
    )
    return executor

def generate_async_nll_jobs_distributed(output_dir: str, dataloader: DataLoader, agent: Agent, train_strategy:str, dpo_beta: float, checkpoint: Optional[str], single_batch: bool): # num_batches, config, device
    executor = get_executor(output_dir, single_batch)
    def launch_job_passthrough_function(checkpoint, dataloader, seed, agent):
        model, tokenizer = agent.model_handler.get_model_tokenizer(checkpoint)
        model = agent.model_handler.prepare_for_training(model)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        losses_list = []
        len_list = []
        with torch.no_grad():
            for data in tqdm.tqdm(dataloader):
                data.to("cuda")
                if "dpo" in train_strategy:
                    assert isinstance(data, DPOCollateReturn)
                    print(f"{data.input_ids.shape=}, {data.indices=}, {data.subjects=}")
                    if "ref_model" not in locals():
                        ref_model, _ = agent.model_handler.get_model_tokenizer()
                        ref_model = agent.model_handler.prepare_for_training(ref_model)
                    else:
                        ref_model = locals()['ref_model']
                    losses = agent.dpo_train_step(model, ref_model, data, dpo_beta)
                elif "reinforce" in train_strategy:
                    assert isinstance(data, ReinforceCollateReturn)
                    print(f"{data.input_ids.shape=}")

                    losses = agent.reinforce_train_step(model, data)
                else:
                    raise Exception(f"invalid option for train_strategy in Trainer {train_strategy}")
                losses_list.append(losses)
                len_list.append(data.input_ids.shape[0])
        # seed = time.time_ns()
        return losses_list, len_list

    jobs = []
    print("launching nll jobs with these generation jobs, they should finish well before.")
    with executor.batch():
        job = executor.submit(launch_job_passthrough_function, checkpoint, dataloader, torch.randint(0,1000000,(1,)).item(), agent)
        jobs.append(job)
        # for data in dataloader:
        #     job = executor.submit(launch_job_passthrough_function, checkpoint, data, torch.randint(0,1000000,(1,)).item())
        #     jobs.append(job)
        #     if single_batch:
        #         break
    return jobs

def generate_async_sample_jobs_distributed(output_dir: str, dataloader: DataLoader, model_handler: BaseModelHandler, checkpoint: Optional[str], temperature: float, single_batch: bool): # num_batches, config, device
    executor = get_executor(output_dir, single_batch)
    def launch_job_passthrough_function(checkpoint, data: GenerationCollateReturn, temperature, seed):
        model, tokenizer = model_handler.get_model_tokenizer(checkpoint)
        model = model_handler.prepare_for_inference(model)
        # seed = time.time_ns()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        data.to("cuda")
        print(f"input_strs len={[len(s) for s in data.input_strs]}, {temperature=}, {data.indices=}, {data.subjects=}")
        with torch.no_grad():
            if temperature == 0.0:
                outputs = model.generate(input_ids=data.input_ids, attention_mask=data.attention_mask, max_new_tokens=1024, use_cache=True, do_sample=False)
            else:
                # could think to do different sampling strategies, like high temp, then greedy, or even high temp then beam search if I can get beam search working that is...
                print('sampling!')
                outputs = model.generate(input_ids=data.input_ids, attention_mask=data.attention_mask, max_new_tokens=1024, use_cache=True, do_sample=True, temperature=temperature)
        model_answers = []
        raw_responses = []
        scores = []
        for i, output in enumerate(tokenizer.batch_decode(outputs[:, data.input_ids.shape[1]:])):
            try:
                if "<|eot_id|>" not in output:
                    raw_answer = output[:output] # this will lead to some examples which go on for ever being reduced in likelihood if they are compared to one which terminates with the correct answer.
                    extracted_answer = None
                    score = 0.0
                else:
                    raw_answer = output[:output.index("<|eot_id|>") + len("<|eot_id|>")]
                    response_text = normalize_response(raw_answer)
                    extracted_answer = None
                    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                        match = re.search(regex, response_text)
                        if match:
                            extracted_answer = normalize_extracted_answer(match.group(1))
                            break
                    score = 1.0 if extracted_answer == data.answers[i] else 0.0
                model_answers.append(extracted_answer)
                raw_responses.append(raw_answer)
                scores.append(score)
            except Exception as e:
                import sys
                txt = f"exception {e}... other details: {outputs.shape=},\n{data.answers[i]=},\n{data.input_strs[i]=},\n{output=}".encode("utf8") # type: ignore
                sys.stdout.buffer.write(txt)
                model_answers.append(None)
                raw_responses.append(None)
                scores.append(0.0)
                pass
        return GeneratedSamples(data.indices, data.input_strs, model_answers, data.subjects, data.answers, raw_responses, scores)
    jobs = []
    print(f"expected time to collect is 6 minutes per batch for 128 batch size {dataloader.batch_size=}, datasetlen={len(dataloader.dataset)}, batches={len(dataloader)}") # type: ignore
    with executor.batch():
        for data in dataloader:
            job = executor.submit(launch_job_passthrough_function, checkpoint, data, temperature, torch.randint(0,1000000,(1,)).item())
            jobs.append(job)
            if single_batch:
                break
    return jobs

def wait_for_submitted_jobs_base(jobs):
    num_fails = 0
    all_results = []
    for job in tqdm.tqdm(jobs):
        try:
            res = job.result()
            all_results = all_results + [res]
        except Exception as e:
            print("Error in getting job result for distributed sampling: ", e)
            num_fails += 1
    print(f"There were {num_fails} fails out of {len(jobs)} attempts")
    return num_fails, all_results

def wait_for_submitted_nll_jobs(jobs):
    num_fails, nll_samples = wait_for_submitted_jobs_base(jobs)
    nll_samples = list(zip(sum((l[0] for l in nll_samples), []), sum((l[1] for l in nll_samples), []))) # concats all the lists if we decide to have many batches per job. For now its easy to eval all batches in one job.
    if len(nll_samples) > 0:
        avg_nll_val = {key: sum([nll.__dict__[key].item() * num_samples for nll, num_samples in nll_samples]) / sum(num_samples for _, num_samples in nll_samples) for key in nll_samples[0][0].__dict__.keys()} 
    else:
        avg_nll_val = None
    return avg_nll_val

def wait_for_submitted_generation_jobs(jobs, output_dir):
    num_fails, all_results_list = wait_for_submitted_jobs_base(jobs)
    all_results = sum(all_results_list, start=GeneratedSamples([],[],[],[],[],[],[]))

    if os.path.exists(os.path.join(output_dir, "generated.csv")):
        pd.DataFrame(data=all_results.__dict__).to_csv(open(os.path.join(output_dir, "generated.csv"), mode='a'), header=False, index=False)
    else:
        pd.DataFrame(data=all_results.__dict__).to_csv(open(os.path.join(output_dir, "generated.csv"), mode='w'), index=False)

    if num_fails >= len(jobs):
        raise Exception(f"Too many fails. check {output_dir} in the submitit directory")
    return all_results

