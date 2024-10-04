import os
import torch
from star.model.model import BaseModelHandler
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List, Any
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
import tqdm
from star.dataset.dataset import GenerationCollateReturn
import pandas as pd


def generate_async_sample_jobs_distributed(output_dir: str, dataloader: DataLoader, model_handler: BaseModelHandler, checkpoint: Optional[str], temperature: float, single_batch: bool): # num_batches, config, device
    import time
    start_time = time.time()
    import submitit
    executor = submitit.AutoExecutor(folder=os.path.join(output_dir, 'submitit_logs/%j'))
    os.unsetenv("SLURM_CPU_BIND")
    executor.update_parameters(
        # set to time out after 2 days
        timeout_min=60 * 24 * 2,
        # set to short session
        # slurm_partition="kargo-lab", # slurm_partition="overcap", slurm_account="overcap", is also required with overcap
        slurm_partition="overcap",
        # slurm_account="overcap",
        slurm_qos="short",
        slurm_exclude="major,crushinator,nestor,chappie,voltron",
        cpus_per_task=14,
        slurm_array_parallelism=55, # only about 11 get scheduled at once, and with batch of 128 gets 08:44 mins
        slurm_additional_parameters= {
            "gpus":"a40:1"
        }
    )
    def launch_job_passthrough_function(checkpoint, data: GenerationCollateReturn, temperature):
        model, tokenizer = model_handler.get_model_tokenizer(checkpoint)
        model = model_handler.prepare_for_inference(model)
        data.to("cuda")
        print(f"input_strs len={[len(s) for s in data.input_strs]}, {temperature=}, {data.indices=}, {data.subjects=}")
        with torch.no_grad():
            if temperature == 0.0:
                outputs = model.generate(input_ids=data.input_ids, attention_mask=data.attention_mask, max_new_tokens=1024, use_cache=True, do_sample=False)
            else:
                outputs = model.generate(input_ids=data.input_ids, attention_mask=data.attention_mask, max_new_tokens=1024, use_cache=True, do_sample=True, temperature=temperature)
        model_answers = []
        raw_responses = []
        scores = []
        import re
        from star.utils.utils import normalize_response, MULTILINGUAL_ANSWER_REGEXES, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE, normalize_extracted_answer
        for i, output in enumerate(tokenizer.batch_decode(outputs[:, data.input_ids.shape[1]:])):
            try:
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
            job = executor.submit(launch_job_passthrough_function, checkpoint, data, temperature)
            jobs.append(job)
            if single_batch:
                break

    return jobs

def wait_for_submitted_jobs(jobs, output_dir):
    num_fails = 0
    all_results = GeneratedSamples([],[],[],[],[],[],[])
    for job in tqdm.tqdm(jobs):
        try:
            res = job.result()
            all_results = all_results + res
        except Exception as e:
            print("Error in getting job result for distributed sampling of diffusion: ", e)
            num_fails += 1
    pd.DataFrame(data=all_results.__dict__).to_csv(os.path.join(output_dir, "generated.csv"))
    print(f"There were {num_fails} fails out of {len(jobs)} attempts")
    if num_fails >= len(jobs):
        raise Exception(f"Too many fails. check {output_dir} in the submitit directory")
    return all_results



@dataclass
class GeneratedSamples:
    indices: list[int]
    prompted_questions: list[str]
    model_answers: list[str]
    subjects: List[str]
    correct_answers: list[str]
    raw_responses: list[str]
    scores: list[float]
    def __add__(self, other):
        return GeneratedSamples(**{k: v + other.__dict__[k] for k,v in self.__dict__.items()})
    def get_subset_by_subject(self, subject: str):
        subset_indices = [i for i, s in enumerate(self.subjects) if s == subject] 
        return GeneratedSamples(**{k: [v[i] for i in subset_indices] for k, v in self.__dict__.items()})

class Agent:
    def __init__(self, model_handler: BaseModelHandler): 
        # should the agent have a model? 
        # the model should restart for every generation, so potentially, 
        # I could just have a get model function or something of that sort passed in.
        # this should have an ability to:
        #   get the base model for training, to save, and for sampling from a saved checkpoint.
        self.model_handler = model_handler
        self.async_job_handlers = dict()
    def get_new_model_tokenizer_optimizer_scheduler_max_steps(self, steps_per_epoch):
        model, tokenizer = self.model_handler.get_model_tokenizer(None)
        self.model_handler.prepare_for_training(model)
        optimizer, scheduler, max_steps = self.model_handler.get_optimizer_scheduler_max_steps(model, steps_per_epoch)
        return model, tokenizer, optimizer, scheduler, max_steps
    def save_model(self, model, checkpoint):
        self.model_handler.save(model, checkpoint)
    def train_step(self, model, data):
        # with torch.autocast('cuda', dtype = torch.bfloat16):
        with torch.amp.autocast('cuda', dtype = torch.bfloat16): # type: ignore
            return model(**data.__dict__)
    
