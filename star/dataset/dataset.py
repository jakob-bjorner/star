import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from math import ceil
import numpy as np
from star.utils.utils import QUERY_TEMPLATE_MULTICHOICE, SYSTEM_PROMPTED_TEMPLATE


@dataclass
class GenerationCollateReturn:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    indices: List[int]
    input_strs: List[str]
    subjects: List[str]
    answers: List[str]
    def to(self, device):
        def tree_map(dict_obj):
            for key, val in dict_obj.items():
                if isinstance(val, torch.Tensor):
                    dict_obj[key] = val.to(device)
                elif isinstance(val, Dict):
                    tree_map(val)
        tree_map(self.__dict__)
        return self


# performing the train, test, val split before entering the dataset. As lists
def train_test_split(dataset, test_size, seed=0):
    n_samples = len(dataset)
    train_size = 1.0 - test_size
    assert isinstance(test_size, float)

    n_test = ceil(test_size * n_samples)
    n_train = int(train_size * n_samples)

    generator = np.random.default_rng(seed)
    permutation = generator.permutation(len(dataset))
    test_indices = permutation[:n_test]
    train_indices = permutation[n_test : (n_test + n_train)]
    return [dataset[i] for i in train_indices], [dataset[j] for j in test_indices]
def get_train_val_test_dataset(dataset_name: str, num_train_questions: int):
    """ "/nethome/jbjorner3/dev/hallucination-fun/simple-evals/mmlu.csv" or 
        "/nethome/jbjorner3/dev/hallucination-fun/simple-evals/mmlu_YO-NG.csv"
    """
    dataset_file = {"mmlu": "/nethome/jbjorner3/dev/hallucination-fun/simple-evals/mmlu.csv", 
                    "mmlu_YO-NG": "/nethome/jbjorner3/dev/hallucination-fun/simple-evals/mmlu_YO-NG.csv"}[dataset_name]
    df = pd.read_csv(dataset_file).rename(columns={"Unnamed: 0": "Subject_question_number"})
    df = df.reset_index()

    train_dataset, split_test = train_test_split(df.to_dict("records"), 0.2)
    val_dataset, test_dataset = train_test_split(split_test, 0.5)
    if num_train_questions >= 0:
        train_dataset = train_dataset[:num_train_questions]
    return train_dataset, val_dataset, test_dataset

# dl = DataLoader(train_dataset, batch_size=batch_size, collate_fn=get_generation_collate_fn(tokenizer), num_workers=num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore

def get_generation_collate_fn(tokenizer, system_message, preamble_to_question):
    def generation_collate_fn(examples):
        # print(examples)
        indices = []
        subjects = []
        answers = []
        input_strs = []
        for example in examples:
            indices.append(example["index"])
            formated_question = QUERY_TEMPLATE_MULTICHOICE.format(Preamble_To_Question=preamble_to_question, **example)
            conversation = {
                "system_message": system_message, # "You are a helpful assistant",
                # "system_message": "Cutting Knowledge Date: December 2023\nToday Date: 26 July 2024\n\nYou are a helpful assistant",
                # "system_message": "Cutting Knowledge Date: December 2023\nToday Date: 23 July 2024\n\nYou are a helpful assistant",
                "user_message": formated_question}
            input_str = SYSTEM_PROMPTED_TEMPLATE.format(**conversation) # + "Let's think through this carefully, step by step."
            input_strs.append(input_str)
            subjects.append(example["Subject"])
            answers.append(example["Answer"])
        tokenized_conversations = tokenizer(text=input_strs, padding=True, return_tensors='pt')
        input_ids = tokenized_conversations["input_ids"]
        attention_mask = tokenized_conversations["attention_mask"]
        collate_return = GenerationCollateReturn(input_ids, attention_mask, indices, input_strs, subjects, answers)
        return collate_return
    return generation_collate_fn



# dataset for rational from correct stuff
class RationalDataset(Dataset):
    def __init__(self, raw_responses, scores, prompted_questions, unique):
        # to instantiate this object I need to be able to get out the rational, which mean I need to somehow remove the answer from the generation process? Or isn't the answer regexed from the rational generation process? so I just need to reinforce the output created by the assistant to the last prompt?
        # for understanding the dataset, I just need to do simple preprocessing, the tokenization and stuff can be done in the collate function.
        # so, which answers do I need?
        self.examples = []
        unique_prompted_questions = set()
        for raw_response, prompted_question, score in zip(raw_responses, prompted_questions, scores):
            if score == 1.0: # answers are already filtered to have the <|eot_id|> to be at their end
                if unique and prompted_question in unique_prompted_questions:
                    continue
                self.examples.append((raw_response, prompted_question))  
                unique_prompted_questions.add(prompted_question)


    def __getitem__(self, i):
        return self.examples[i]
    def __len__(self):
        return len(self.examples)

@dataclass
class ReinforceCollateReturn:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor # specifically made to only reinforce the output string, till the last <|eot_id|> after the assistant. and if that didn't occur in the generation, then??? I shouldn't reinforce it...
    labels: torch.Tensor
    def to(self, device):
        def tree_map(dict_obj):
            for key, val in dict_obj.items():
                if isinstance(val, torch.Tensor):
                    dict_obj[key] = val.to(device)
                elif isinstance(val, Dict):
                    tree_map(val)
        tree_map(self.__dict__)
        return self
def get_reinforce_collate_fn(tokenizer):
    def reinforce_collate_fn(examples):
        full_question_and_responses = []
        raw_responses = []
        prompted_questions = []
        for example in examples:
            raw_response, prompted_question = example
            raw_responses.append(raw_response)
            prompted_questions.append(prompted_question)
            full_question_and_response = prompted_question + raw_response
            # print(full_question_and_response)
            full_question_and_responses.append(full_question_and_response)
        tokenizer.padding_side = "right"
        tokenized_conversations = tokenizer(full_question_and_responses, padding=True, return_tensors='pt')
        input_ids = tokenized_conversations["input_ids"]
        attention_mask = tokenized_conversations["attention_mask"]
        # import ipdb; ipdb.set_trace()
        labels = input_ids.clone()
        labels[attention_mask!=1] = -100
        for i in range(len(labels)):
            end_of_header_id = 128007 # when we get the last header, we get the beggining of the response.
            index_of_response_start = torch.where(labels[i] == end_of_header_id)[0][-1].item() + 1
            labels[i, :index_of_response_start] = -100
        
        return ReinforceCollateReturn(
            input_ids,
            attention_mask,
            labels
        )
    return reinforce_collate_fn
# reinforce_dataset = RationalDataset(raw_responses, scores, prompted_questions)
# dl_reinforce = DataLoader(reinforce_dataset, batch_size=8, collate_fn=reinforce_collate_fn, num_workers=0, pin_memory=True, pin_memory_device="cuda")


from collections import defaultdict
from star.utils.utils import GeneratedSamples
class DPODataset(Dataset):
    internal_data: list # list of tuples, giving the prompted qeustions, the answers, the raw responses, and the indices
    def __init__(self, subjects, indices, prompted_questions, answers, raw_responses, scores):
        # we now assume that everything is given to us flat, so we have to accumulate things by their index to process it the way we have below.
        aggrigating_samples_by_index = defaultdict(lambda: GeneratedSamples([],[],[],[],[],[],[]))
        for subject, index, prompted_question, answer, raw_response, score in zip(subjects, indices, prompted_questions, answers, raw_responses, scores):
            aggrigating_samples_by_index[index] += GeneratedSamples([index],[prompted_question],[answer],[subject],['No Answer needed'],[raw_response],[score])
        # filter to only the ones where the scores don't agree on some examples. 
        samples_by_index = list(aggrigating_samples_by_index.values())

        self.internal_data = []
        for i, scores_for_one_question in enumerate([sample_by_index.scores for sample_by_index in samples_by_index]):
            if 0 < sum(scores_for_one_question) < len(scores_for_one_question):
                # mark this as contrastive examples.
                for correct_index in [j for j, s in enumerate(scores_for_one_question) if s == 1.0]:
                    for incorrect_index in [j for j, s in enumerate(scores_for_one_question) if s == 0.0]:
                        if samples_by_index[i].raw_responses[correct_index] is None:
                            continue
                        if samples_by_index[i].raw_responses[incorrect_index] is None:
                            continue
                        self.internal_data.append(
                            (samples_by_index[i].indices[0], samples_by_index[i].subjects[0], samples_by_index[i].prompted_questions[0], 
                             (samples_by_index[i].model_answers[correct_index], samples_by_index[i].scores[correct_index], samples_by_index[i].raw_responses[correct_index]), 
                             (samples_by_index[i].model_answers[incorrect_index], samples_by_index[i].scores[incorrect_index], samples_by_index[i].raw_responses[incorrect_index])
                            )
                        )

    def __len__(self):
        return len(self.internal_data)
    def __getitem__(self, idx):
        return self.internal_data[idx]

@dataclass
class DPOCollateReturn:
    indices: List[int]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompted_questions: List[str]
    raw_responses: list[tuple[str, str]]
    subjects: List[str]
    def to(self, device):
        def tree_map(dict_obj):
            for key, val in dict_obj.items():
                if isinstance(val, torch.Tensor):
                    dict_obj[key] = val.to(device)
                elif isinstance(val, Dict):
                    tree_map(val)
        tree_map(self.__dict__)
        return self

def get_dpo_collate_fn(tokenizer):
    def dpo_collate_fn(examples):
        subjects = []
        inputs_to_combine = []
        indices = []
        prompted_questions = []
        raw_responses = []
        tokenizer.padding_side = 'right'
        for example in examples:
            indices.append(example[0])
            subjects.append(example[1])
            prompted_question = example[2]
            raw_response_correct = example[3][2]
            raw_response_incorrect = example[4][2]
            prompted_questions.append(prompted_question)
            raw_responses.append((raw_response_correct, raw_response_incorrect))
            inputs_to_combine.extend([ # prompted_question
                [tokenizer(prompted_question, add_special_tokens=False).input_ids, tokenizer(raw_response_correct, add_special_tokens=False).input_ids],
                [tokenizer(prompted_question, add_special_tokens=False).input_ids, tokenizer(raw_response_incorrect, add_special_tokens=False).input_ids]
            ])

        inputs_to_pad = [sum(l, []) for l in inputs_to_combine]
        inputs_padded = []
        max_len_input = max(len(l) for l in inputs_to_pad) 
        attention_mask = torch.ones((len(inputs_to_pad), max_len_input), dtype=torch.float32)
        right_pads = []
        for i, input_to_pad in enumerate(inputs_to_pad):
            right_pad = max_len_input - len(input_to_pad)
            inputs_padded.append(input_to_pad + [128004] * right_pad)
            right_pads.append(right_pad)
            attention_mask[i, len(input_to_pad):] = 0
        input_ids = torch.tensor(inputs_padded)
        labels = torch.full_like(input_ids, fill_value=-100)
        for i, (input_to_combine, right_pad) in enumerate(zip(inputs_to_combine, right_pads)):
            labels[i,-len(input_to_combine[-1])-right_pad:max_len_input-right_pad] = input_ids[i, -len(input_to_combine[-1])-right_pad:max_len_input-right_pad] # take only the response portion of the answer from the labels

        return DPOCollateReturn(
            indices,
            input_ids, 
            attention_mask,
            labels,
            prompted_questions,
            raw_responses,
            subjects
        )
    return dpo_collate_fn


# dpo_dataset = DPODataset(data, indices[0], prompted_questions[0], answers[0], raw_responses[0], scores[0])
# dpo_dataloader = DataLoader(dpo_dataset, batch_size=8, collate_fn=dpo_collate_fn, num_workers=0, pin_memory=True, pin_memory_device="cuda")
