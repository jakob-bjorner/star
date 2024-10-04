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
def get_train_val_test_dataset(dataset_name: str):
    """ "/nethome/jbjorner3/dev/hallucination-fun/simple-evals/mmlu.csv" or 
        "/nethome/jbjorner3/dev/hallucination-fun/simple-evals/mmlu_YO-NG.csv"
    """
    dataset_name = {"mmlu": "/nethome/jbjorner3/dev/hallucination-fun/simple-evals/mmlu.csv", 
                    "mmlu_YO-NG": "/nethome/jbjorner3/dev/hallucination-fun/simple-evals/mmlu_YO-NG.csv"}[dataset_name]
    df = pd.read_csv(dataset_name).rename(columns={"Unnamed: 0": "Subject_question_number"})
    df = df.reset_index()

    # Note the dataset is loaded globably mainly for convinience
    train_dataset, split_test = train_test_split(df.to_dict("records"), 0.2)
    val_dataset, test_dataset = train_test_split(split_test, 0.5)
    return train_dataset, val_dataset, test_dataset

# dl = DataLoader(train_dataset, batch_size=batch_size, collate_fn=get_generation_collate_fn(tokenizer), num_workers=num_workers, pin_memory=True, pin_memory_device="cuda") # type: ignore

def get_generation_collate_fn(tokenizer):
    def generation_collate_fn(examples):
        # print(examples)
        indices = []
        subjects = []
        answers = []
        input_strs = []
        for example in examples:
            indices.append(example["index"])
            formated_question = QUERY_TEMPLATE_MULTICHOICE.format(**example)
            conversation = {
                "system_message": "You are a helpful assistant",
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
    def __init__(self, raw_responses, scores, prompted_questions):
        # to instantiate this object I need to be able to get out the rational, which mean I need to somehow remove the answer from the generation process? Or isn't the answer regexed from the rational generation process? so I just need to reinforce the output created by the assistant to the last prompt?
        # for understanding the dataset, I just need to do simple preprocessing, the tokenization and stuff can be done in the collate function.
        # so, which answers do I need?
        self.examples = []
        for raw_response, prompted_question, score in zip(raw_responses, prompted_questions, scores):
            if score == 1.0: # answers are already filtered to have the <|eot_id|> to be at their end
                self.examples.append((raw_response, prompted_question))       
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
