# 
# data.py
# 
# Author(s):
# Philip Wiese <wiesep@student.ethz.ch>
# 
# Copyright (c) 2023 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import os
import torch
import torchvision

from systems.utils.data import default_dataset_cv_split
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset

from datasets import load_dataset

# Inspired by https://nni.readthedocs.io/en/stable/tutorials/pruning_bert_glue.html
task_to_keys = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}

def load_data_set(partition: str,
                  path_data: str,
                  n_folds: int,
                  current_fold_id: int,
                  cv_seed: int,
                  transform: torchvision.transforms.Compose,
                  size=1000,
                  task_name='cola') -> Dataset:

    sentence1_key, sentence2_key = task_to_keys[task_name]

    tokenizer = RobertaTokenizerFast.from_pretrained("kssteven/ibert-roberta-base", cache_dir=path_data)
    # Disabling tokenizer parallelism, we're using DataLoader multithreading already
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def tokenize_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*args, padding="max_length", truncation=True)

    if partition == 'train':
        dataset = load_dataset("glue", task_name, cache_dir=path_data, split='train')
    elif partition == 'valid':
        if task_name == 'mnli':
            # WIESEP: Load matched or unmatched set?
            dataset = load_dataset("glue", task_name, cache_dir=path_data, split='validation_matched')
        else:
            dataset = load_dataset("glue", task_name, cache_dir=path_data, split='validation')
    elif partition == 'test':
        if task_name == 'mnli':
            # WIESEP: Load matched or unmatched set?
            dataset = load_dataset("glue", task_name, cache_dir=path_data, split='test_matched')
        else:
            dataset = load_dataset("glue", task_name, cache_dir=path_data, split='test')
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Remove the text columns because the model does not accept raw text as an input
    for key in task_to_keys[task_name]:
        if key is not None:
            tokenized_datasets = tokenized_datasets.remove_columns(key)
    # Remove the idx column because the model does not accept idx as an input
    tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
    # Rename the label column to labels because the model expects the argument to be named labels
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # Set the format of the dataset to return PyTorch tensors instead of lists:
    tokenized_datasets.set_format("torch")
    # tokenized_datasets.set_transform(transform=transform)

    if size != -1 and size <= dataset.num_rows:
        tokenized_datasets = tokenized_datasets.select(range(size))

    if partition in {'train', 'valid'}:
        if n_folds > 1:  # this is a cross-validation experiment
            train_fold_indices, valid_fold_indices = default_dataset_cv_split(dataset=tokenized_datasets, n_folds=n_folds, current_fold_id=current_fold_id, cv_seed=cv_seed)

            if partition == 'train':
                tokenized_datasets = tokenized_datasets.select(train_fold_indices)
            elif partition == 'valid':
                tokenized_datasets = tokenized_datasets.shuffle(seed=cv_seed).select(valid_fold_indices)

        else:
            tokenized_datasets = tokenized_datasets.shuffle(seed=cv_seed)
    else:
        assert partition == 'test'
        tokenized_datasets = tokenized_datasets.shuffle(seed=cv_seed)

    return tokenized_datasets

