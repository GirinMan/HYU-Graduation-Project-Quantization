from loguru import logger
import argparse

import torch
import datasets
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

class number():
    def __init__(self) -> None:
        self.count = 0
        self.total = 0
    
    def add_one(self):
        self.count += 1

    def count_total(self):
        self.total += 1

def get_prompt_dataset(dataset,
                       tokenizer,
                       max_len=64,
                       max_label_len=4,
                       prefix='다음 문장은 긍정일까요 부정일까요?\n',
                       suffix='\n정답:',
                       columns=['document', 'label'],
                       ids_to_labels=None):
    
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    max_label_len = max_label_len + 1

    template=lambda o: f'{prefix}{o.strip()}{suffix}'

    tokenized_prefix = tokenizer.encode(prefix)
    tokenized_suffix = tokenizer.encode(suffix)
    prefix_len = len(tokenized_prefix)
    suffix_len = len(tokenized_suffix)

    #counter = number()

    def generate_prompt(exmaples):
        documents = exmaples[columns[0]]
        documents = [template(d) for d in documents]
        labels = exmaples[columns[1]]
        if ids_to_labels is not None:
            labels = [ids_to_labels[i] for i in labels]
        labels = [l + tokenizer.eos_token for l in labels]
        
        tokenized_documents = tokenizer(documents)['input_ids']
        tokenized_labels = tokenizer(labels)['input_ids']

        token_ids = []
        label_ids = []
        prompt_ids = []
        prompt_attention_masks = []
        for document_toked, label_toked in zip(tokenized_documents, tokenized_labels):
            document_len = len(document_toked)
            label_len = len(label_toked)
            #counter.count_total()
            if document_len + label_len > max_len:
                #counter.add_one()
                label_toked = label_toked[-max_label_len:]
                document_toked = document_toked[prefix_len:-suffix_len]
                document_toked = document_toked[:max_len - prefix_len - suffix_len - max_label_len]
                document_toked = tokenized_prefix + document_toked + tokenized_suffix
                document_len = len(document_toked)
                label_len = len(label_toked)
                #print(tokenizer.decode(document_toked+label_toked))
            label_id = [
                -100,
            ] * document_len + label_toked
            while len(label_id) < max_len:
                label_id += [-100]
            token_id = document_toked + label_toked
            while len(token_id) < max_len:
                token_id += [tokenizer.pad_token_id]
            prompt_attention_mask = [0] * (max_len - document_len) + [1] * document_len
            while len(document_toked) < max_len:
                document_toked = [tokenizer.pad_token_id] + document_toked

            token_ids.append(token_id)
            label_ids.append(label_id)
            prompt_ids.append(document_toked)
            prompt_attention_masks.append(prompt_attention_mask)

        return {
            'input_ids' : token_ids,
            'labels' : label_ids,
            'prompt_ids' : prompt_ids,
            'prompt_attention_masks' : prompt_attention_masks,
            'decoded_labels' : exmaples[columns[1]],
        }

    features = dataset["train"].features
    dataset = dataset.map(generate_prompt, batched=True, remove_columns=features)

    #print(f"{counter.count} elements has been truncated due to max length limit out of total {counter.total} entities.")
    
    return dataset

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {'input_ids': torch.LongTensor(input_ids), 'labels': torch.LongTensor(labels)}

def collate_fn_eval(batch):
    prompt_ids = [item['prompt_ids'] for item in batch]
    prompt_attention_masks = [item['prompt_attention_masks'] for item in batch]
    decoded_labels = [item['decoded_labels'] for item in batch]
    return {'input_ids': torch.LongTensor(prompt_ids), 'attention_mask': torch.LongTensor(prompt_attention_masks), 'decoded_labels': decoded_labels}

# return the dataloader for train split
def get_train_dataloader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn):
    train = DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True, collate_fn=collate_fn)
    return train

def get_eval_dataloader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn_eval):
    eval = DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False, collate_fn=collate_fn)
    return eval

    
if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")

    nsmc = datasets.load_dataset('nsmc')
    nsmc = get_prompt_dataset(nsmc, tokenizer, max_label_len=1, ids_to_labels={0:"부정", 1:"긍정"})

    train = get_train_dataloader(nsmc)
    for batch in train:
        print(batch)
        decoded_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
        for text in decoded_inputs:
            print(text)
            print()
        break

    eval = get_eval_dataloader(nsmc)
    for batch in eval:
        print(batch)
        decoded_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
        for text in decoded_inputs:
            print(text)
            print()
        break
    
    print("End")
