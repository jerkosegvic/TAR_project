from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Union
from transformers import AutoTokenizer, AutoModel

class MyDataset(Dataset):
    def __init__(self, 
                ids: List[str], 
                speakers: List[str], 
                sexes: List[str], 
                texts: List[str], 
                texts_en: List[str], 
                labels: List[bool],
                device: torch.device = torch.device('cpu'),
                model_name: str = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
                max_length: int = 512
        ):
        assert len(ids) == len(speakers) == len(sexes) == len(texts) == len(texts_en) == len(labels)
        self.ids = []
        self.speakers = []
        self.sexes = []
        self.texts = []
        self.texts_en = []
        self.embeddings = []
        self.attention_masks = []
        self.labels = []
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for i in range(len(ids)):
            text = texts[i]
            inputs = self.tokenizer(text, add_special_tokens=True, return_tensors='pt', padding='max_length',max_length=max_length)
            if inputs['input_ids'].shape[1] <= max_length:
                self.ids.append(ids[i])
                self.speakers.append(speakers[i])
                self.sexes.append(sexes[i])
                self.texts.append(texts[i])
                self.texts_en.append(texts_en[i])
                self.embeddings.append(inputs['input_ids'][0])
                self.attention_masks.append(inputs['attention_mask'])
                self.labels.append(torch.tensor((labels[i]), dtype=torch.long))
                
        print(f'Loaded {len(self.ids)}/{len(ids)} samples.')

    def __getitem__(self, index):
        return self.ids[index], self.speakers[index], self.sexes[index], self.texts[index], \
                self.texts_en[index], self.embeddings[index].to(self.device), self.attention_masks[index].to(self.device), self.labels[index]
            
    def __len__(self):
        return len(self.ids)

    def set_device(self, device: torch.device):
        '''
        Sets the device to the given device.
        '''
        self.device = device