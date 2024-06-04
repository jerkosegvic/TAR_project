from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Union
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from dataset import MyDataset
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(dataset: Dataset, model: PreTrainedModel, device: torch.device = torch.device('cpu'), plot: bool = False):
    '''
    Evaluates the model on the given dataset.
    
    Parameters:
        dataset: Dataset
            The dataset to evaluate on.
        model: PreTrainedModel
            The model to evaluate.
        device: torch.device
            The device to use.
        plot: bool
    '''
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    correct_labels = []
    model_predictions = []
    with torch.no_grad():
        for batch in loader:
            id_, speaker, sex, text, text_en, embedding, attention_mask, label = batch
            model_output = model(input_ids=embedding, labels=label, attention_mask=attention_mask)
            logits = model_output.logits
            predictions = torch.argmax(logits, dim=1)
            correct_labels.extend(label.cpu().numpy())
            model_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(correct_labels, model_predictions)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion matrix:\n{confusion_matrix(correct_labels, model_predictions)}')


def train(dataset_train: Dataset, dataset_val: Dataset, model: PreTrainedModel, optimizer_type: type = torch.optim.Adam, 
        batch_size: int = 8, epochs: int = 5, device: torch.device = torch.device('cpu'), lr: float = 1e-4, gamma: Union[float,None] = None):
    '''
    Trains the model on the given dataset.

    Parameters:
        dataset_train: Dataset
            The training dataset.
        dataset_val: Dataset
            The validation dataset.
        model: PreTrainedModel
            The model to train.
        optimizer_type: type
            The optimizer type to use.
        batch_size: int
            The batch size.
        epochs: int
            The number of epochs.
        device: torch.device
            The device to use.
        lr: float
            The learning rate.
        gamma: Union[float,None]
            The gamma parameter for the scheduler.
    '''
    model.to(device)
    optimizer = optimizer_type(model.parameters())

    if gamma is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    log_rate = len(train_loader) // 20

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for ind, batch in enumerate(train_loader):
            optimizer.zero_grad()
            id_, speaker, sex, text, text_en, embedding, attention_mask, label = batch
            breakpoint()
            model_output = model(input_ids=embedding, labels=label, attention_mask=attention_mask)
            
            ##TODO: change this if you want to use a different loss function
            ## or the model that outputs logits
            loss = model_output.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if ind % log_rate == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {ind+1}/{len(train_loader)}, Batch loss: {loss.item()}, Average epoch loss: {epoch_loss/(ind+1)}')
                
        print(f'Epoch {epoch+1}/{epochs}, Average epoch loss: {epoch_loss/len(train_loader)}')

        evaluate(dataset_val, model, device=device)

        if gamma is not None:
            scheduler.step()

    return model

if __name__ == '__main__':
    table = pd.read_csv('data/power/power-hr-train.tsv', sep='\t')
    ids = table['id'].tolist()
    speakers = table['speaker'].tolist()
    sexes = table['sex'].tolist()
    texts = table['text'].tolist()
    texts_en = table['text_en'].tolist()
    labels = table['label'].tolist()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    dataset = MyDataset(ids, speakers, sexes, texts, texts_en, labels)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train(
        model=model,
        optimizer_type=torch.optim.Adam,
        dataset_train=train_dataset,
        dataset_val=test_dataset,
        epochs=5,
        batch_size=8,
        lr=1e-5,
        device=torch.device('cpu'),
        gamma=0.85,
    )