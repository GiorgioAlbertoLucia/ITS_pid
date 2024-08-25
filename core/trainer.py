'''
    Class to handle training of the model
'''

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

import sys
sys.path.append('..')
from core.dataset import DataHandler

class NeuralNetworkTrainer:
    def __init__(self, model, train_dataset:DataHandler, validation_dataset:DataHandler, loss, optimizer, nn_mode, batch_size=32, num_workers=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        #self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #self.validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        self.loss = loss
        self.optimizer = optimizer
        self.nn_mode = nn_mode
        self.softmax_f = nn.Softmax(dim=1)

    def train(self, **kwargs):
        '''
            Train the model

            Parameters
            ----------
            **kwargs:
        '''

        self.model.train()        
        running_loss = 0.0
        batch_idx = 0
        for inputs, labels in self.train_loader:
            
            if batch_idx % 1000 == 0:
                print('processing batch: ', batch_idx)
            batch_idx += 1

            assert labels.min() >= 0 and labels.max() < 7, f"Invalid labels: {labels.min()} {labels.max()}"
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                print("Warning: nan values found in the inputs or labels")
                continue

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if torch.isnan(outputs).any():
                print("Warning: nan values found in the outputs")
                continue
            loss = self.loss(outputs, labels)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Warning: nan or inf values found in the loss")
                continue
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validation_loss(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                running_loss += loss.item()
        
        epoch_loss = running_loss / len(self.validation_loader)
        return epoch_loss

    def evaluate(self, mode:str='test'):
        self.model.eval()
        all_preds = []
        all_labels = []

        loader = None
        if mode == 'test':
            loader = self.validation_loader
        elif mode == 'train':
            loader = self.train_loader
        else:
            raise ValueError("mode must be either 'test' or 'train'")

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs, preds = None, None
                if self.nn_mode == 'classification':
                    outputs = self.softmax_f(self.model(inputs))
                    _, preds = torch.max(outputs, 1)
                elif self.nn_mode == 'regression':
                    preds = self.model(inputs)
                else:
                    raise ValueError("Invalid mode")
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        print(f"Accuracy on test set: {accuracy:.4f}")
        print(f"Precision on test set: {precision:.4f}")
        print(f"Recall on test set: {recall:.4f}")

        return accuracy, precision, recall
