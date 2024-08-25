'''
    Class to handle testing of the model
'''

import copy
import tqdm
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from ROOT import TFile, TCanvas
import sys
sys.path.append('..')
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from core.dataset import DataHandler
from framework.utils.matplotlib_to_root import save_mpl_to_root


class NeuralNetworkTester:
    def __init__(self, model, data_handler:DataHandler, loss, nn_mode, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.data_handler = data_handler
        self.data_loader = DataLoader(data_handler, batch_size=batch_size, shuffle=True)
        self.loss = loss
        self.nn_mode = nn_mode
        self.softmax_f = nn.Softmax(dim=1)

    def evaluate(self, mode:str='test'):
        self.model.eval()
        all_preds = []
        all_labels = []

        loader = None
        if mode == 'test':
            loader = self.test_loader
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
                    preds = torch.argmax(outputs, 1)
                elif self.nn_mode == 'regression':
                    preds = self.model(inputs)
                else:
                    raise ValueError("Invalid model type")
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        print(f"Accuracy on test set: {accuracy:.4f}")
        print(f"Precision on test set: {precision:.4f}")
        print(f"Recall on test set: {recall:.4f}")

        return accuracy, precision, recall
    
    def make_predictions(self):
        '''
            Add a output column (with all the predictions) to the dataset
        '''

        self.model.eval()
        all_preds = []
        predictions = []

        with torch.no_grad():
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs, preds = None, None
                if self.nn_mode == 'classification':
                    outputs = self.softmax_f(self.model(inputs))
                    preds = torch.argmax(outputs, 1)
                elif self.nn_mode == 'regression':
                    preds = self.model(inputs)
                else:
                    raise ValueError("Invalid model type")
                
                all_preds.extend(outputs.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
        
        if self.nn_mode == 'classification':
            for iclass in range(len(all_preds[0])):
                column_name = f'pred_class{iclass}'
                iclass_pred = [pred[iclass] for pred in all_preds]
                print(f'Adding column {column_name} to the dataset')
                self.data_handler.dataset = self.data_handler.dataset.with_columns(pl.Series(name=column_name, values=iclass_pred))
            print(f'Adding column prediction to the dataset')
            self.data_handler.dataset = self.data_handler.dataset.with_columns(pl.Series(name='prediction', values=predictions))
        elif self.nn_mode == 'regression':
            print(f'Adding column prediction to the dataset')
            self.data_handler.dataset = self.data_handler.dataset.with_columns(pl.Series(name='prediction', values=predictions))
        else:
            raise ValueError("Invalid model type")
        
    
    def draw_confusion_matrix(self, output_file:TFile):
        '''
            Draw the confusion matrix
        '''
        assert 'prediction' in self.data_handler.dataset.columns, 'No predictions found in the dataset. Run make_predictions() first'

        # create confusion matrix
        conf_matrix = []
        for itrue_species in range(self.data_handler.num_particles):
            row = []
            ispecies_tot = self.data_handler.dataset.filter(pl.col('fPartID') == itrue_species).shape[0]
            for ipred_species in range(self.data_handler.num_particles):
                ipred_pos = self.data_handler.dataset.filter((pl.col('fPartID') == itrue_species) & (pl.col('prediction') == ipred_species)).shape[0]
                if ispecies_tot != 0:
                    row.append(ipred_pos/ispecies_tot)
                else:
                    row.append(0)
            conf_matrix.append(row)

        conf_matrix = np.array(conf_matrix)

        # plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        save_mpl_to_root(fig, output_file, 'confusion_matrix')


    def class_separation(self, ispecies:int, species:str,  output_file:TFile):
        '''
            Plot the separation for a species versus all other species. Classification only.
        '''
        assert 'prediction' in self.data_handler.dataset.columns, 'No predictions found in the dataset. Run make_predictions() first'

        print(self.data_handler.dataset[[f'pred_class{iclass}' for iclass in range(self.data_handler.num_particles)]].describe())
        pos_axis_spec = AxisSpec(44, 0, 1.1, f'pos_pred_class_{ispecies}', f'{species}: prediction for class {ispecies}; probability; counts')
        neg_axis_spec = AxisSpec(44, 0, 1.1, f'neg_pred_class_{ispecies}', f'all: prediction for class {ispecies}; probability; counts')
        
        pos_dataset = self.data_handler.dataset.filter((pl.col('fPartID') == ispecies))
        neg_dataset = self.data_handler.dataset.filter((pl.col('fPartID') != ispecies))

        pos_handler = HistHandler.createInstance(pos_dataset)
        pos_hist = pos_handler.buildTH1(f'pred_class{ispecies}', pos_axis_spec)
        pos_hist.SetFillColorAlpha(797, 0.5)

        neg_handler = HistHandler.createInstance(neg_dataset)
        neg_hist = neg_handler.buildTH1(f'pred_class{ispecies}', neg_axis_spec)
        neg_hist.SetFillColorAlpha(867, 0.5)

        canvas = TCanvas(f'{species}_vs_all', f'{species} vs all; probability; counts', 800, 600)
        pos_hist.Draw('hist f')
        neg_hist.Draw('hist f same')
        output_file.cd()
        canvas.BuildLegend(0.55, 0.6, 0.9, 0.9)
        canvas.SetLogy()
        canvas.Write()

    def prediction_separation(self, ispecies:int, species:str,  output_file:TFile):
        '''
            Plot the separation for a species versus all other species. Classification only.
        '''
        assert 'prediction' in self.data_handler.dataset.columns, 'No predictions found in the dataset. Run make_predictions() first'

        print(self.data_handler.dataset.describe())
        pos_axis_spec = AxisSpec(44, 0, 1.1, f'pos_pred_class_{ispecies}', f'{species}: prediction for class {ispecies}; probability; counts')
        neg_axis_spec = AxisSpec(44, 0, 1.1, f'neg_pred_class_{ispecies}', f'all: prediction for class {ispecies}; probability; counts')
        
        pos_dataset = self.data_handler.dataset.filter((pl.col('fPartID') == ispecies))
        neg_dataset = self.data_handler.dataset.filter((pl.col('fPartID') != ispecies))

        pos_handler = HistHandler.createInstance(pos_dataset)
        pos_hist = pos_handler.buildTH1(f'prediction', pos_axis_spec)
        pos_hist.SetFillColorAlpha(797, 0.5)

        neg_handler = HistHandler.createInstance(neg_dataset)
        neg_hist = neg_handler.buildTH1(f'prediction', neg_axis_spec)
        neg_hist.SetFillColorAlpha(867, 0.5)

        canvas = TCanvas(f'{species}_vs_all', f'{species} vs all; probability; counts', 800, 600)
        pos_hist.Draw('hist')
        neg_hist.Draw('hist same')
        output_file.cd()
        canvas.BuildLegend(0.55, 0.6, 0.9, 0.9)
        canvas.SetLogy()
        canvas.Write() 

