'''
    Class to handle and preprocess the data
'''

import copy
import numpy as np
import polars as pl
import yaml
import torch
from torch.utils.data import Dataset

import sys  
sys.path.append('..')
from core.load_data import LoadData
from utils.particles import ParticlePDG, ParticleMasses
from framework.utils.terminal_colors import TerminalColors as tc

class DataHandler(Dataset):

    def __init__(self, dataFiles: list, cfgFile: str, **kwargs):
        '''
            Initialize the class

            Parameters
            ----------
            dataFiles (list): list of input files
            features (list): list of features (for machine learning)
            target (str): target variable (for machine learning)
        '''

        with open(cfgFile, 'r') as file:
            print(f'Loading configuration file: '+tc.UNDERLINE+tc.BLUE+cfgFile+tc.RESET)
            self.cfg = yaml.safe_load(file)

        self.mode = kwargs.get('mode', 'train')

        self.dataset: pl.DataFrame = LoadData(dataFiles)
        self._new_columns()
        self.normalized_dataset: pl.DataFrame = copy.deepcopy(self.dataset)
        self.features = self.cfg['features']
        self.target = self.cfg['target']

        self.one_hot_encode = self.cfg['oneHot']
        self.use_classes = self.cfg['useClasses']


    def __len__(self):
        return self.dataset.shape[0]
        
    def __getitem__(self, idx):
        '''
            Get the data for a given index

            Parameters
            ----------
            idx (int): index
        '''
        # Convert features and target to numpy arrays and then to torch tensors
        features = self.normalized_dataset[self.features][idx].to_numpy()
        target = self.normalized_dataset[self.target][idx]
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Convert target to one-hot encoding
        if self.one_hot_encode:
            target_tensor = torch.zeros(self.num_particles, dtype=torch.long)
            target_tensor[int(target)] = 1
        elif self.use_classes:
            target_tensor = torch.tensor(target, dtype=torch.long)
        else:
            target_tensor = torch.tensor(target, dtype=torch.float32)
    
        return features_tensor, target_tensor
    
    @property
    def num_particles(self):
        return len(self.cfg['species'])

    def _new_columns(self):
        '''
            Create new columns in the dataset

            Parameters
            ----------
            columns (list): list of columns
        '''
        
        # cluster size
        self.dataset = self.dataset.with_columns(fPAbs=np.abs(pl.col('fP')))
        self.dataset = self.dataset.with_columns(fPt=pl.col('fP') / np.cosh(pl.col('fEta')))
        self.dataset = self.dataset.with_columns(fCosL=(1 / np.cosh(pl.col('fEta'))))
        for layer in range(7):
            read4Bits = lambda x, layer: (x >> layer*4) & 0b1111
            self.dataset = self.dataset.with_columns(pl.col('fItsClusterSize').apply(read4Bits, args=(layer,))).alias(f'fItsClusterSizeL{layer}')
        self.dataset = self.dataset.with_columns(fMeanItsClSize=((pl.col('fItsClusterSizeL0') + pl.col('fItsClusterSizeL1') + pl.col('fItsClusterSizeL2') + pl.col('fItsClusterSizeL3') + pl.col('fItsClusterSizeL4') + pl.col('fItsClusterSizeL5') + pl.col('fItsClusterSizeL6')) / 7))
        self.dataset = self.dataset.with_columns(fClSizeCosL=(pl.col('fMeanItsClSize') * pl.col('fCosL')))
        
        #self.dataset = self.dataset.with_columns([
        #    pl.when(pl.col(f'fItsClusterSizeL{layer}') < 0).then(np.nan).otherwise(pl.col(f'clSizeL{layer}')).alias(f'clSizeL{layer}')
        #    for layer in range(7)
        #    ])

        self.dataset = self.dataset.with_columns(pl.col('fPartID').apply(lambda x: ParticleMasses[self.cfg['species'][x+1]])).alias('fMass')
        self.dataset = self.dataset.with_columns(fBeta=(pl.col('fP') / np.sqrt(pl.col('fPAbs')**2 + pl.col('fMass')**2)))
        self.dataset = self.dataset.with_columns(fBetaAbs=np.abs(pl.col('fBeta')))

        # drop unidentified particles
        if self.mode == 'train':
            self.dataset = self.dataset.filter(pl.col('fPartID') != 0)
        
    def normalize(self):
        '''
            Normalize the dataset
        '''
        
        for feature in self.features:
            mean = np.nanmean(self.normalized_dataset[feature])
            std = np.nanstd(self.normalized_dataset[feature])
            factor = 1.
            if 'fItsClusterSizeL' in feature: factor = 2.
            self.normalized_dataset = self.normalized_dataset.with_columns(((pl.col(feature) - mean) / (factor * std)).alias(feature))
    
    @property
    def class_weights(self):
        '''
            Compute the class weights
        '''
        
        weights = np.zeros(self.num_particles)
        for ipart, part in enumerate(self.cfg['species']):
            weights[ipart] = len(self.dataset.filter(pl.col('fPartID') == ipart+1)) / len(self.dataset)
        
        return weights
    
    @property
    def class_counts(self):
        '''
            Compute the class counts
        '''
        
        counts = np.zeros(self.num_particles)
        for ipart, part in enumerate(self.cfg['species']):
            counts[ipart] = len(self.dataset.filter(pl.col('partID') == ipart+1))
        
        return counts
        
    def train_test_split(self, test_size:float=0.2):
        '''
            Split the dataset into training and testing datasets

            Parameters
            ----------
            test_size (float): fraction of the dataset to be used for testing
        '''
        
        mask = np.random.rand(len(self.normalized_dataset)) < (1 - test_size)
        train_dataset = self.normalized_dataset[mask]
        test_dataset = self.normalized_dataset[~mask]
        
        return train_dataset, test_dataset
    
    def oversample(self):
        '''
            Oversample the dataset
        '''
        
        max_count = max(self.class_counts)
        for ipart, part in enumerate(self.cfg['species']):
            part_count = len(self.dataset.filter(pl.col('fPartID') == ipart+1))
            if part_count < max_count:
                n_new_samples = max_count - part_count
                filtered_ds = self.dataset.filter(pl.col('fPartID') == ipart+1)
                sampled_ds = filtered_ds.sample(n=n_new_samples, with_replacement=True)
                self.dataset = pl.concat([self.dataset, sampled_ds])



        

        