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
        self.dataset = self.dataset.with_columns(cosL=(1 / (pl.col('tgL')**2 + 1)))
        self.dataset = self.dataset.with_columns(meanClSize=((pl.col('clSizeL0') + pl.col('clSizeL1') + pl.col('clSizeL2') + pl.col('clSizeL3') + pl.col('clSizeL4') + pl.col('clSizeL5') + pl.col('clSizeL6')) / 7))
        self.dataset = self.dataset.with_columns(clSizeCosL=(pl.col('meanClSize') * pl.col('cosL')))
        self.dataset = self.dataset.with_columns(
            nSigmaAbsDeu=abs(pl.col('nSigmaDeu')),
            nSigmaAbsP=abs(pl.col('nSigmaP')),
            nSigmaAbsK=abs(pl.col('nSigmaK')),
            nSigmaAbsPi=abs(pl.col('nSigmaPi')),
            nSigmaAbsE=abs(pl.col('nSigmaE'))
            )
        
        self.dataset = self.dataset.with_columns([
            pl.when(pl.col(f'clSizeL{layer}') < 0).then(np.nan).otherwise(pl.col(f'clSizeL{layer}')).alias(f'clSizeL{layer}')
            for layer in range(7)
            ])
        
        # deltaP
        self.dataset = self.dataset.with_columns(deltaP = (pl.col('pTPC') - pl.col('pITS')) / pl.col('pTPC'))

        # partID, mass, beta
        self.dataset = self.dataset.with_columns(
            partPDG=np.nan,
            partID=np.nan,
            mass=np.nan,
            beta=np.nan
            )
        
        for idx, part in enumerate(self.cfg['species']):    
            
            cfgTags = self.cfg['selTags'][part]
            self.dataset = self.dataset.with_columns(
                partPDG=pl.when((pl.col(f'nSigmaAbs{part}') < cfgTags['selfSel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part1"]}') > cfgTags['part1Sel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part2"]}') > cfgTags['part2Sel']),
                                (pl.col('p') < cfgTags['pmax'])).then(ParticlePDG[part]).otherwise(pl.col('partPDG')),
                partID=pl.when((pl.col(f'nSigmaAbs{part}') < cfgTags['selfSel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part1"]}') > cfgTags['part1Sel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part2"]}') > cfgTags['part2Sel']),
                                (pl.col('p') < cfgTags['pmax'])).then(idx).otherwise(pl.col('partID')),
                mass=pl.when((pl.col(f'nSigmaAbs{part}') < cfgTags['selfSel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part1"]}') > cfgTags['part1Sel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part2"]}') > cfgTags['part2Sel']),
                                (pl.col('p') < cfgTags['pmax'])).then(ParticleMasses[part]).otherwise(pl.col('mass')),
                beta=pl.when((pl.col(f'nSigmaAbs{part}') < cfgTags['selfSel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part1"]}') > cfgTags['part1Sel']),
                                (pl.col(f'nSigmaAbs{cfgTags["part2"]}') > cfgTags['part2Sel']),
                                (pl.col('p') < cfgTags['pmax'])).then(pl.col('p') / np.sqrt(pl.col('p')**2 + ParticleMasses[part]**2)).otherwise(pl.col('beta'))
                )
            
        # drop unidentified particles
        if self.mode == 'train':
            self.dataset = self.dataset.filter(pl.col('partID') != np.nan)

    def apply_quality_selection_cuts(self):
        '''
            Apply quality selection
        '''

        self.dataset = self.dataset.filter((pl.col('nClusTPC') > self.cfg['cuts']['nClusTPCmin']) & 
                                         (pl.col('chi2ITSTPC') < self.cfg['cuts']['chi2ITSTPCmax']) &
                                         (abs(pl.col('eta')) > self.cfg['cuts']['etamax'])
                                         )
        
    def normalize(self):
        '''
            Normalize the dataset
        '''
        
        for feature in self.features:
            mean = np.nanmean(self.normalized_dataset[feature])
            std = np.nanstd(self.normalized_dataset[feature])
            factor = 1.
            if 'clSizeL' in feature: factor = 2.
            self.normalized_dataset = self.normalized_dataset.with_columns(((pl.col(feature) - mean) / (factor * std)).alias(feature))
        
    def eliminate_nan(self):
        '''
            Eliminate NaN values
        '''
        
        self.normalized_dataset = self.normalized_dataset.with_columns([
            pl.when(pl.col(f'clSizeL{layer}') == np.nan).then(-1.).otherwise(pl.col(f'clSizeL{layer}')).alias(f'clSizeL{layer}')
            for layer in range(7)
            ])
    
    @property
    def class_weights(self):
        '''
            Compute the class weights
        '''
        
        weights = np.zeros(self.num_particles)
        for idx, part in enumerate(self.cfg['species']):
            weights[idx] = len(self.dataset.filter(pl.col('partID') == idx)) / len(self.dataset)
        
        return weights
    
    @property
    def class_counts(self):
        '''
            Compute the class counts
        '''
        
        counts = np.zeros(self.num_particles)
        for ipart, part in enumerate(self.cfg['species']):
            counts[ipart] = len(self.dataset.filter(pl.col('partID') == ipart))
        
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
            part_count = len(self.dataset.filter(pl.col('partID') == ipart))
            if part_count < max_count:
                n_new_samples = max_count - part_count
                filtered_ds = self.dataset.filter(pl.col('partID') == ipart)
                sampled_ds = filtered_ds.sample(n=n_new_samples, with_replacement=True)
                self.dataset = pl.concat([self.dataset, sampled_ds])



        

        