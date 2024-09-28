'''
    Class to handle and preprocess the data
'''

import copy
import numpy as np
import polars as pl
import yaml
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

from ROOT import TFile, TH1F, TH2F

import sys  
sys.path.append('..')
from core.load_data import LoadData
from utils.particles import ParticlePDG, ParticleMasses
from framework.utils.terminal_colors import TerminalColors as tc

class DataHandler(Dataset):

    def __init__(self, data, cfgFile: str, **kwargs):
        '''
            Initialize the class

            Parameters
            ----------
            data: list of input files or dataframe
            features (list): list of features (for machine learning)
            target (str): target variable (for machine learning)
        '''

        self.cfg_file = cfgFile
        with open(cfgFile, 'r') as file:
            print(f'Loading configuration file: '+tc.UNDERLINE+tc.BLUE+cfgFile+tc.RESET)
            self.cfg = yaml.safe_load(file)
        self.part_list = self.cfg['species']

        self.mode = kwargs.get('mode', 'train')

        if type(data) is list:
            self.dataset: pl.DataFrame = LoadData(data, **kwargs)
        else:
            self.dataset: pl.DataFrame = data
        self._new_columns(rigidity_he=kwargs.get('rigidity_he', False))
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
        return len(self.part_list)
    
    @staticmethod
    def rms_cl_size(mean, clsize_layers, n_hit_layers):
        terms = []
        for clsize_layer in clsize_layers:
            term = pl.when(clsize_layer > 0).then((clsize_layer - mean)**2).otherwise(0)
            terms.append(term)
        rms = (np.sum(terms) / n_hit_layers).sqrt()
        return rms

    def _new_columns(self, rigidity_he:bool=False):
        '''
            Create new columns in the dataset

            Parameters
            ----------
            rigidity_he (bool): correct the momentum for He3 
        '''
        
        # cluster size
        self.dataset = self.dataset.with_columns(fPAbs=np.abs(pl.col('fP')))
        self.dataset = self.dataset.with_columns(fCosL=(1 / np.cosh(pl.col('fEta'))))
        for layer in range(7):
            self.dataset = self.dataset.with_columns((np.right_shift(pl.col("fItsClusterSize"), (layer * 4)) & 0xF).alias(f'fItsClusterSizeL{layer}'))
        
        self.dataset = self.dataset.with_columns(fNClustersIts=((pl.col('fItsClusterSizeL0') > 0).cast(pl.Int32) + (pl.col('fItsClusterSizeL1') > 0).cast(pl.Int32) + (pl.col('fItsClusterSizeL2') > 0).cast(pl.Int32) + (pl.col('fItsClusterSizeL3') > 0).cast(pl.Int32) + (pl.col('fItsClusterSizeL4') > 0).cast(pl.Int32) + (pl.col('fItsClusterSizeL5') > 0).cast(pl.Int32) + (pl.col('fItsClusterSizeL6') > 0).cast(pl.Int32)))
        self.dataset = self.dataset.filter(pl.col('fNClustersIts') > 0)
        self.dataset = self.dataset.with_columns(fMeanItsClSize=((pl.col('fItsClusterSizeL0') + pl.col('fItsClusterSizeL1') + pl.col('fItsClusterSizeL2') + pl.col('fItsClusterSizeL3') + pl.col('fItsClusterSizeL4') + pl.col('fItsClusterSizeL5') + pl.col('fItsClusterSizeL6')) / pl.col('fNClustersIts')))
        self.dataset = self.dataset.with_columns(fClSizeCosL=(pl.col('fMeanItsClSize') * pl.col('fCosL')))
        self.dataset = self.dataset.with_columns(fRMSClSize=(self.rms_cl_size(pl.col('fMeanItsClSize'), [pl.col(f'fItsClusterSizeL{i}') for i in range(6)], pl.col('fNClustersIts'))))

        self.dataset = self.dataset.with_columns(pl.col('fPartID').apply(lambda x: ParticleMasses[self.part_list[x]]).alias('fMass'))

        # correct momentum for He3 (the value stored is rigidity)
        if 'He' in self.part_list and rigidity_he:
            self.dataset = self.dataset.with_columns(pl.when(pl.col('fPartID').eq(self.part_list.index('He'))).then(pl.col('fP') * 2).otherwise(pl.col('fP')).alias('fP'))
        self.dataset = self.dataset.with_columns(fPAbs=np.abs(pl.col('fP')))
        self.dataset = self.dataset.with_columns(fPt=pl.col('fP') / np.cosh(pl.col('fEta')))

        self.dataset = self.dataset.with_columns(fBeta=(pl.col('fP') / np.sqrt(pl.col('fPAbs')**2 + pl.col('fMass')**2)))
        self.dataset = self.dataset.with_columns(fBetaAbs=np.abs(pl.col('fBeta')))
        self.dataset = self.dataset.with_columns(fBetaGamma=(pl.col('fBetaAbs') / np.sqrt(1 - pl.col('fBeta')**2)))

    def drop_label_zero(self):
        '''
            Drop the label 0 from the dataset
        '''
        self.dataset = self.dataset.filter(pl.col('fPartID') != 0)
        self.normalized_dataset = self.normalized_dataset.filter(pl.col('fPartID') != 0)
        self.part_list = self.part_list.remove('Unidentified')

    def correct_for_pid_in_trk(self):
        '''
            Correct the He3 momentum based on the PID for tracking information
        '''
        print(tc.UNDERLINE+tc.BLUE+'Correcting momentum for He3 based on PID for tracking information'+tc.RESET)

        # check if the fPIDforTrk column is available
        if 'fPIDinTrk' not in self.dataset.columns:
            print(tc.RED+'fPIDinTrk column not available'+tc.RESET)
            return

        curveParams = {'kp0': -0.200281,
                       'kp1': 0.103039,
                       'kp2': -0.012325
                       }

        # change values only to rows where fPIDtrk == 6
        # pol2 correction
        self.dataset = self.dataset.with_columns(pl.when(pl.col('fPIDinTrk').eq(6)).then(pl.col('fPt') + pl.lit(curveParams['kp0']) + curveParams['kp1'] * pl.col('fPt') + curveParams['kp2'] * pl.col('fPt')**2).otherwise(pl.col('fPt')).alias('fPt'))

        # update fP, fPAbs and fBeta
        self.dataset = self.dataset.with_columns(fP=self.dataset['fPt'] * np.cosh(self.dataset['fEta']))
        self.dataset = self.dataset.with_columns(fPAbs=np.abs(self.dataset['fP']))
        self.dataset = self.dataset.with_columns(fBeta=(self.dataset['fP'] / np.sqrt(self.dataset['fPAbs']**2 + self.dataset['fMass']**2)))
        self.dataset = self.dataset.with_columns(fBetaAbs=np.abs(self.dataset['fBeta']))
        self.dataset = self.dataset.with_columns(fBetaGamma=(pl.col('fBetaAbs') / np.sqrt(1 - pl.col('fBeta')**2)))

    def clean_protons(self):
        '''
            Clean the protons in the dataset
        '''
        print(tc.UNDERLINE+tc.BLUE+'Cleaning protons'+tc.RESET)
        
        ds_protons = self.dataset.filter(pl.col('fPartID') == self.part_list.index('Pr'))
        BB_params = {
            'kp1': -0.031712,
            'kp2': -45.0275,
            'kp3': -0.997645,
            'kp4': 1.68228,
            'kp5': 0.0108484
        }

        def BBfunc(x):
            x = np.abs(x)
            beta = x / np.sqrt(1 + x**2)
            aa = beta**BB_params['kp4']
            bb = (1/x)**BB_params['kp5']
            bb = np.log(BB_params['kp3'] + bb)
            return (BB_params['kp2'] - aa - bb) * BB_params['kp1'] / aa
        ds_protons = ds_protons.with_columns(fExpClSizeCosLPr=BBfunc(pl.col('fBetaAbs')))
        sigma_params = {
            'kp0': 0.418451,
            'kp1': -0.040885
        }
        ds_protons = ds_protons.with_columns(fSigmaClSizeCosLPr=(pl.col('fExpClSizeCosLPr')*(sigma_params['kp0'] + sigma_params['kp1'] * pl.col('fExpClSizeCosLPr'))))
        ds_protons = ds_protons.with_columns(fNSigmaPr=((pl.col('fClSizeCosL') - pl.col('fExpClSizeCosLPr')) / pl.col('fSigmaClSizeCosLPr')))
        ds_protons = ds_protons.filter(pl.col('fNSigmaPr').abs() < 2)
        self.dataset = pl.concat([self.dataset.filter(pl.col('fPartID') != self.part_list.index('Pr')), ds_protons[self.dataset.columns]])

    def auto_normalize(self)-> Tuple[np.ndarray, np.ndarray]:
        '''
            Normalize the dataset
        '''

        means = []
        stds = []
        
        self.normalized_dataset = copy.deepcopy(self.dataset)
        for feature in self.features:
            mean = np.nanmean(self.normalized_dataset[feature])
            std = np.nanstd(self.normalized_dataset[feature])
            factor = 1.
            #if 'fItsClusterSizeL' in feature: factor = 2.
            self.normalized_dataset = self.normalized_dataset.with_columns(((pl.col(feature) - mean) / (factor * std)).alias(feature))
            means.append(mean)
            stds.append(std)

        return np.array(means), np.array(stds)

    def normalize(self, means:np.ndarray=None, stds:np.ndarray=None):

        if means is None or stds is None:
            means, stds = self.auto_normalize()
        
        self.normalized_dataset = copy.deepcopy(self.dataset)
        for ifeature, feature in enumerate(self.features):
            factor = 1.
            #if 'fItsClusterSizeL' in feature: factor = 2.
            self.normalized_dataset = self.normalized_dataset.with_columns(((pl.col(feature) - means[ifeature]) / (factor * stds[ifeature])).alias(feature))

    @property
    def class_weights(self):
        '''
            Compute the class weights
        '''
        
        weights = np.zeros(self.num_particles)
        for ipart, part in enumerate(self.part_list):
            weights[ipart] = len(self.dataset.filter(pl.col('fPartID') == ipart)) / len(self.dataset)
        
        return weights
    
    @property
    def class_counts(self):
        '''
            Compute the class counts
        '''
        
        counts = np.zeros(self.num_particles)
        for ipart, part in enumerate(self.part_list):
            counts[ipart] = len(self.dataset.filter(pl.col('fPartID') == ipart))
        
        return counts
        
    def train_test_split(self, test_size:float=0.2):
        '''
            Split the dataset into training and testing datasets

            Parameters
            ----------
            test_size (float): fraction of the dataset to be used for testing
        '''
        
        self.dataset = self.dataset.sample(fraction=1, shuffle=True)
        test_size = int(test_size * len(self.dataset))
        test_dataset, train_dataset = self.dataset.head(test_size), self.dataset.tail(-test_size)

        train_handler = DataHandler(train_dataset, self.cfg_file, mode='train')
        train_handler.part_list = self.part_list
        train_handler._new_columns()
        test_handler = DataHandler(test_dataset, self.cfg_file, mode='test')
        test_handler.part_list = self.part_list
        test_handler._new_columns()
        
        return train_handler, test_handler
    
    def class_oversample(self):
        '''
            Oversample the dataset
        '''
        
        max_count = max(self.class_counts)
        for ipart in self.dataset['fPartID'].unique():
            part_count = len(self.dataset.filter(pl.col('fPartID') == ipart))
            if part_count < max_count:
                n_new_samples = max_count - part_count
                filtered_ds = self.dataset.filter(pl.col('fPartID') == ipart)
                sampled_ds = filtered_ds.sample(n=n_new_samples, with_replacement=True)
                self.dataset = pl.concat([self.dataset, sampled_ds])

    def enhance_class(self, particle:str, factor:float):
        '''
            Enhance the dataset for a given particle

            Parameters
            ----------
            particle (str): particle
            factor (float): enhancement factor
        '''
        
        part_id = self.part_list.index(particle)
        part_count = len(self.dataset.filter(pl.col('fPartID') == part_id))
        n_new_samples = int(factor * part_count)
        filtered_ds = self.dataset.filter(pl.col('fPartID') == part_id)
        sampled_ds = filtered_ds.sample(n=n_new_samples, with_replacement=True)
        self.dataset = pl.concat([self.dataset, sampled_ds])

    def variable_oversample(self, variable:str, n_bins:int, var_min:float=None, var_max:float=None, max_per_bin:int=1000):
        '''
            Oversample the dataset based on a given variable. 
            Provides a dataset with the same number of samples for each bin of the variable
            (using bootstrap methods)

            Parameters
            ----------
            variable (str): variable
            n_bins (int): number of bins
        '''
        
        dss_part = []
        if var_min is None: var_min = self.dataset[variable].min()
        if var_max is None: var_max = self.dataset[variable].max()
        bin_width = (var_max - var_min) / n_bins
        for ipart, part in enumerate(self.part_list):
            ds_part = self.dataset.filter(pl.col('fPartID') == ipart)
            if len(ds_part) == 0: 
                continue
            ds_part_bins = []
            for ibin in range(n_bins):
                bin_ds = ds_part.filter(pl.col(variable).is_between(var_min + ibin * bin_width, var_min + (ibin+1) * bin_width))
                ds_part_bins.append(bin_ds)
            max_count = max([len(ds) for ds in ds_part_bins])
            max_count = min(max_count, max_per_bin)
            for ds in ds_part_bins:
                n_new_samples = max_count
                sampled_ds = ds.sample(n=n_new_samples, with_replacement=True)
                dss_part.append(sampled_ds)
        
        self.dataset = pl.concat(dss_part)

    def variable_and_class_flattening(self, variable:str, n_bins:int, var_min:float=None, var_max:float=None, n_per_bin:int=1000):
        '''
            Flatten the dataset based on a given variable for all classes.
            (using bootstrap methods)

            Parameters
            ----------
            variable (str): variable
            n_bins (int): number of bins
        '''
        
        dss_part = []
        if var_min is None: var_min = self.dataset[variable].min()
        if var_max is None: var_max = self.dataset[variable].max()
        bin_width = (var_max - var_min) / n_bins
        for ipart, part in enumerate(self.part_list):
            ds_part = self.dataset.filter(pl.col('fPartID') == ipart)
            if len(ds_part) == 0: 
                continue
            ds_part_bins = []
            for ibin in range(n_bins):
                bin_ds = ds_part.filter(pl.col(variable).is_between(var_min + ibin * bin_width, var_min + (ibin+1) * bin_width))
                sampled_ds = bin_ds.sample(n=n_per_bin, with_replacement=True)
                dss_part.append(sampled_ds)
            
        self.dataset = pl.concat(dss_part)

    def data_augmentation(self, particle:str, n_bins:int, beta_min:float=0., beta_max:float=1., maximum_new_samples:int=10000):
        '''
            Augment the dataset based on a given beta interval for a given particle.
            (using bootstrap methods from other species)

            Parameters
            ----------
            particle (str): particle
        '''
        
        part_id = self.part_list.index(particle)
        ds_for_augmentation = self.dataset.filter(pl.col('fPartID') != part_id)
        ds_for_augmentation = ds_for_augmentation.filter(pl.col('fBetaAbs').is_between(beta_min, beta_max))
        if len(ds_for_augmentation) == 0: 
            return
        n_new_samples = min(maximum_new_samples, len(ds_for_augmentation))
        sampled_ds = ds_for_augmentation.sample(n=n_new_samples, with_replacement=True)
        sampled_ds = sampled_ds.with_columns(fPartID=pl.lit(part_id, dtype=pl.UInt8))
        self.dataset = pl.concat([self.dataset, sampled_ds])
        self._new_columns()
            

    def reduced_dataset(self, n_samples:int):
        '''
            Reduce the dataset to a given number of samples

            Parameters
            ----------
            n_samples (int): number of samples
        '''
        
        self.dataset = self.dataset.sample(n=n_samples)
        self.normalized_dataset = self.normalized_dataset.sample(n=n_samples)

    def select_species(self, species: List[str]):
        '''
            Select a given species

            Parameters
            ----------
            species (str): species
        '''
        
        species_index = [self.part_list.index(sp) for sp in species]
        self.dataset = self.dataset.filter(pl.col('fPartID').is_in(species_index))
        self.normalized_dataset = self.normalized_dataset.filter(pl.col('fPartID').is_in(species_index))
        self.part_list = species

    def rename_classes(self):
        '''
            Rename the coumn fPartID so that the classes start from 0
        '''

        available_classes = self.dataset['fPartID'].unique()
        for iclass, class_id in enumerate(available_classes):
            print(f'Class {self.part_list[iclass]}: {class_id} -> {iclass}')
            self.dataset = self.dataset.with_columns(pl.when(pl.col('fPartID').eq(class_id)).then(iclass).otherwise(pl.col('fPartID')).alias('fPartID'))
            self.normalized_dataset = self.normalized_dataset.with_columns(pl.when(pl.col('fPartID').eq(class_id)).then(iclass).otherwise(pl.col('fPartID')).alias('fPartID'))