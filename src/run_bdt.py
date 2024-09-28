'''
    Function to train a BDT from xgboost
'''

import os
import logging
import numpy as np
import xgboost as xgb
import polars as pl
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from typing import Tuple, List
import sys
sys.path.append('..')
from core.dataset import DataHandler
from core.bdt import BDTRegressorTrainer, BDTClassifierTrainer, BDTClassifierEnsembleTrainer
from framework.utils.terminal_colors import TerminalColors as tc
from framework.utils.timeit import timeit
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from framework.utils.root_setter import obj_setter
from framework.utils.matplotlib_to_root import save_mpl_to_root
import pickle

from ROOT import TFile, TDirectory, TH1F, TH2F

def run_regressor(input_files: List[str], input_files_he: List[str], input_files_pkpi: List[str], cfg_data_file:str, cfg_output_file:str, output_file:str, train:bool):

    output_file_root = TFile(output_file, 'RECREATE')
    outdir = output_file_root.mkdir('bdt_output')

    #train_species = ['Pi', 'Ka', 'Pr', 'De', 'He']
    train_species = ['Pi', 'Ka', 'Pr']
    #train_species = ['Pi', 'Pr', 'De']
    
    #train_handler, validation_handler, test_handler = data_preparation_with_he(input_files, input_files_he, output_file_root, cfg_data_file, cfg_output_file, force_option='AO2D',
    train_handler, validation_handler, test_handler = data_preparation(input_files, output_file_root, cfg_data_file, cfg_output_file, force_option='AO2D',
                                                            normalize=True, 
                                                            rename_classes=True, 
                                                            split=True, 
                                                            species_selection=[True, train_species], 
                                                            data_augmentation=[True, ['Ka', 'Pi'], 0.3, 0.5, 100000],
                                                            flatten_samples=['fPAbs', 200, 0.5, 2.5, 500],
                                                            variable_selection=[True, 'fPAbs', 0.5, 2.5], 
                                                            clean_protons=True, 
                                                            n_samples=None, 
                                                            oversample=False,
                                                            #minimum_hits=5,
                                                            #test_path=input_files_pkpi,
                                                            debug=False
                                                            )

    ## Regressor
    cfg_bdt_file = '../config/config_bdt_reg.yml'
    bdt_regressor = BDTRegressorTrainer(cfg_bdt_file, outdir)
    bdt_regressor.load_data(train_handler, validation_handler, test_handler, normalized=True)
    #bdt_regressor.hyperparameter_optimization()
    if train:
        bdt_regressor.train()
        bdt_regressor.save_model()
    else:
        bdt_regressor.load_model()
    bdt_regressor.evaluate()
    bdt_regressor.prepare_for_plots()

    bdt_regressor.save_output()
    bdt_regressor.draw_beta_distribution()
    bdt_regressor.draw_part_id_distribution()
    bdt_regressor.draw_delta_beta_distribution()

    output_file_root.Close()

def run_classifier(input_files: List[str], input_files_he: List[str], cfg_data_file:str, cfg_output_file:str, output_file:str, train:bool):

    output_file_root = TFile(output_file, 'RECREATE')
    outdir = output_file_root.mkdir('bdt_output')

    train_species = ['Pi', 'Ka', 'Pr'] #, 'De', 'He']
    train_handler, validation_handler, test_handler = data_preparation(input_files, outdir, cfg_data_file, cfg_output_file, force_option='AO2D',
                                                            #input_files_he=input_files_he,
                                                            normalize=True, 
                                                            rename_classes=True, 
                                                            split=True, 
                                                            species_selection=[True, train_species], 
                                                            variable_selection=[True, 'fPAbs', 0.35, 1.05], 
                                                            flatten_samples=['fPAbs', 70, 0.35, 1.05, 2000], 
                                                            clean_protons=True, 
                                                            n_samples=None, 
                                                            oversample=False,
                                                            #minimum_hits=7
                                                            )

    ## Classifier
    cfg_bdt_file = '../config/config_bdt_cls.yml'
    bdt_classifier = BDTClassifierTrainer(cfg_bdt_file, output_file_root)
    bdt_classifier.load_data(train_handler, validation_handler, test_handler, normalized=True)
    print(train_handler.dataset['fPartID'].unique())
    bdt_classifier.hyperparameter_optimization()
    if train:
        bdt_classifier.train()
        bdt_classifier.save_model()
    else:
        bdt_classifier.load_model()
    bdt_classifier.evaluate()
    bdt_classifier.prepare_for_plots()

    bdt_classifier.save_output()
    bdt_classifier.draw_class_scores()
    #bdt_classifier.draw_part_id_distribution()
    #bdt_classifier.draw_efficiency_purity_vs_momentum(0.3, 1.0, 0.1)
    #bdt_classifier.draw_confusion_matrix()

    output_file_root.Close()

def run_classifier_ensemble(input_files: List[str], input_files_he: List[str], cfg_data_file:str, cfg_output_file:str, output_file:str, train:bool):

    output_file_root = TFile(output_file, 'RECREATE')
    outdir = output_file_root.mkdir('bdt_output')

    train_species = ['Pi', 'Ka', 'Pr']#, 'De', 'He']
    train_handler, validation_handler, test_handler = data_preparation(input_files, output_file_root, cfg_data_file, cfg_output_file, #input_files_he=input_files_he, 
                                                            force_option='AO2D',
                                                            normalize=False, 
                                                            rename_classes=True, 
                                                            split=True, 
                                                            species_selection=[True, train_species], 
                                                            variable_selection=[True, 'fPAbs', 0.35, 1.05], 
                                                            flatten_samples=['fPAbs', 70, 0.35, 1.05, 2000], 
                                                            clean_protons=True, 
                                                            n_samples=None, 
                                                            oversample=False,
                                                            minimum_hits=7
                                                            )

    ## Classifier
    cfg_bdt_file = '../config/config_bdt_cls.yml'
    bdt_classifier_ensemble = BDTClassifierEnsembleTrainer(cfg_bdt_file, output_file_root, momentum_bins=[0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05])
    bdt_classifier_ensemble.load_data(train_handler, validation_handler, test_handler, normalized=False)
    print(train_handler.dataset['fPartID'].unique())
    #bdt_classifier_ensemble.hyperparameter_optimization()
    if train:
        bdt_classifier_ensemble.train()
        bdt_classifier_ensemble.save_models()
        #return
    else:
        bdt_classifier_ensemble.load_model()
    bdt_classifier_ensemble.evaluate()
    bdt_classifier_ensemble.prepare_for_plots()

    #bdt_classifier_ensemble.save_output()
    bdt_classifier_ensemble.draw_class_scores()
    bdt_classifier_ensemble.draw_part_id_distribution()
    #bdt_classifier_ensemble.draw_feature_importance()
    #bdt_classifier_ensemble.draw_efficiency_purity_vs_momentum(0.3, 1.0, 0.1)
    #bdt_classifier_ensemble.draw_confusion_matrix()

    output_file_root.Close()


if __name__ == '__main__':
    
    from data_preparation import data_preparation

    # Configure logging
    os.remove("/home/galucia/ITS_pid/output/output_bdt.log")
    logging.basicConfig(filename="/home/galucia/ITS_pid/output/output_bdt.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    #input_files = ['../../data/0720/its_PIDStudy.root']
    input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root']
    input_files_he = ['/data/galucia/its_pid/LHC23_pass4_skimmed/LHC23_pass4_skimmed.root']
    input_files_pkpi = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_slice_pkpi.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = '../config/config_outputs.yml'

    #run_regressor(input_files, input_files_he, input_files_pkpi, cfg_data_file, cfg_output_file, output_file='../output/bdt_beta_05.root', train=True)
    #run_classifier(input_files, input_files_he, cfg_data_file, cfg_output_file, output_file='../output/bdt_cls_30082024_single.root', train=True)
    run_classifier_ensemble(input_files, input_files_he, cfg_data_file, cfg_output_file, output_file='../output/bdt_cls_11092024_7hits2.root', train=True)
                                                                                                     #'../output/bdt_cls_30082024.root', train=True)

