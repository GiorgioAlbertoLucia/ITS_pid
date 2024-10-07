'''
    Function to train a BDT from xgboost
'''

import os
import argparse
import logging
import matplotlib.pyplot as plt
import yaml
from typing import List
import sys
sys.path.append('..')
from core.bdt import BDTRegressorTrainer, BDTClassifierTrainer, BDTClassifierEnsembleTrainer
from framework.utils.terminal_colors import TerminalColors as tc

from ROOT import TFile, TDirectory, TH1F, TH2F

def run_regressor(cfg, cfg_data_file:str, cfg_output_file:str):

    output_file_root = TFile(cfg['output_file'], 'RECREATE')
    outdir = output_file_root.mkdir('bdt_output')

    train_handler, validation_handler, test_handler = data_preparation(cfg['input_files'], output_file_root, cfg_data_file, cfg_output_file, force_option='AO2D',
                                                            input_files_he=cfg['input_files_he'],
                                                            normalize=True, 
                                                            rename_classes=True, 
                                                            split=True, 
                                                            species_selection=[True, cfg['train_species']], 
                                                            data_augmentation=[True, ['Ka', 'Pi'], 0.3, 0.5, 100000],
                                                            flatten_samples=['fPAbs', 200, 0.5, 2.5, 500],
                                                            variable_selection=[True, 'fPAbs', 0.5, 2.5], 
                                                            clean_protons=True, 
                                                            n_samples=None, 
                                                            oversample=False,
                                                            #minimum_hits=cfg['minimum_ITS_hits'],
                                                            #test_path=cfg['input_files_pkpi'],
                                                            debug=cfg['debug']
                                                            )

    ## Regressor
    bdt_regressor = BDTRegressorTrainer(cfg['cfg_bdt_file'], outdir)
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

def run_classifier(cfg, cfg_data_file:str, cfg_output_file:str):

    output_file_root = TFile(cfg['output_file'], 'RECREATE')
    outdir = output_file_root.mkdir('bdt_output')

    train_handler, validation_handler, test_handler = data_preparation(cfg['input_files'], outdir, cfg_data_file, cfg_output_file, force_option='AO2D',
                                                            input_files_he=cfg['input_files_he'],
                                                            normalize=True, 
                                                            rename_classes=True, 
                                                            split=True, 
                                                            species_selection=[True, cfg['train_species']], 
                                                            variable_selection=[True, 'fPAbs', 0.35, 1.05], 
                                                            flatten_samples=['fPAbs', 70, 0.35, 1.05, 2000], 
                                                            clean_protons=True, 
                                                            n_samples=None, 
                                                            oversample=False,
                                                            #minimum_hits=cfg['minimum_ITS_hits']
                                                            )

    ## Classifier
    bdt_classifier = BDTClassifierTrainer(cfg['cfg_bdt_file'], output_file_root)
    bdt_classifier.load_data(train_handler, validation_handler, test_handler, normalized=True)
    print(train_handler.dataset['fPartID'].unique())
    bdt_classifier.hyperparameter_optimization()
    if cfg['train']:
        bdt_classifier.train()
        bdt_classifier.save_model()
    else:
        bdt_classifier.load_model()
    bdt_classifier.evaluate()
    bdt_classifier.prepare_for_plots()

    bdt_classifier.save_output()
    bdt_classifier.draw_class_scores()
    bdt_classifier.draw_part_id_distribution()
    momentum_bins = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    #bdt_classifier.draw_efficiency_purity(momentum_bins)
    

    output_file_root.Close()

def run_classifier_ensemble(cfg, cfg_data_file:str, cfg_output_file:str):

    output_file_root = TFile(cfg['output_file'], 'RECREATE')
    outdir = output_file_root.mkdir('bdt_output')

    train_handler, validation_handler, test_handler = data_preparation(cfg['input_files'], output_file_root, cfg_data_file, cfg_output_file, 
                                                            input_files_he=cfg['input_files_he'], 
                                                            force_option='AO2D',
                                                            normalize=False, 
                                                            rename_classes=True, 
                                                            split=True, 
                                                            species_selection=[True, cfg['train_species']], 
                                                            variable_selection=[True, 'fPAbs', 0.35, 1.05], 
                                                            flatten_samples=['fPAbs', 70, 0.35, 1.05, 2000], 
                                                            clean_protons=True, 
                                                            n_samples=None, 
                                                            oversample=False,
                                                            minimum_hits=cfg['minimum_ITS_hits']
                                                            )

    ## Classifier
    bdt_classifier_ensemble = BDTClassifierEnsembleTrainer(cfg['cfg_bdt_file'], output_file_root, momentum_bins=cfg['momentum_bins'])
    bdt_classifier_ensemble.load_data(train_handler, validation_handler, test_handler, normalized=False)
    print(train_handler.dataset['fPartID'].unique())
    #bdt_classifier_ensemble.hyperparameter_optimization()
    if cfg['train']:
        bdt_classifier_ensemble.train()
        bdt_classifier_ensemble.save_models()
        #return
    else:
        bdt_classifier_ensemble.load_model()
    bdt_classifier_ensemble.evaluate()
    bdt_classifier_ensemble.prepare_for_plots()

    bdt_classifier_ensemble.save_output()
    bdt_classifier_ensemble.draw_class_scores()
    bdt_classifier_ensemble.draw_part_id_distribution()
    bdt_classifier_ensemble.draw_efficiency_purity()

    output_file_root.Close()


if __name__ == '__main__':
    
    from data_preparation import data_preparation

    # Configure logging
    os.remove("/home/galucia/ITS_pid/output/output_bdt.log")
    logging.basicConfig(filename="/home/galucia/ITS_pid/output/output_bdt.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = '../config/config_outputs.yml'

    parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
    parser.add_argument('--config-file', dest='cfgInputFile',
                        help='path to the YAML file with configuration.', default='')
    args = parser.parse_args()

    if args.cfgInputFile == '':
        print(tc.RED+'[ERROR]: '+tc.RESET+'No config file provided, exiting.')
        exit(1)

    cfg = yaml.safe_load(open(args.cfgInputFile, 'r'))

    if cfg['mode'] == 'regressor':              run_regressor(cfg, cfg_data_file, cfg_output_file)
    elif cfg['mode'] == 'classifier':           run_classifier(cfg, cfg_data_file, cfg_output_file)
    elif cfg['mode'] == 'classifier_ensemble':  run_classifier_ensemble(cfg, cfg_data_file, cfg_output_file)
    else:
        print(tc.RED+'[ERROR]: '+tc.RESET+'Invalid mode, exiting.')
        exit(1)