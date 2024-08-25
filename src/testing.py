'''
    Function to train the model
'''

import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from core.dataset import DataHandler
from core.fcnn import FCNN
from core.tester import NeuralNetworkTester
from framework.utils.terminal_colors import TerminalColors as tc
from framework.utils.timeit import timeit

from ROOT import TFile, TCanvas, TDirectory

@timeit
def test_model(train_handler:DataHandler, test_handler:DataHandler, output_file_rooot:TDirectory, model, cfg_data_file:str):
    '''
        Test the model

        Parameters
        ----------
        train_handler (DataHandler): training dataset
        test_handler (DataHandler): testing dataset
        model (torch.nn.Module): model
        cfg (dict): configuration
    '''

    print(tc.BOLD+tc.GREEN+'Test the model'+tc.RESET)
    with open(cfg_data_file, 'r') as file:
        cfg_data = yaml.safe_load(file)
    
    loss = nn.CrossEntropyLoss()

    BATCH_SIZE = cfg_data['batch_size']
    NN_MODE = cfg_data['nn_mode']
    train_tester = NeuralNetworkTester(model, train_handler, loss, NN_MODE, BATCH_SIZE)
    test_tester = NeuralNetworkTester(model, test_handler, loss, NN_MODE, BATCH_SIZE)

    print('Make predictions')
    train_tester.make_predictions()
    #test_tester.make_predictions()


    # confusion matrix
    print('Draw confusion matrix')
    train_tester.draw_confusion_matrix(output_file_root)
    #test_tester.draw_confusion_matrix()

    # separation for pair of classes
    print('Draw class separation')
    for ipart, part in enumerate(train_handler.part_list):
        train_tester.class_separation(ipart, part, output_file_root)
        #test_tester.class_separation(ipart, part, output_file)

   

if __name__ == '__main__':

    from data_preparation import data_preparation

    input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = '../config/config_outputs.yml'
    cfg_output_file = '../config/config_outputs.yml'
    
    
    output_file_path = '../output/testing_nn.root'
    print('Output file: '+tc.UNDERLINE+tc.BLUE+output_file_path+tc.RESET)
    output_file_root = TFile(output_file_path, 'RECREATE')

    train_species = ['Pi', 'Ka', 'Pr']
    #train_handler, test_handler = data_preparation(input_files, output_file_root, cfg_data_file, cfg_output_file, normalize=True, rename_classes=False, force_option='AO2D', split=True, species_selection=[True, train_species], variable_selection=[True, 'fPAbs', 0.3, 1.0], clean_protons=True, oversample_momentum=True, n_samples=int(1e5), oversample=True)
    train_handler, validation_handler, test_handler = data_preparation(input_files, output_file_root, cfg_data_file, cfg_output_file, force_option='AO2D', 
                                                   normalize=True, 
                                                   rename_classes=True, 
                                                   split=True, 
                                                   species_selection=[True, train_species], 
                                                   #variable_selection=[True, 'fPAbs', 0., 1.0], 
                                                   flatten_samples=['fPAbs', 250, 0., 2.5, 500], 
                                                   clean_protons=True, 
                                                   oversample_momentum=[True, 'fPAbs', 500, 0., 5.], 
                                                   n_samples=None, 
                                                   oversample=False)

    INPUT_SIZE = len(train_handler.features)
    OUTPUT_SIZE = train_handler.num_particles
    print('Output size: ', OUTPUT_SIZE)
    model = FCNN(INPUT_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load('../models/fc_model.pth'))

    test_model(train_handler, test_handler, output_file_root, model, cfg_data_file)

    output_file_root.Close()