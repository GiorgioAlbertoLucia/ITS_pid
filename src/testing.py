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

from ROOT import TFile, TCanvas

@timeit
def test_model(train_dataset:DataHandler, test_dataset:DataHandler, model, cfg_data_file:str):
    '''
        Test the model

        Parameters
        ----------
        train_dataset (DataHandler): training dataset
        test_dataset (DataHandler): testing dataset
        model (torch.nn.Module): model
        cfg (dict): configuration
    '''

    print(tc.BOLD+tc.GREEN+'Test the model'+tc.RESET)
    with open(cfg_data_file, 'r') as file:
        cfg_data = yaml.safe_load(file)
    
    loss = nn.CrossEntropyLoss()

    BATCH_SIZE = cfg_data['batch_size']
    NN_MODE = cfg_data['nn_mode']
    train_tester = NeuralNetworkTester(model, train_dataset, loss,NN_MODE, BATCH_SIZE)
    test_tester = NeuralNetworkTester(model, test_dataset, loss,NN_MODE, BATCH_SIZE)

    print('Make predictions')
    train_tester.make_predictions()
    #test_tester.make_predictions()

    output_file_path = '../output/testing.root'
    print('Output file: '+tc.UNDERLINE+tc.BLUE+output_file_path+tc.RESET)
    output_file = TFile(output_file_path, 'RECREATE')

    # confusion matrix
    print('Draw confusion matrix')
    train_tester.draw_confusion_matrix(output_file)
    #test_tester.draw_confusion_matrix()

    # separation for pair of classes
    print('Draw class separation')
    for ipart, part in enumerate(cfg_data['species']):
        train_tester.class_separation(ipart, part, output_file)
        #test_tester.class_separation(ipart, part, output_file)
    output_file.Close()
   

if __name__ == '__main__':

    from data_preparation import data_preparation

    input_files = ['../../data/0720/its_PIDStudy.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = '../config/config_outputs.yml'
    output_file = '../output/data_preparation.root'

    train_handler, test_handler = data_preparation(input_files, output_file, cfg_data_file, cfg_output_file)
    test_handler = copy.deepcopy(train_handler)

    INPUT_SIZE = len(train_handler.features)
    OUTPUT_SIZE = train_handler.num_particles
    model = FCNN(INPUT_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load('../models/fc_model.pth'))

    test_model(train_handler, test_handler, model, cfg_data_file)