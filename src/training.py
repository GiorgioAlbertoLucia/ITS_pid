'''
    Function to train the model
'''

import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import sys
sys.path.append('..')
from core.dataset import DataHandler
from core.fcnn import FCNN
from core.trainer import NeuralNetworkTrainer
from framework.utils.terminal_colors import TerminalColors as tc

def train_model(train_dataset:DataHandler, test_dataset:DataHandler, cfg_data_file:str):
    '''
        Train the model

        Parameters
        ----------
        train_dataset (DataHandler): training dataset
        test_dataset (DataHandler): testing dataset
        cfg (dict): configuration
    '''

    print(tc.BOLD+tc.GREEN+'Training the model'+tc.RESET)
    with open(cfg_data_file, 'r') as file:
        cfg = yaml.safe_load(file)

    INPUT_SIZE = len(train_dataset.features)
    OUTPUT_SIZE = 0
    if cfg['nn_model'] == 'classification':
        OUTPUT_SIZE = train_dataset.num_particles
    elif cfg['nn_model'] == 'regression':
        OUTPUT_SIZE = 1
    else:
        raise ValueError("Invalid model type")
    
    model = FCNN(INPUT_SIZE, OUTPUT_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).to(device)
    loss = None
    if cfg['nn_mode'] == 'classification':
        #loss = nn.CrossEntropyLoss(weight=class_weights)
        loss = nn.CrossEntropyLoss()
    elif cfg['nn_mode'] == 'regression':
        loss = nn.MSELoss()
    else:
        raise ValueError("Invalid model type")
    
    LEARNING_RATE = cfg['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    BATCH_SIZE = cfg['batch_size']
    NN_MODE = cfg['nn_mode']
    NUM_EPOCHS = cfg['num_epochs']
    trainer = NeuralNetworkTrainer(model, train_dataset, test_dataset, loss, optimizer, NN_MODE, BATCH_SIZE)

    train_losses = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        
        train_loss = trainer.train()
        #scheduler.step()
        scheduler.step(train_loss)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")
    
    torch.save(trainer.model.state_dict(), '../models/fc_model.pth')

    # Plot training results
    epochs = [iepoch for iepoch in range(NUM_EPOCHS)]

    plt.figure(figsize=(12, 4))
    plt.plot(epochs, train_losses, label="Train")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig('../output/fc_loss.png')

    trainer.evaluate()

    return model

if __name__ == '__main__':

    from data_preparation import data_preparation

    input_files = ['../../data/0720/its_PIDStudy.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = '../config/config_outputs.yml'
    output_file = '../output/data_preparation.root'

    train_handler, test_handler = data_preparation(input_files, output_file, cfg_data_file, cfg_output_file)
    test_handler = copy.deepcopy(train_handler)

    train_model(train_handler, train_handler, cfg_data_file)