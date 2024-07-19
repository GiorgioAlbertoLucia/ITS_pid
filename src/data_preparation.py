'''
    Function to prepare the data for the model
'''

import copy
import polars as pl
import yaml
from tqdm import tqdm
from ROOT import TFile, TDirectory

import sys
sys.path.append('..')
sys.path.append('../..')
from core.dataset import DataHandler
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from framework.utils.terminal_colors import TerminalColors as tc
from framework.utils.timeit import timeit

def data_visualization(train_handler:DataHandler, cfg_output:dict, cfg_data:dict, output_file:str):

    # data visualization    
    print(tc.BOLD+tc.GREEN+'Data visualization'+tc.RESET)
    print('Output file: '+tc.UNDERLINE+tc.BLUE+output_file+tc.RESET)
    output_file = TFile(output_file, 'RECREATE')
    
    out_dirs = {dir: output_file.mkdir(dir) for dir in cfg_output['outDirs']}
    print(type(train_handler.dataset))
    hist_handler = {'all': HistHandler.createInstance(train_handler.dataset)}
    for ipart, part in enumerate(cfg_data['species']):    
        ds = train_handler.dataset.filter(pl.col('fPartID') == ipart+1)
        hist_handler[part] = HistHandler.createInstance(ds)
    
    
    for key, cfg in tqdm(cfg_output.items()):
            
        if key == 'outDirs':                continue
        
        for part in cfg['particle']:

            if 'TH1' in cfg['type']:

                axisSpecX = AxisSpec(cfg['nXBins'], cfg['xMin'], cfg['xMax'], cfg['name'], cfg['title'])
                hist = hist_handler[part].buildTH1(cfg['xVariable'], axisSpecX)

                out_dirs[part].cd()
                hist.Write()

            if 'TH2' in cfg['type']:

                axisSpecX = AxisSpec(cfg['nXBins'], cfg['xMin'], cfg['xMax'], cfg['name'], cfg['title'])
                axisSpecY = AxisSpec(cfg['nYBins'], cfg['yMin'], cfg['yMax'], cfg['name'], cfg['title'])
                hist = hist_handler[part].buildTH2(cfg['xVariable'], cfg['yVariable'], axisSpecX, axisSpecY)

                out_dirs[part].cd()
                hist.Write()

    

    output_file.Close()
    

@timeit
def data_preparation(input_files:list, output_file:str, cfg_data_file:str, cfg_output_file:str, **kwargs):
    '''
        Prepare the data for the model

        Parameters
        ----------
        input_files (list): list of input files
        output_file (str): output file
        cfg_data_file (str): configuration file for the data
        cfg_output_file (str): configuration file for the output
    '''

    with open(cfg_output_file, 'r') as file:
        cfg_output = yaml.safe_load(file)
    with open(cfg_data_file, 'r') as file:
        cfg_data = yaml.safe_load(file)

    print(tc.BOLD+tc.GREEN+'Data preparation'+tc.RESET)
    data_handler = DataHandler(input_files, cfg_data_file, **kwargs)
    #data_handler.apply_quality_selection_cuts()
    #data_handler.eliminate_nan()
    #if cfg_data['oversample']:  data_handler.oversample()
    #data_handler.normalize()

    # train_handler, test_handler = data_handler.train_test_split(cfg_data['test_size'])
    train_handler = data_handler
    test_handler = None

    if cfg_data['visualize']:
        data_visualization(train_handler, cfg_output, cfg_data, output_file)

    return train_handler, test_handler

if __name__ == '__main__':

    #input_files = ['../../data/0720/its_PIDStudy.root']
    input_files = ['/home/galucia/ITS_pid/o2/tree_creator/AO2D.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = '../config/config_outputs.yml'
    output_file = '../output/data_preparation.root'
    tree_name = 'O2clsttableextra'
    folder_name = 'DF_*'

    data_preparation(input_files, output_file, cfg_data_file, cfg_output_file, tree_name=tree_name, folder_name=folder_name)

