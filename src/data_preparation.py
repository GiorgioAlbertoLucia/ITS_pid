'''
    Function to prepare the data for the model
'''

import copy
import numpy as np
import polars as pl
import yaml
from tqdm import tqdm
from ROOT import TFile, TDirectory
from typing import List

import sys
sys.path.append('..')
sys.path.append('../..')
from core.dataset import DataHandler
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from framework.utils.terminal_colors import TerminalColors as tc
from framework.utils.timeit import timeit
from utils.particles import ParticlePDG

def print_debug_info(data_handler:DataHandler, debug_messages:List[str]):
    for msg in debug_messages:
        print(tc.MAGENTA+'[DEBUG]: '+tc.RESET+msg)
    print(tc.MAGENTA+'[DEBUG]: '+tc.RESET+'particles in data_handler:', data_handler.dataset['fPartID'].unique())
    print(tc.MAGENTA+'[DEBUG]: '+tc.RESET+'particles in data_handler:', data_handler.id_part_map)

def data_visualization(train_handler:DataHandler, cfg_output:dict, cfg_data:dict, output_file:TDirectory, **kwargs):

    # data visualization    
    print(tc.BOLD+tc.GREEN+'Data visualization'+tc.RESET)
    
    is_mc = kwargs.get('is_mc', False)
    out_dirs = {dir: output_file.mkdir(dir) for dir in cfg_output['outDirs']}
    print(type(train_handler.dataset))
    print(train_handler.dataset['fPartID'].unique())
    print(train_handler.id_part_map)
    hist_handler = {'all': HistHandler.createInstance(train_handler.dataset)}
    for ipart, part in train_handler.id_part_map.items():    
        ds = None
        if is_mc:   ds = train_handler.dataset.filter(np.abs(pl.col('fPartIDMc')) == ParticlePDG[part])
        else:       ds = train_handler.dataset.filter(pl.col('fPartID') == ipart)
        hist_handler[part] = HistHandler.createInstance(ds)
    
    
    for key, cfg in tqdm(cfg_output.items()):
            
        if key == 'outDirs':                continue
        
        for part in cfg['particle']:

            if part not in hist_handler.keys():
                continue

            if 'TH1' in cfg['type']:
                
                if cfg['xVariable'] not in train_handler.dataset.columns:
                    print(tc.UNDERLINE+tc.RED+'ERROR:'+tc.RESET,cfg['xVariable'],'not present in dataset!')
                    continue

                axisSpecX = AxisSpec(cfg['nXBins'], cfg['xMin'], cfg['xMax'], cfg['name'], cfg['title'])
                hist = hist_handler[part].buildTH1(cfg['xVariable'], axisSpecX)

                out_dirs[part].cd()
                hist.Write()

            if 'TH2' in cfg['type']:
                
                if cfg['xVariable'] not in train_handler.dataset.columns:
                        print(tc.UNDERLINE+tc.RED+'ERROR:'+tc.RESET,cfg['xVariable'],'not present in dataset!')
                        continue
                elif cfg['yVariable'] not in train_handler.dataset.columns:
                    print(tc.UNDERLINE+tc.RED+'ERROR:'+tc.RESET,cfg['yVariable'],'not present in dataset!')
                    continue

                axisSpecX = AxisSpec(cfg['nXBins'], cfg['xMin'], cfg['xMax'], cfg['name'], cfg['title'])
                axisSpecY = AxisSpec(cfg['nYBins'], cfg['yMin'], cfg['yMax'], cfg['name'], cfg['title'])
                hist = hist_handler[part].buildTH2(cfg['xVariable'], cfg['yVariable'], axisSpecX, axisSpecY)

                out_dirs[part].cd()
                hist.Write()
    

@timeit
def data_preparation(input_files:list, output_file:TDirectory, cfg_data_file:str, cfg_output_file:str, input_files_he=None, **kwargs):
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

    if input_files_he is not None:
        data_handler_he = DataHandler(input_files_he, cfg_data_file, rigidity_he=False, **kwargs)
        data_handler_he.correct_for_pid_in_trk()

        data_handler.drop_particle('He')
        print('DEBUG:', data_handler.id_part_map)
        print('particles in data_handler:', data_handler.dataset['fPartID'].unique())
        print('particles in data_handler_he:', data_handler_he.dataset['fPartID'].unique())

        data_handler.merge_datasets(data_handler_he)
        print('DEBUG:', data_handler.id_part_map)
        print('particles in data_handler:', data_handler.dataset['fPartID'].unique())
    
    data_handler.drop_particle('Unidentified')
    data_handler.drop_particle('El')

    species_selection_opt = kwargs.get('species_selection', [False, 'list of species'])
    if species_selection_opt[0]:
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Selecting species:', species_selection_opt[1])
        data_handler.select_species(species_selection_opt[1])
        if kwargs.get('debug', False):
            print('species selection')
            
    if kwargs.get('rename_classes', False):
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Renaming classes')
        data_handler.rename_classes()
        if kwargs.get('debug', False):  print_debug_info(data_handler, ['rename classes'])

    if kwargs.get('split', False):
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Splitting the dataset')
        train_handler, test_handler = data_handler.train_test_split(cfg_data['test_size'])
        if kwargs.get('debug', False):  print_debug_info(train_handler, ['split'])
    elif kwargs.get('test_path', None):
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Loading test dataset')
        test_dict = kwargs
        test_dict['tree_name'] = 'O2clsttableextra'
        test_handler = DataHandler(kwargs['test_path'], cfg_data_file, **kwargs)
        if species_selection_opt[0]:
            test_handler.select_species(species_selection_opt[1])
        if kwargs.get('rename_classes', False):
            test_handler.rename_classes()
        train_handler = data_handler
    else:
        train_handler = data_handler
        test_handler = None

    if kwargs.get('minimum_hits', None):
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Minimum hits:', kwargs['minimum_hits'])
        train_handler.dataset = train_handler.dataset.filter(pl.col('fNClustersIts') >= kwargs['minimum_hits'])
        if test_handler:    test_handler.dataset = test_handler.dataset.filter(pl.col('fNClustersIts') >= kwargs['minimum_hits'])
        if kwargs.get('debug', False):  print_debug_info(train_handler, ['minimum hits'])
    
    variable_selection_opt = kwargs.get('variable_selection', [False, 'var', 'min:float', 'max:float'])
    if variable_selection_opt[0]:
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Variable selection:', variable_selection_opt[1], variable_selection_opt[2], variable_selection_opt[3])
        train_handler.dataset = train_handler.dataset.filter(pl.col(variable_selection_opt[1]).is_between(variable_selection_opt[2], variable_selection_opt[3]))
        if test_handler:    test_handler.dataset = test_handler.dataset.filter(pl.col(variable_selection_opt[1]).is_between(variable_selection_opt[2], variable_selection_opt[3]))
        if kwargs.get('debug', False):  print_debug_info(train_handler, ['variable selection'])

    if kwargs.get('clean_protons', False):
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Cleaning protons')
        train_handler.clean_protons()
        if kwargs.get('debug', False):  print_debug_info(train_handler, ['clean protons'])

    data_augmentation_opt = kwargs.get('data_augmentation', [False, 'particles:List[str]', 'min:float', 'max:float', 'n_samples:int'])
    if data_augmentation_opt[0]:
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Data augmentation:', data_augmentation_opt[1], data_augmentation_opt[2], data_augmentation_opt[3], data_augmentation_opt[4])
        for part in data_augmentation_opt[1]:
            train_handler.data_augmentation(part, data_augmentation_opt[2], data_augmentation_opt[3], data_augmentation_opt[4])
        if kwargs.get('debug', False):  print_debug_info(train_handler, ['data augmentation'])
    
    variable_oversample_opt = kwargs.get('oversample_momentum', [False, 'var', 'int:nsamples', 'min:float', 'max:float'])
    if variable_oversample_opt[0]:
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Oversample momentum:', variable_oversample_opt[1], variable_oversample_opt[2], variable_oversample_opt[3], variable_oversample_opt[4])
        train_handler.variable_oversample(variable_oversample_opt[1], variable_oversample_opt[2], variable_oversample_opt[3], variable_oversample_opt[4])
        if kwargs.get('debug', False):  print_debug_info(train_handler, ['oversample momentum'])
    
    if kwargs.get('oversample', False):
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Oversampling')
        train_handler.class_oversample()
        if kwargs.get('debug', False):  print_debug_info(train_handler, ['oversample'])
    
    n_samples_opt = kwargs.get('n_samples', None)
    if n_samples_opt:   
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Reducing dataset to:', n_samples_opt)
        train_handler.reduced_dataset(n_samples_opt)
        if kwargs.get('debug', False):  print_debug_info(train_handler, ['n samples'])
    
    flatten_samples_opt = kwargs.get('flatten_samples', ['fPAbs', 250, 0., 2.5, None])
    if flatten_samples_opt[4]:
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Flatten samples:', flatten_samples_opt[0], flatten_samples_opt[1], flatten_samples_opt[2], flatten_samples_opt[3], flatten_samples_opt[4])
        train_handler.variable_and_class_flattening(flatten_samples_opt[0], flatten_samples_opt[1], flatten_samples_opt[2], flatten_samples_opt[3], flatten_samples_opt[4])
        if kwargs.get('debug', False):  print_debug_info(train_handler, ['flatten samples'])
    
    #data_augmentation_opt = kwargs.get('data_augmentation', [False, 'particles:List[str]', 'min:float', 'max:float', 'n_samples:int'])
    #if data_augmentation_opt[0]:
    #    for part in data_augmentation_opt[1]:
    #        train_handler.data_augmentation(part, data_augmentation_opt[2], data_augmentation_opt[3], data_augmentation_opt[4])
    #    if kwargs.get('debug', False):
    #        print('data augmentation')
    #        print('particles in train_handler:', train_handler.dataset['fPartID'].unique())
    #        print('particles in train_handler:', train_handler.id_part_map)
    
    #train_handler.enhance_class('Pi', 2)

    #train_handler, validation_handler = train_handler.train_test_split(cfg_data['validation_size'])
    if kwargs.get('validation', True):
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Produce validation set')
        test_handler, validation_handler = test_handler.train_test_split(cfg_data['validation_size'])
    else:
        validation_handler = None
    if kwargs.get('debug', False):
            print('validation')
            print('particles in train_handler:', train_handler.dataset['fPartID'].unique())
            print('masses in train_handler:', train_handler.dataset['fMass'].unique())
            print('NORM masses in train_handler:', train_handler.normalized_dataset['fMass'].unique())
    
    if kwargs.get('normalize', False):  
        print(tc.GREEN+'[INFO]: '+tc.RESET+'Feature scaling') 
        means, stds = train_handler.auto_normalize()
        validation_handler.normalize(means, stds)
        test_handler.normalize(means, stds)
        if kwargs.get('debug', False):
            print('normalize')
            print('particles in train_handler:', train_handler.dataset['fPartID'].unique())
            print('masses in train_handler:', train_handler.dataset['fMass'].unique())
            print('NORM masses in train_handler:', train_handler.normalized_dataset['fMass'].unique())
            print('particles in train_handler:', train_handler.normalized_dataset['fPartID'].unique())
            print('particles in train_handler:', train_handler.id_part_map)
    
    if cfg_data['visualize']:
        data_visualization(train_handler, cfg_output, cfg_data, output_file, **kwargs)

    return train_handler, validation_handler, test_handler


if __name__ == '__main__':

    #input_files = ['../../data/0720/its_PIDStudy.root']
    #input_files = ['/home/galucia/ITS_pid/o2/tree_creator/AO2D.root']
    #input_files = ['/data/galucia/its_pid/MC_LHC24f3/MC_LHC24f3_small.root']
    #input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_slice_pkpi.root']
    input_files_he = ['/data/galucia/its_pid/LHC23_pass4_skimmed/LHC23_pass4_skimmed.root']
    #input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small_pkpi.root']
    input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root',
                   '/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_longK.root']
    #input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small_old2.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = '../config/config_outputs.yml'
    #output_dir = '../output/MC'
    output_dir = '../output/LHC22o_pass7_minBias_small'
    #output_dir = '../output/LHC23_pass4_skimmed'
    #output_file = output_dir+'/data_preparation_olddata2.root'
    #output_file = output_dir+'/data_preparation.root'
    output_file = output_dir+'/data_preparation_minhits7.root'
    #tree_name = 'O2clsttablemcext'
    tree_name = 'O2clsttable'
    #tree_name = 'O2clsttableextra'
    folder_name = 'DF_*'

    output_file_root = TFile(output_file, 'RECREATE')

    output_file_root = TFile(output_file, 'RECREATE')
    #data_preparation(input_files, output_file_root, cfg_data_file, cfg_output_file, tree_name=tree_name, folder_name=folder_name, force_option='AO2D', is_mc=False, validation=False)
    data_preparation(input_files, output_file_root, cfg_data_file, cfg_output_file, input_files_he=input_files_he, 
                                                            folder_name=folder_name,
                                                            tree_name=tree_name,
                                                            force_option='AO2D', 
                                                            validation=False,
                                                            rename_classes=True,
                                                            minimum_hits=7)
    output_file_root.Close()

