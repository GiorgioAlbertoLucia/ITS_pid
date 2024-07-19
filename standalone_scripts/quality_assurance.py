'''
    Scripts to produce quality assurance plots for the ITS PID analysis.
'''

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


def cl_size_vs_eta(inFiles:list, cfgFileData:str, cfgFileOutput:str, outFile:TFile, **kwargs):

    with open(cfgFileData, 'r') as file:
        cfg_data = yaml.safe_load(file)
    with open(cfgFileOutput, 'r') as file:
        cfg_output = yaml.safe_load(file)

    print(tc.BOLD+tc.GREEN+'Data preparation'+tc.RESET)
    data_handler = DataHandler(inFiles, cfgFileData, **kwargs)
    data_handler.dataset = data_handler.dataset.filter((pl.col('fPAbs') > 1.) & (pl.col('fPAbs') < 1.3))

    hist_handler = {'all': HistHandler.createInstance(data_handler.dataset)}
    for ipart, part in enumerate(cfg_data['species']):
        ds = data_handler.dataset.filter(pl.col('fPartID') == ipart+1)
        hist_handler[part] = HistHandler.createInstance(ds)

    cfg_output = cfg_output['cl_size_vs_eta']

    print(tc.BOLD+tc.GREEN+'Cluster size vs eta'+tc.RESET)

    outDir = outFile.mkdir('cl_size_vs_eta')
    outDir.cd()
    
    for key, cfg in tqdm(cfg_output.items()):
            
        if key == 'outDirs':                continue
        
        for part in cfg['particle']:

            if 'TH1' in cfg['type']:
                
                title = cfg['title'].split(';')[0] + f' ({part})' 
                for term in cfg['title'].split(';')[1:]:
                    title += ';' + term
                axisSpecX = AxisSpec(cfg['nXBins'], cfg['xMin'], cfg['xMax'], cfg['name']+f'_{part}', title)
                hist = hist_handler[part].buildTH1(cfg['xVariable'], axisSpecX)

                hist.Write()

            if 'TH2' in cfg['type']:
                
                title = cfg['title'].split(';')[0] + f' ({part})' 
                for term in cfg['title'].split(';')[1:]:
                    title += ';' + term
                axisSpecX = AxisSpec(cfg['nXBins'], cfg['xMin'], cfg['xMax'], cfg['name']+f'_{part}', title)
                axisSpecY = AxisSpec(cfg['nYBins'], cfg['yMin'], cfg['yMax'], cfg['name']+f'_{part}', title)
                hist = hist_handler[part].buildTH2(cfg['xVariable'], cfg['yVariable'], axisSpecX, axisSpecY)

                hist.Write()

if __name__ == '__main__':

    #input_files = ['../../data/0720/its_PIDStudy.root']
    input_files = ['/home/galucia/ITS_pid/o2/tree_creator/AO2D.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = 'quality_assurance.yml'
    output_file = '../output/quality_assurance.root'
    tree_name = 'O2clsttableextra'
    folder_name = 'DF_*' 

    outFile = TFile(output_file, 'RECREATE')
    cl_size_vs_eta(input_files, cfg_data_file, cfg_output_file, outFile, tree_name=tree_name, folder_name=folder_name)

    outFile.Close()

