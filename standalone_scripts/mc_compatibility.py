'''
    Check the compatibility of the MC with data for the ITS cluster size
'''

import os
import sys
import yaml
import numpy as np
import polars as pl
from ROOT import (TFile, TDirectory, TH1F, TH2F, TF1, TCanvas, gInterpreter, 
                  TGraph, TMultiGraph, kRed, kGreen, kBlue, kOrange, kCyan)
import logging
from typing import Dict, List, Tuple

# Include BetheBloch C++ file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BETHEBLOCH_DIR = os.path.join(CURRENT_DIR, '..', 'include', 'BetheBloch.hh')
gInterpreter.ProcessLine(f'#include "{BETHEBLOCH_DIR}"')
from ROOT import BetheBloch

# Custom imports
sys.path.append('..')
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from framework.utils.terminal_colors import TerminalColors as tc
from framework.utils.root_setter import obj_setter
from framework.utils.timeit import timeit
from core.dataset import DataHandler
from utils.particles import ParticlePDG

lgging = logging.getLogger(__name__)

class McCompatibility:
    def __init__(self, data_handler: DataHandler, mc_handler: DataHandler, config_output_file:str, output_file:TDirectory):
        
        self.output_file = output_file
        self._load_config(config_output_file)
        self.data_handler = data_handler
        self.mc_handler = mc_handler

    def _load_config(self, config_file: str) -> None:
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def Kolmogorov_Smirnov(self, particle:str) -> TGraph:
        '''
            Run the Kolmogorov-Smirnov test for different bins of a configured variable
        '''

        ks_cfg = self.config['KS']
        x_var = ks_cfg['x_var']
        x_range = np.arange(ks_cfg['x_min'], ks_cfg['x_max'], ks_cfg['x_step'])
        step = ks_cfg['x_step']
        ks_var = ks_cfg['ks_var']
        axis_spec_data = AxisSpec(ks_cfg['nBins'], ks_cfg['min'], ks_cfg['max'], ks_cfg['name']+'_data', ks_cfg['title'])
        axis_spec_data = AxisSpec(ks_cfg['nBins'], ks_cfg['min'], ks_cfg['max'], ks_cfg['name']+'_mc', ks_cfg['title'])

        ks_results = {}

        full_data_ds = self.data_handler.dataset.filter(pl.col('fPartID') == self.config['species'].index(particle) + 1)
        full_mc_ds = self.mc_handler.dataset.filter(np.abs(pl.col('fPartIDMc')) == ParticlePDG[particle])

        for i, val in enumerate(x_range):
            if i == len(x_range)-1:
                continue

            data_ds = full_data_ds.filter((pl.col(x_var) > val) & (pl.col(x_var) < val+step))
            mc_ds = full_mc_ds.filter((pl.col(x_var) > val) & (pl.col(x_var) < val+step))
            
            hist_handler_data = HistHandler.createInstance(data_ds)
            hist_handler_mc = HistHandler.createInstance(mc_ds)
            hist_data = hist_handler_data.buildTH1(ks_var, axis_spec_data)
            hist_mc = hist_handler_mc.buildTH1(ks_var, axis_spec_mc)

            hist_data.Scale(1/hist_data.Integral())
            hist_mc.Scale(1/hist_mc.Integral())

            ks_result = hist_data.KolmogorovTest(hist_mc)
            ks_results[val+step/2.] = ks_result

            del hist_data, hist_mc, hist_handler_data, hist_handler_mc

        ks_graph = TGraph(len(ks_results), np.array(list(ks_results.keys()), dtype=np.float32), np.array(list(ks_results.values()), dtype=np.float32))  
        obj_setter(ks_graph, name=f'KS_{particle}_{ks_var}', title=f'KS test for {particle}; {ks_cfg["xLabel"]}; KS probability', marker_color=kOrange-3, marker_style=20)
        
        return ks_graph

    def run_KS(self) -> None:
        '''
            Run the Kolmogorov-Smirnov test for different bins of the momentum
        '''
        
        ks_dir = self.output_file.mkdir('KS')
        for part in self.config['species']:
            ks_graph = self.Kolmogorov_Smirnov(part)
            ks_dir.cd()
            ks_graph.Write()





if __name__ == '__main__':
    
    # Configure logging
    os.remove("output.log")
    logging.basicConfig(filename="output.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(filename="output.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    input_files_data = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root']
    input_files_mc = ['/home/galucia/ITS_pid/o2/tree_creator/AO2D_MC_LHC24f3.root']
    #input_files = ['/Users/glucia/Projects/ALICE/data/its_pid/LHC22o_pass7_minBias_small.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = 'mc_compatibility.yml'
    output_dir = '../output/MC'
    output_file = output_dir+'/mc_compatibility.root'
    tree_name_data = 'O2clsttable'
    tree_name_mc = 'O2clsttablemcext'
    folder_name = 'DF_*' 

    data_handler = DataHandler(input_files_data, cfg_data_file, tree_name=tree_name_data, folder_name=folder_name, force_option='AO2D')
    mc_handler = DataHandler(input_files_mc, cfg_data_file, tree_name=tree_name_mc, folder_name=folder_name, force_option='AO2D')

    outFile = TFile(output_file, 'RECREATE')
    mc_compatibility = McCompatibility(data_handler, mc_handler, cfg_output_file, outFile)
    mc_compatibility.run_KS()
    
    outFile.Close()