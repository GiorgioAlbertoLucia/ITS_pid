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
from framework.src.graph_handler import GraphHandler
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
        axis_spec_mc = AxisSpec(ks_cfg['nBins'], ks_cfg['min'], ks_cfg['max'], ks_cfg['name']+'_mc', ks_cfg['title'])

        ks_results = pl.DataFrame({'x': pl.Series(values=[], dtype=pl.Float64),
                                   'KS': pl.Series(values=[], dtype=pl.Float64),
                                   'KS_critical': pl.Series(values=[], dtype=pl.Float64)})

        full_data_ds = self.data_handler.dataset.filter(pl.col('fPartID') == self.config['species'].index(particle) + 1)
        print(full_data_ds[['fPAbs', 'fClSizeCosL', 'fPartID', 'fMass']].describe())
        full_mc_ds = self.mc_handler.dataset.filter(np.abs(pl.col('fPartIDMc')) == ParticlePDG[particle])
        print(full_mc_ds[['fPAbs', 'fClSizeCosL', 'fPartID', 'fMass']].describe())

        for i, val in enumerate(x_range):
            if i == len(x_range)-1:
                continue

            data_ds = full_data_ds.filter((pl.col(x_var) > val) & (pl.col(x_var) < val+step))
            mc_ds = full_mc_ds.filter((pl.col(x_var) > val) & (pl.col(x_var) < val+step))

            if data_ds.shape[0] == 0 or mc_ds.shape[0] == 0:
                continue
            
            hist_handler_data = HistHandler.createInstance(data_ds)
            hist_handler_mc = HistHandler.createInstance(mc_ds)
            hist_data = hist_handler_data.buildTH1(ks_var, axis_spec_data)
            hist_mc = hist_handler_mc.buildTH1(ks_var, axis_spec_mc)

            #if hist_data.Integral() > 0.:
            #    hist_data.Scale(1/hist_data.Integral())
            #if hist_mc.Integral() > 0.:
            #    hist_mc.Scale(1/hist_mc.Integral())

            ks_result = hist_data.KolmogorovTest(hist_mc, 'm')
            #ks_results[val+step/2.] = ks_result
            ks_critical = 1.358*np.sqrt((hist_data.GetEntries() + hist_mc.GetEntries())/(hist_data.GetEntries()*hist_mc.GetEntries()))
            print(ks_result)
            results = pl.DataFrame({'x': [val+step/2.], 'KS': [ks_result], 'KS_critical': [ks_critical]})
            ks_results = pl.concat([ks_results, results])

            del hist_data, hist_mc, hist_handler_data, hist_handler_mc

        
        graph_handler = GraphHandler(ks_results)
        ks_graph = graph_handler.createTGraph('x', 'KS')
        obj_setter(ks_graph, name=f'KS_{particle}_{ks_var}', title=f'KS test for {particle}; {ks_cfg["xLabel"]}; KS probability', marker_color=kOrange-3, marker_style=20)
        ks_critical = graph_handler.createTGraph('x', 'KS_critical')
        obj_setter(ks_critical, name=f'KS_critical_{particle}_{ks_var}', title=f'KS critical value for {particle}; {ks_cfg["xLabel"]}; KS critical value', marker_color=kCyan-3, marker_style=21)

        mg = TMultiGraph(f'KS_{particle}_{ks_var}', f'KS test for {particle}; {ks_cfg["xLabel"]}; KS distance')
        mg.Add(ks_graph)
        mg.Add(ks_critical)
        canvas = TCanvas(f'KS_{particle}_{ks_var}', f'KS test for {particle}', 800, 600)
        mg.Draw('AP')
        canvas.BuildLegend()

        self.output_file.cd()
        canvas.Write()

        return ks_graph, ks_critical, canvas

    def run_KS(self) -> None:
        '''
            Run the Kolmogorov-Smirnov test for different bins of the momentum
        '''
        
        ks_dir = self.output_file.mkdir('KS')
        #for part in self.config['species']:
        
        ks_graph_Pi, ks_critical_Pi, canvas_Pi = self.Kolmogorov_Smirnov('Pi')
        ks_graph_Pr, ks_critical_Pr, canvas_Pr = self.Kolmogorov_Smirnov('Pr')
        ks_dir.cd()
        #ks_graph_Pi.Write()
        #ks_critical_Pi.Write()
        #canvas_Pi.Write()
        #ks_graph_Pr.Write()
        #ks_critical_Pr.Write()
        #canvas_Pr.Write()

    def compare_distributions(self, particle: str) -> None:
        '''
            Compare the distributions of the ITS cluster size for data and MC
        '''

        out_dir = self.output_file.mkdir(f'clsize_comparison_{particle}')
        ks_cfg = self.config['KS']
        x_var = ks_cfg['x_var']
        x_range = np.arange(ks_cfg['x_min'], ks_cfg['x_max'], ks_cfg['x_step'])
        step = ks_cfg['x_step']
        ks_var = ks_cfg['ks_var']
        axis_spec_data = AxisSpec(ks_cfg['nBins'], ks_cfg['min'], ks_cfg['max'], ks_cfg['name']+'_data', ks_cfg['title'])
        axis_spec_mc = AxisSpec(ks_cfg['nBins'], ks_cfg['min'], ks_cfg['max'], ks_cfg['name']+'_mc', ks_cfg['title'])

        full_data_ds = self.data_handler.dataset.filter(pl.col('fPartID') == self.config['species'].index(particle) + 1)
        print(full_data_ds[['fPAbs', 'fClSizeCosL', 'fPartID', 'fMass']].describe())
        full_mc_ds = self.mc_handler.dataset.filter(np.abs(pl.col('fPartIDMc')) == ParticlePDG[particle])
        print(full_mc_ds[['fPAbs', 'fClSizeCosL', 'fPartID', 'fMass']].describe())

        for i, val in enumerate(x_range):
            if i == len(x_range)-1:
                continue

            data_ds = full_data_ds.filter((pl.col(x_var) > val) & (pl.col(x_var) < val+step))
            mc_ds = full_mc_ds.filter((pl.col(x_var) > val) & (pl.col(x_var) < val+step))

            if data_ds.shape[0] == 0 or mc_ds.shape[0] == 0:
                continue
            
            hist_handler_data = HistHandler.createInstance(data_ds)
            hist_handler_mc = HistHandler.createInstance(mc_ds)
            hist_data = hist_handler_data.buildTH1(ks_var, axis_spec_data)
            hist_data.Scale(1/hist_data.Integral())
            obj_setter(hist_data, name=f'clsize_data_{particle}_{val}', title=f'Data ({particle}) - {val} < {x_var} < {val+step}; {ks_cfg["xLabel"]}; Cluster size', fill_style=3001, fill_color=kBlue)
            hist_mc = hist_handler_mc.buildTH1(ks_var, axis_spec_mc)
            hist_mc.Scale(1/hist_mc.Integral())
            obj_setter(hist_mc, name=f'clsize_mc_{particle}_{val}', title=f'MC ({particle}) - {val} < {x_var} < {val+step}; {ks_cfg["xLabel"]}; Cluster size', fill_style=3001, fill_color=kRed)

            canvas = TCanvas(f'clsize_comparison_{particle}_{val}', f'Cluster size comparison for {particle} in {val} < {x_var} < {val+step}', 800, 600)
            hmax = max(hist_data.GetMaximum(), hist_mc.GetMaximum())
            hframe = canvas.DrawFrame(axis_spec_data.xmin, 0, axis_spec_data.xmax, 1.1*hmax)
            hist_data.Draw('hist same')
            hist_mc.Draw('hist same')
            canvas.BuildLegend()
            out_dir.cd()
            canvas.Write()

            del hist_data, hist_mc, hist_handler_data, hist_handler_mc
    
    def run_comparison(self) -> None:

        for part in self.config['species']:
            self.compare_distributions(part)





if __name__ == '__main__':
    
    # Configure logging
    os.remove("output.log")
    logging.basicConfig(filename="output.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(filename="output.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    input_files_data = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root']
    input_files_mc = ['/data/galucia/its_pid/MC_LHC24f3/MC_LHC24f3_small.root']
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
    mc_compatibility.run_comparison()
    
    outFile.Close()