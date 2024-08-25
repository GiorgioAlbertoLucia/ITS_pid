'''
    Script to perform the Bethe Bloch parametrisation 
'''

import os
import sys
import yaml
import numpy as np
import polars as pl
from ROOT import (TFile, TDirectory, TH1F, TH2F, TF1, TCanvas, gInterpreter, 
                  TGraphErrors, TMultiGraph, kRed, kGreen, kBlue, kOrange, kCyan)
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
from core.bethe_bloch_parametrisation_new import BetheBlochParametrisation
import ROOT


def bethe_bloch_momentum_all(bb_param: BetheBlochParametrisation, outFile: TDirectory):

    dir_p = outFile.mkdir('clsize_vs_p')
    bb_param.run_all('p', dir_p, 'Pr')

def bethe_bloch_beta_all(bb_param: BetheBlochParametrisation, outFile: TDirectory):

    dir_beta = outFile.mkdir('clsize_vs_beta')
    bb_param.run_all('beta', dir_beta, 'Pr')

def bethe_bloch_momentum_fit(data_handler: DataHandler, bb_param: BetheBlochParametrisation, outFile: TDirectory, particle: str, cfg_output_file: str):

    bb_param.load_config(cfg_output_file)
    dir_betagamma = outFile.mkdir(f'clsize_vs_p_{particle}')
    bb_param._set_output_dir(dir_betagamma)
    bb_param.select_fit_particle(particle)
    bb_param.reset_fit_results()
    bb_param.init_config('p')
        
    ds_p = data_handler.dataset.filter(pl.col('fPartID') == data_handler.part_list.index(particle))
    bb_param.upload_h2(ds_p)
    bb_param.generate_bethe_bloch_points()
    bb_param.save_fit_results('../output/LHC22o_pass7_minBias_small/fit_results_Pr.csv')
    bb_param.fit_bethe_bloch()

def bethe_bloch_beta_fit(bb_param: BetheBlochParametrisation, outFile: TDirectory, particle: str, cfg_output_file: str):

    bb_param.load_config(cfg_output_file)
    dir_beta = outFile.mkdir(f'clsize_vs_beta_{particle}')
    bb_param._set_output_dir(dir_beta)
    bb_param.select_fit_particle(particle)
    bb_param.reset_fit_results()
    bb_param.init_config('beta')
        
    bb_param.select_fit_particle(particle)
    bb_param.generate_bethe_bloch_points()
    bb_param.fit_bethe_bloch()

def bethe_bloch_betagamma_fit(data_handler: DataHandler, bb_param: BetheBlochParametrisation, outFile: TDirectory, particle: str, cfg_output_file: str):

    bb_param.load_config(cfg_output_file)
    dir_betagamma = outFile.mkdir(f'clsize_vs_betagamma_{particle}')
    bb_param._set_output_dir(dir_betagamma)
    bb_param.select_fit_particle(particle)
    bb_param.reset_fit_results()
    bb_param.init_config('betagamma')
        
    #print('particles:', data_handler.part_list)
    #for p in data_handler.part_list:
    #    ds_p = data_handler.dataset.filter(pl.col('fPartID') == data_handler.part_list.index(p))
    #    #print('particle:', data_handler.part_list.index(particle))
    #    print('particles:', ds_p['fPartID'].unique())
    #    th2 = TH2F(f'test_{p}', 'Bethe Bloch Fit; #beta; #LT Cluster size #GT #times #LT cos#lambda #GT; Counts', 200, 0.3, 5.0, 90, 0, 15)
    #    for x, y in zip(ds_p['fBetaGamma'], ds_p['fClSizeCosL']):
    #        th2.Fill(x, y)
    #    dir_betagamma.cd()
    #    th2.Write()
    ds_p = data_handler.dataset.filter(pl.col('fPartID') == data_handler.part_list.index(particle))
    bb_param.upload_h2(ds_p)
    bb_param.generate_bethe_bloch_points()
    bb_param.save_fit_results('../output/LHC22o_pass7_minBias_small/fit_results_Pr.csv')
    bb_param.fit_bethe_bloch()

@timeit
def draw_nsigma_momentum(data_handler: DataHandler, bb_param: BetheBlochParametrisation, outFile: TDirectory, cfg_output_file: str, particle:str):

    bb_param.load_config(cfg_output_file)
    dir_comb = outFile.mkdir(f'clsize_vs_p_nsigma_{particle}')
    bb_param._set_output_dir(dir_comb)
    bb_param.init_config('p')
    BB_params = {
        'kp1': -0.031712,
        'kp2': -45.0275,
        'kp3': -0.997645,
        'kp4': 1.68228,
        'kp5': 0.0108484
    }
    bb_param.upload_bethe_bloch_params(BB_params)
    bb_param.select_fit_particle(particle)
    data_handler.dataset = bb_param.add_nsigma_column(data_handler.dataset, convert_to_beta=True)
    for part in bb_param.config['species']:
        ds_part = data_handler.dataset.filter(pl.col('fPartID') == data_handler.part_list.index(part))
        bb_param.draw_nsigma_distribution(ds_part, part)
    #bb_param.purity_efficiency(2, other_particle='Ka')
    #bb_param.purity_efficiency(2, other_particle='Pi')

    #bb_param.data_handler.dataset = bb_param.data_handler.dataset.filter(pl.col(f'fNSigma{particle}').abs() < 2)
    #bb_param.generate_h2()  
    #dir_comb.cd()
    #bb_param.h2.Write('h2_Pr_nsigma2')

def draw_bb_fit_beta(bb_param: BetheBlochParametrisation, outFile: TDirectory, cfg: Dict[str, any]):

    dir_comb = outFile.mkdir('clsize_vs_beta_betaParam')
    bb_param._set_output_dir(dir_comb)
    bb_param.init_config('beta')
    BB_params = {
        'kp1': -0.031712,
        'kp2': -45.0275,
        'kp3': -0.997645,
        'kp4': 1.68228,
        'kp5': 0.0108484
    }
    bb_param.select_fit_particle('Pr')
    bb_param.upload_bethe_bloch_params(BB_params)
    bb_param.generate_h2()
    bb_param.draw_bethe_bloch_fit('Pr', cfg, xmin_fit=0.3, xmax_fit=1.0)


def plot_th2_with_tf1(file_path, th2_name, canvas_name, output_pdf):
    # Open the file
    file = ROOT.TFile(file_path, "READ")

    # Retrieve the TH2 and TF1 from the file
    th2 = file.Get(th2_name)
    tf1 = ROOT.TF1('bethe_bloch', BetheBloch, 0.3, 1.0, 5)
    BB_params = {
        'kp1': -1.86059e-02,
        'kp2': -7.50438e+01,
        'kp3': -9.86288e-01,
        'kp4': 1.52671e+00,
        'kp5': 1.12361e+00,
    }
    for i, (key, value) in enumerate(BB_params.items()):
        tf1.SetParameter(i, value)
       # Create a canvas
    canvas = ROOT.TCanvas(canvas_name, canvas_name)
    canvas.cd()

    print(type(th2))
    # Draw the TH2
    ROOT.gStyle.SetOptStat(0)
    canvas.SetRightMargin(0.15)
    
    hframe = canvas.DrawFrame(0.2, 0.5, 1.0, 12.0, '#LT Cluster size #GT #times #LT cos#lambda #GT vs #beta; #beta; #LT Cluster size #GT #times #LT cos#lambda #GT; Counts')
    th2.Draw("colz same")
    tf1.Draw("SAME")
    canvas.SetLogz()
    


    # Save the canvas to a PDF file
    canvas.SaveAs(output_pdf)

    # Close the file
    #file.Close()


if __name__ == '__main__':

    # Configure logging
    os.remove("/home/galucia/ITS_pid/output/output.log")
    logging.basicConfig(filename="/home/galucia/ITS_pid/output/output.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(filename="/home/galucia/ITS_pid/output/output.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    #input_files = ['/data/galucia/its_pid/LHC23_pass4_skimmed/LHC23_pass4_skimmed.root']
    input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root']
    #input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small_old2.root']
    #input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_slice_pkpi.root']
    #input_files = ['/data/galucia/its_pid/MC_LHC24f3/MC_LHC24f3_small.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file_He = '/home/galucia/ITS_pid/config/bethe_bloch_parametrisation_He.yml'
    cfg_output_file_De = '/home/galucia/ITS_pid/config/bethe_bloch_parametrisation_De.yml'
    cfg_output_file_Pr = '/home/galucia/ITS_pid/config/bethe_bloch_parametrisation_Pr.yml'
    #cfg_output_file_Pr = '/home/galucia/ITS_pid/config/bethe_bloch_parametrisation_Pr_mc.yml'
    cfg_output_file_Ka = '/home/galucia/ITS_pid/config/bethe_bloch_parametrisation_Ka.yml'
    cfg_output_file_Pi = '/home/galucia/ITS_pid/config/bethe_bloch_parametrisation_Pi.yml'
    #cfg_output_file_Ka = '/home/galucia/ITS_pid/config/bethe_bloch_parametrisation_Pr_mc.yml'
    #output_dir = '../output/LHC23_pass4_skimmed'
    output_dir = '../output/LHC22o_pass7_minBias_small'
    #output_dir = '../output/MC'
    #output_file = output_dir+'/bethe_bloch_parametrisation.root'
    #output_file = output_dir+'/bethe_bloch_parametrisation_olddata2_fm.root'
    #output_file = output_dir+'/bethe_bloch_parametrisation_Pr.root'
    output_file = output_dir+'/bethe_bloch_parametrisation_nsigma.root'
    #output_file = output_dir+'/bethe_bloch_parametrisation_pkpi.root'
    #output_file = output_dir+'/bethe_bloch_parametrisation_small_phi.root'
    #output_file = output_dir+'/bethe_bloch_parametrisation_very_small.root'
    tree_name = 'O2clsttable'
    #tree_name = 'O2clsttableextra'
    #tree_name = 'O2clsttablemcext'
    folder_name = 'DF_*' 

    print(tc.GREEN+tc.BOLD+'Data uploading'+tc.RESET)
    data_handler = DataHandler(input_files, cfg_data_file, tree_name=tree_name, folder_name=folder_name, force_option='AO2D', rigidity_he=False)
    #data_handler.correct_for_pid_in_trk()
    #data_handler.dataset = data_handler.dataset.filter(np.abs(pl.col('fEta')) < 0.5)
    #data_handler.dataset = data_handler.dataset.filter(np.abs(pl.col('fPhi')) < 0.5)

    outFile = TFile(output_file, 'RECREATE')
    print('Output file created: '+tc.UNDERLINE+tc.BLUE+output_file+tc.RESET)
    print(tc.BOLD+tc.GREEN+'Bethe Bloch parametrisation...'+tc.RESET)
    bb_param = BetheBlochParametrisation()

    #bethe_bloch_momentum_all(bb_param, outFile)
    #bethe_bloch_beta_all(bb_param, outFile)
    
    #bethe_bloch_momentum_fit(data_handler, bb_param, outFile, 'He', cfg_output_file_He)
    #bethe_bloch_momentum_fit(data_handler, bb_param, outFile, 'De', cfg_output_file_De)
    #bethe_bloch_momentum_fit(data_handler, bb_param, outFile, 'Pr', cfg_output_file_Pr)
    #bethe_bloch_momentum_fit(data_handler, bb_param, outFile, 'Ka', cfg_output_file_Ka)
    #bethe_bloch_momentum_fit(data_handler, bb_param, outFile, 'Pi', cfg_output_file_Pi)
    
    #bethe_bloch_beta_fit(bb_param, outFile, 'He', cfg_output_file_He)
    #bethe_bloch_beta_fit(bb_param, outFile, 'De', cfg_output_file_De)
    #bethe_bloch_beta_fit(bb_param, outFile, 'Pr', cfg_output_file_Pr)
    #bethe_bloch_beta_fit(bb_param, outFile, 'Ka', cfg_output_file_Ka)
    #bethe_bloch_beta_fit(bb_param, outFile, 'Pi', cfg_output_file_Pi)
    
    #bethe_bloch_betagamma_fit(data_handler, bb_param, outFile, 'He', cfg_output_file_He)
    #bethe_bloch_betagamma_fit(data_handler, bb_param, outFile, 'De', cfg_output_file_De)
    #bethe_bloch_betagamma_fit(data_handler, bb_param, outFile, 'Pr', cfg_output_file_Pr)
    #bethe_bloch_betagamma_fit(data_handler, bb_param, outFile, 'Ka', cfg_output_file_Ka)
    #bethe_bloch_betagamma_fit(data_handler, bb_param, outFile, 'Pi', cfg_output_file_Pi)
    
    draw_nsigma_momentum(data_handler, bb_param, outFile, cfg_output_file_Pr, 'Pr')
    draw_nsigma_momentum(data_handler, bb_param, outFile, cfg_output_file_Ka, 'Ka')
    draw_nsigma_momentum(data_handler, bb_param, outFile, cfg_output_file_Pi, 'Pi')
    
    #with open(cfg_output_file, 'r') as file:
    #    cfg = yaml.safe_load(file)['beta']
    #draw_bb_fit_beta(bb_param, outFile, cfg)

    #plot_th2_with_tf1(output_file, 'clsize_vs_beta/h2_BB_Pr', 'Bethe Bloch Fit', 'bethe_bloch_fit.pdf')
    
    #outFile.Close()