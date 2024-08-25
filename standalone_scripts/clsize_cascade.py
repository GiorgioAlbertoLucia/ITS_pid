'''
    Evaluate Bethe Bloch for cluster size distribution of the cascades vs beta
'''

'''
    Script to perform the Bethe Bloch parametrisation 
'''

import os
import sys
import yaml
import numpy as np
import polars as pl
from ROOT import (TFile, TDirectory, TH1F, TH2F, TH2, TF1, TCanvas, gInterpreter, 
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
from core.bethe_bloch_parametrisation import BetheBlochParametrisation

def merge_hists(h1:TH2F, h2:TH2F):
    h1.Add(h2)
    return h1

def convert_momentum_to_beta(h2: TH2F, mass:float):

    h2_beta = TH2F(h2.GetName()+'_beta', h2.GetTitle(), 100, 0, 1, h2.GetNbinsY(), h2.GetYaxis().GetXmin(), h2.GetYaxis().GetXmax())
    h2_beta.SetName(h2.GetName()+'_beta')
    h2_beta.SetTitle(';#beta;#LT Cluster size #GT #times #LT cos#lambda #GT')

    for ix in range(1, h2.GetNbinsX()+1):
        for iy in range(1, h2.GetNbinsY()+1):
            p = h2.GetXaxis().GetBinCenter(ix)
            beta = p/np.sqrt(p**2 + mass**2)
            h2_beta.Fill(beta, h2.GetYaxis().GetBinCenter(iy), h2.GetBinContent(ix, iy)) 
            h2_beta.SetBinError(h2.GetXaxis().FindBin(beta), iy, np.sqrt(h2.GetBinContent(ix, iy)))

    return h2_beta

def convert_momentum_to_betagamma(h2: TH2F, mass:float):

    h2_betagamma = TH2F(h2.GetName()+'_betagamma', h2.GetTitle(), 100, 0, 10, h2.GetNbinsY(), h2.GetYaxis().GetXmin(), h2.GetYaxis().GetXmax())
    h2_betagamma.SetName(h2.GetName()+'_betagamma')
    h2_betagamma.SetTitle(';#beta#gamma;#LT Cluster size #GT #times #LT cos#lambda #GT')

    for ix in range(1, h2.GetNbinsX()+1):
        for iy in range(1, h2.GetNbinsY()+1):
            p = h2.GetXaxis().GetBinCenter(ix)
            beta = p/np.sqrt(p**2 + mass**2)
            betagamma = p/mass  
            h2_betagamma.Fill(betagamma, h2.GetYaxis().GetBinCenter(iy), h2.GetBinContent(ix, iy))
            h2_betagamma.SetBinError(h2.GetXaxis().FindBin(betagamma), iy, np.sqrt(h2.GetBinContent(ix, iy)))

    return h2_betagamma

def bethe_bloch_beta_fit(bb_param: BetheBlochParametrisation, outFile: TDirectory, h2: TH2F, cfg_output_file: str, particle:str):

    bb_param.load_config(cfg_output_file)
    dir_beta = outFile.mkdir(f'clsize_vs_beta_{particle}')
    bb_param._set_output_dir(dir_beta)
    bb_param.select_fit_particle(particle)
    bb_param.reset_fit_results()
    bb_param.init_config('beta')
    
    bb_param.generate_bethe_bloch_points(h2)
    bb_param.fit_bethe_bloch()

def bethe_bloch_betagamma_fit(bb_param: BetheBlochParametrisation, outFile: TDirectory, h2: TH2F, cfg_output_file: str, particle:str):

    bb_param.load_config(cfg_output_file)
    dir_beta = outFile.mkdir(f'clsize_vs_betagamma_{particle}')
    bb_param._set_output_dir(dir_beta)
    bb_param.select_fit_particle(particle)
    bb_param.reset_fit_results()
    bb_param.init_config('betagamma')
    
    bb_param.generate_bethe_bloch_points(h2)
    bb_param.fit_bethe_bloch()


if __name__ == '__main__':

    # Load the configuration
    #with open('config.yaml', 'r') as file:
    #    cfg = yaml.safe_load(file)

    
    output_file = TFile('../output/cascade.root', 'RECREATE')
    input_file = TFile.Open('/data/galucia/its_pid/clusterSizeCascades.root')

    xi_hs = [input_file.Get(f'xi_pos_avgclustersize_cosL'), input_file.Get(f'xi_neg_avgclustersize_cosL')]
    omega_hs = [input_file.Get(f'omega_pos_avgclustersize_cosL'), input_file.Get(f'omega_neg_avgclustersize_cosL')]

    xi_h = merge_hists(xi_hs[0], xi_hs[1])
    omega_h = merge_hists(omega_hs[0], omega_hs[1])

    xi_h_beta = convert_momentum_to_beta(xi_h, 1.321)
    xi_h_beta.Rebin(4)
    xi_h_betagamma = convert_momentum_to_betagamma(xi_h, 1.321)
    xi_h_betagamma.Rebin(4)
    for ix in range(1, xi_h_beta.GetNbinsX()+1):
        for iy in range(1, xi_h_beta.GetNbinsY()+1):
            xi_h_beta.SetBinError(ix, iy, np.sqrt(xi_h_beta.GetBinContent(ix, iy)))
    for ix in range(1, xi_h_betagamma.GetNbinsX()+1):
        for iy in range(1, xi_h_betagamma.GetNbinsY()+1):
            xi_h_betagamma.SetBinError(ix, iy, np.sqrt(xi_h_betagamma.GetBinContent(ix, iy)))
    
    omega_h_beta = convert_momentum_to_beta(omega_h, 1.672)
    omega_h_beta.Rebin(4)
    omega_h_betagamma = convert_momentum_to_betagamma(omega_h, 1.672)
    omega_h_betagamma.Rebin(4)
    for ix in range(1, omega_h_beta.GetNbinsX()+1):
        for iy in range(1, omega_h_beta.GetNbinsY()+1):
            omega_h_beta.SetBinError(ix, iy, np.sqrt(omega_h_beta.GetBinContent(ix, iy)))
    for ix in range(1, omega_h_betagamma.GetNbinsX()+1):
        for iy in range(1, omega_h_betagamma.GetNbinsY()+1):
            omega_h_betagamma.SetBinError(ix, iy, np.sqrt(omega_h_betagamma.GetBinContent(ix, iy)))

    bb_param = BetheBlochParametrisation(debug=True)

    cfg_output_file_Xi = '/home/galucia/ITS_pid/config/bethe_bloch_parametrisation_Xi.yml'
    cfg_output_file_Omega = '/home/galucia/ITS_pid/config/bethe_bloch_parametrisation_Omega.yml'

    bethe_bloch_beta_fit(bb_param, output_file, xi_h_beta, cfg_output_file_Xi, 'Xi')
    bethe_bloch_beta_fit(bb_param, output_file, omega_h_beta, cfg_output_file_Omega, 'Omega')
    bethe_bloch_betagamma_fit(bb_param, output_file, xi_h_betagamma, cfg_output_file_Xi, 'Xi')
    bethe_bloch_betagamma_fit(bb_param, output_file, omega_h_betagamma, cfg_output_file_Omega, 'Omega')

    output_file.cd()
    xi_h.Write()
    omega_h.Write()
    xi_h_beta.Write()
    omega_h_beta.Write()
    xi_h_betagamma.Write()
    omega_h_betagamma.Write()

    output_file.Close()