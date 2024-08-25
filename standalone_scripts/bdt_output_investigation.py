'''
    Investigation of the BDT output
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
from core.bethe_bloch_parametrisation import BetheBlochParametrisation
import ROOT

def study_dirty_protons(dataset: pl.DataFrame, output_file: TDirectory):
    '''
        
    '''
    
    ds_protons = dataset.filter(pl.col('fPartID') == 4) # protons have fPartID = 4
    BB_params = {
        'kp1': -0.031712,
        'kp2': -45.0275,
        'kp3': -0.997645,
        'kp4': 1.68228,
        'kp5': 0.0108484
    }

    def BBfunc(x):
        x = np.abs(x)
        beta = x / np.sqrt(1 + x**2)
        aa = beta**BB_params['kp4']
        bb = (1/x)**BB_params['kp5']
        bb = np.log(BB_params['kp3'] + bb)
        return (BB_params['kp2'] - aa - bb) * BB_params['kp1'] / aa
    ds_protons = ds_protons.with_columns(fExpClSizeCosLPr=BBfunc(pl.col('fBetaAbs')))
    sigma_params = {
        'kp0': 0.418451,
        'kp1': -0.040885
    }
    ds_protons = ds_protons.with_columns(fSigmaClSizeCosLPr=(pl.col('fExpClSizeCosLPr')*(sigma_params['kp0'] + sigma_params['kp1'] * pl.col('fExpClSizeCosLPr'))))
    ds_protons = ds_protons.with_columns(fNSigmaPr=((pl.col('fClSizeCosL') - pl.col('fExpClSizeCosLPr')) / pl.col('fSigmaClSizeCosLPr')))
    ds_dirty_protons = ds_protons.filter(pl.col('fNSigmaPr') < -2)

    hist_handler = HistHandler.createInstance(ds_dirty_protons)

    # Save the histograms
    axis_spec_X_pred = AxisSpec(200, -5, 5, 'FakeMatchPr__betaml_vs_p', '; #it{p} (GeV/#it{c}); #beta_{ML}; Counts')
    axis_spec_Y_pred = AxisSpec(200, 0, 1, 'FakeMatchPr__betaml_vs_p', '; #it{p} (GeV/#it{c}); #beta_{ML}; Counts')
    hist_betaml = hist_handler.buildTH2('fP', 'fBetaML', axis_spec_X_pred, axis_spec_Y_pred)

    axis_spec_X_clsize = AxisSpec(200, -5, 5, 'FakeMatchPr__clsize_vs_p', '; #it{p} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos#lambda #GT; Counts')
    axis_spec_Y_clsize = AxisSpec(90, 0, 15, 'FakeMatchPr__clsize_vs_p', '; #it{p} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos#lambda #GT; Counts')
    hist_clsize = hist_handler.buildTH2('fP', 'fClSizeCosL', axis_spec_X_clsize, axis_spec_Y_clsize)

    output_file.cd()
    hist_betaml.Write()
    hist_clsize.Write()

def study_high_cl_pions(dataset: pl.DataFrame, output_file: TDirectory):

    ds_pions = dataset.filter((pl.col('fPartID') == 2) & (pl.col('fClSizeCosL') > 6)) # pions have fPartID = 2
    def rms_cl_size(mean, clsize_layers, n_hit_layers):
        terms = []
        for clsize_layer in clsize_layers:
            term = pl.when(clsize_layer > 0).then((clsize_layer - mean)**2).otherwise(0)
            terms.append(term)
        rms = (np.sum(terms) / n_hit_layers).sqrt()
        return rms

    ds_pions = ds_pions.with_columns(fRMSClSize=(rms_cl_size(pl.col('fMeanItsClSize'), [pl.col(f'fItsClusterSizeL{i}') for i in range(6)], pl.col('fNClustersIts'))))
    hist_handler = HistHandler.createInstance(ds_pions)

    axis_spec_X_pred = AxisSpec(200, -5, 5, 'HighClSizePi__betaml_vs_p', '; #it{p} (GeV/#it{c}); #beta_{ML}; Counts')
    axis_spec_Y_pred = AxisSpec(200, 0, 1, 'HighClSizePi__betaml_vs_p', '; #it{p} (GeV/#it{c}); #beta_{ML}; Counts')
    hist_betaml = hist_handler.buildTH2('fP', 'fBetaML', axis_spec_X_pred, axis_spec_Y_pred)

    axis_spec_X_clsize = AxisSpec(200, -5, 5, 'HighClSizePi__clsize_vs_p', '; #it{p} (GeV/#it{c}); RMS #LT Cluster size #GT #times #LT cos#lambda #GT; Counts')
    axis_spec_Y_clsize = AxisSpec(45, 0, 15, 'HighClSizePi__clsize_vs_p', '; #it{p} (GeV/#it{c}); RMS #LT Cluster size #GT #times #LT cos#lambda #GT; Counts')
    hist_clsize = hist_handler.buildTH2('fP', 'fRMSClSize', axis_spec_X_clsize, axis_spec_Y_clsize)

    axis_spec_X_nclsits = AxisSpec(8, -0.5, 7.5, 'HighClSizePi__nclsits_vs_p', '; n clusters ITS; Counts')
    hist_nclsits = hist_handler.buildTH1('fNClustersIts', axis_spec_X_nclsits)

    ds_pions_low = dataset.filter((pl.col('fPartID') == 2) & (pl.col('fClSizeCosL') < 6)) # pions have fPartID = 2
    hist_handler_low = HistHandler.createInstance(ds_pions_low)

    axis_spec_X_pred_low = AxisSpec(200, -5, 5, 'LowClSizePi__betaml_vs_p', '; #it{p} (GeV/#it{c}); #beta_{ML}; Counts')
    axis_spec_Y_pred_low = AxisSpec(200, 0, 1, 'LowClSizePi__betaml_vs_p', '; #it{p} (GeV/#it{c}); #beta_{ML}; Counts')
    hist_betaml_low = hist_handler_low.buildTH2('fP', 'fBetaML', axis_spec_X_pred_low, axis_spec_Y_pred_low)

    output_file.cd()
    hist_clsize.Write()
    hist_betaml.Write()
    hist_betaml_low.Write()
    hist_nclsits.Write()

def study_K_beta_band(train_dataset, output_file):

    betaband_dir = output_file.mkdir('K_beta_band')

    width = 0.1
    K_mass = 0.493677
    train_dataset = train_dataset.filter((pl.col('fBetaML') - pl.col('fPAbs') / (pl.col('fPAbs')**2 + K_mass**2)**0.5).abs() < width)

    hist_handler = HistHandler.createInstance(train_dataset)
    print(train_dataset['fPartID'].unique())
    for ipart, part in zip(train_dataset['fPartID'].unique(), ['Pi', 'Ka', 'Pr']):
        part_dataset = train_dataset.filter(pl.col('fPartID') == ipart)
        part_hist_handler = HistHandler.createInstance(part_dataset)

        axis_spec_X_pred = AxisSpec(200, -5, 5, f'K_beta_band__betaml_vs_p_{part}', f'; #it{{p}} (GeV/#it{{c}}); #beta_{{ML}}; Counts')
        axis_spec_Y_pred = AxisSpec(200, 0, 1, f'K_beta_band__betaml_vs_p_{part}', f'; #it{{p}} (GeV/#it{{c}}); #beta_{{ML}}; Counts')
        hist_betaml = part_hist_handler.buildTH2('fP', 'fBetaML', axis_spec_X_pred, axis_spec_Y_pred)

        axis_spec_X_clsize = AxisSpec(200, -5, 5, f'K_beta_band__clsize_vs_p_{part}', f'; #it{{p}} (GeV/#it{{c}}); #LT Cluster size #GT #times #LT cos#lambda #GT; Counts')
        axis_spec_Y_clsize = AxisSpec(90, 0, 15, f'K_beta_band__clsize_vs_p_{part}', f'; #it{{p}} (GeV/#it{{c}}); #LT Cluster size #GT #times #LT cos#lambda #GT; Counts')
        hist_clsize = part_hist_handler.buildTH2('fP', 'fClSizeCosL', axis_spec_X_clsize, axis_spec_Y_clsize)

        betaband_dir.cd()
        hist_betaml.Write()
        hist_clsize.Write()

    axis_spec_X_pred = AxisSpec(200, -5, 5, 'K_beta_band__betaml_vs_p', '; #it{p} (GeV/#it{c}); #beta_{ML}; Counts')
    axis_spec_Y_pred = AxisSpec(200, 0, 1, 'K_beta_band__betaml_vs_p', '; #it{p} (GeV/#it{c}); #beta_{ML}; Counts')
    hist_betaml = hist_handler.buildTH2('fP', 'fBetaML', axis_spec_X_pred, axis_spec_Y_pred)

    axis_spec_X_clsize = AxisSpec(200, -5, 5, 'K_beta_band__clsize_vs_p', '; #it{p} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos#lambda #GT; Counts')
    axis_spec_Y_clsize = AxisSpec(90, 0, 15, 'K_beta_band__clsize_vs_p', '; #it{p} (GeV/#it{c}); #LT Cluster size #GT #times #LT cos#lambda #GT; Counts')
    hist_clsize = hist_handler.buildTH2('fP', 'fClSizeCosL', axis_spec_X_clsize, axis_spec_Y_clsize)

    betaband_dir.cd()
    hist_betaml.Write()
    hist_clsize.Write()



if __name__ == '__main__':

    train_file = '/home/galucia/ITS_pid/output/bdt_beta_train.parquet'
    test_file = '/home/galucia/ITS_pid/output/bdt_beta_test.parquet'
    output_file_path = '/home/galucia/ITS_pid/output/bdt_output_investigation.root'

    train_dataset = pl.read_parquet(train_file)
    test_dataset = pl.read_parquet(test_file)
    output_file = TFile(output_file_path, 'RECREATE')

    study_dirty_protons(test_dataset, output_file)
    study_high_cl_pions(test_dataset, output_file)
    study_K_beta_band(train_dataset, output_file)

