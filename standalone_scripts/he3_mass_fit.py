'''
    Script to determin a Bethe Bloch-like parametrisation for the cluster size distribution
'''

import os
import sys
import yaml
import ctypes
import numpy as np
import polars as pl
from ROOT import (TFile, TDirectory, TH1F, TH2F, TF1, TCanvas, gInterpreter, TLegend, TText, gStyle,
                  TGraphErrors, TMultiGraph, kRed, kGreen, kBlue, kOrange, kCyan)
import logging
from typing import Dict, List, Tuple

# Include BetheBloch C++ file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BETHEBLOCH_DIR = os.path.join(CURRENT_DIR, '..', 'include', 'BetheBloch.hh')
gInterpreter.ProcessLine(f'#include "{BETHEBLOCH_DIR}"')
EMG_DIR = os.path.join(CURRENT_DIR, '..', 'include', 'ExponentiallyModifiedGaussian.hh')
gInterpreter.ProcessLine(f'#include "{EMG_DIR}"')
from ROOT import BetheBloch, ExponentiallyModifiedGaussian

# Custom imports
sys.path.append('..')
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from framework.src.graph_handler import GraphHandler
from framework.utils.terminal_colors import TerminalColors as tc
from framework.utils.root_setter import obj_setter
from framework.utils.timeit import timeit
from core.dataset import DataHandler

#def fit_he3_mass(data_handler: DataHandler, output_path: str):
def fit_he3_mass(hist: TH1F, output_path: str): 

    gStyle.SetOptStat(0)
    mass_fit = TF1('mass_fit', 'gaus', 2.65, 3.)
    mass_fit.SetParameters(1000, 2.84, 0.1)
    hist.Fit(mass_fit, 'RMSL+')

    # Create the canvas
    canvas = TCanvas('canvas', 'canvas', 800, 600)
    hist.SetTitle('TOF mass of ^{3}He; TOF mass (GeV/c^{2}); Counts')
    hist.Draw('hist')
    mass_fit.Draw('same')

    # Create legend
    legend = TLegend(0.6, 0.7, 0.89, 0.89)
    legend.SetBorderSize(0)
    legend.AddEntry(hist, 'TOF mass', 'l')
    legend.AddEntry(mass_fit, 'Gaussian fit', 'l')
    legend.Draw('same')

    # Write fit parameters and results  
    texts = [TText(3.5, 1400, f'Mean: {mass_fit.GetParameter(1):.2f}'),
            TText(3.5, 1300, f'Sigma: {mass_fit.GetParameter(2):.2f}'),
            TText(3.5, 1200, f'Chi2/NDF: {mass_fit.GetChisquare():.2f}/{mass_fit.GetNDF()}')]
    for text in texts:
        text.SetTextSize(0.03)
        #text.SetFontStyle(42)
        text.Draw()

    # Save the canvas
    canvas.SaveAs(output_path)


if __name__ == '__main__':

    output_file = '../plots/he3_mass_fit.pdf'
    in_file = TFile.Open('/data/galucia/its_pid/LHC23_pass4_skimmed/AnalysisResults.root')
    h2 = in_file.Get('lf-tree-creator-cluster-studies/LFTreeCreator/TOFmassHe')
    hist = h2.ProjectionY('hist', 1, h2.GetNbinsX())
    fit_he3_mass(hist, output_file)
    
    