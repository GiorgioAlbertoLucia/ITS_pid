
import os
import sys
import yaml
import numpy as np
import polars as pl
from ROOT import (TFile, TDirectory, TH1F, TH2F, TF1, TCanvas, gInterpreter, 
                  TGraphErrors, TMultiGraph, kRed, kGreen, kBlue, kOrange, kCyan)

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

if __name__ == '__main__':

    input_file = '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_small.root'
    input_file_eta_cut = '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_small_eta.root'

    inFile = TFile.Open(input_file, 'READ')
    inFile_eta_cut = TFile.Open(input_file_eta_cut, 'READ')

    mg = TMultiGraph('BB_param_Pr_eta', 'Bethe Bloch parameterisation for Protons; #beta; #LT Cluster size #GT #times #LT cos#lambda #GT')
    g_Pr_eta_cut = inFile_eta_cut.Get('clsize_vs_beta_Pr/clSizeCosL_vs_beta_Pr_sig_points')
    obj_setter(g_Pr_eta_cut, title='Protons, |#eta| < 0.5', marker_color=kOrange, marker_style=20, marker_size=1.5)
    g_Pr = inFile.Get('clsize_vs_beta_Pr/clSizeCosL_vs_beta_Pr_sig_points')
    obj_setter(g_Pr, title='Protons', marker_color=kCyan-3, marker_style=21, marker_size=1.5)

    mg.Add(g_Pr_eta_cut)
    mg.Add(g_Pr)
    mg.GetXaxis().SetLimits(0.3, 1.0)

    BB_Pr = TF1('BB_Pr_no_cut', BetheBloch, 0.4, 1.0, 5)
    obj_setter(BB_Pr, title='BB Protons, no cut', line_color=kBlue)
    BB_Pr.SetParameters(-1.87030e-01,
                        -8.28695e+00,
                        -7.54629e-01,
                         1.35783e+00,
                         8.49713e-01)
    BB_Pr_eta_cut = TF1('BB_Pr_eta_cut', BetheBloch, 0.4, 1.0, 5)
    obj_setter(BB_Pr_eta_cut, title='BB Protons, |#eta| < 0.5', line_color=kRed)
    BB_Pr_eta_cut.SetParameters( -1.47536e-01,
                                 -1.14403e+01,
                                 -9.31692e-01,
                                 1.41496e+00,
                                 3.10087e-01)

    c = TCanvas('c', 'c', 800, 600)
    hframe = c.DrawFrame(0.35, 0.0, 1.05, 8, 'Bethe Bloch parameterisation for Protons; #beta; #LT Cluster size #GT #times #LT cos#lambda #GT')
    mg.Draw('P')
    BB_Pr.Draw('same')
    BB_Pr_eta_cut.Draw('same')

    leg = c.BuildLegend(0.4, 0.7, 0.9, 0.9)
    leg.SetTextSize(0.03)
    c.SaveAs('clsize_vs_beta_Pr_eta.pdf')