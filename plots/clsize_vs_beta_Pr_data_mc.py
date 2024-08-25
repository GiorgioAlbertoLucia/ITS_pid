
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

    input_file_data = '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_small.root'
    input_file_mc = '/home/galucia/ITS_pid/output/MC/bethe_bloch_parametrisation_small.root'

    inFileData = TFile.Open(input_file_data, 'READ')
    inFileMC = TFile.Open(input_file_mc, 'READ')

    mg = TMultiGraph('BB_param_Pr_data_mc', 'Bethe Bloch parameterisation for Protons from data and mc sample; #beta; #LT Cluster size #GT #times #LT cos#lambda #GT')
    g_Pr_data = inFileData.Get('clsize_vs_beta_Pr/clSizeCosL_vs_beta_Pr_sig_points')
    obj_setter(g_Pr_data, title='Data', marker_color=kCyan-3, marker_style=21, marker_size=1.5)
    g_Pr_mc = inFileMC.Get('clsize_vs_beta_Pr/clSizeCosL_vs_beta_Pr_sig_points')
    obj_setter(g_Pr_mc, title='MC', marker_color=kOrange, marker_style=20, marker_size=1.5)

    mg.Add(g_Pr_data)
    mg.Add(g_Pr_mc)

    BB_Pr_data = TF1('BB_Pr_data', BetheBloch, 0.4, 1.0, 5)
    obj_setter(BB_Pr_data, title='BB Data', line_color=kBlue)
    BB_Pr_data.SetParameters(-1.87030e-01,
                             -8.28695e+00,
                             -7.54629e-01,
                              1.35783e+00,
                              8.49713e-01)

    BB_Pr_mc = TF1('BB_Pr_mc', BetheBloch, 0.1, 1.0, 5)
    obj_setter(BB_Pr_mc, title='BB MC', line_color=kRed)
    BB_Pr_mc.SetParameters(-0.456394,
                           -4.711892,
                           -0.234565,
                           0.5352267,
                           6.0502629)

    c = TCanvas('c', 'c', 800, 600)
    mg.Draw('AP')
    BB_Pr_data.Draw('same')
    BB_Pr_mc.Draw('same')

    c.BuildLegend(0.7, 0.7, 0.9, 0.9)
    c.SaveAs('clsize_vs_beta_Pr_data_mc.pdf')