
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

    input_file_Pr = '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation.root'
    input_file_He = '/home/galucia/ITS_pid/output/LHC22o_pass4 skimmed/bethe_bloch_parametrisation.root'

    inFile_Pr = TFile.Open(input_file_Pr, 'READ')
    inFile_He = TFile.Open(input_file_He, 'READ')

    mg = TMultiGraph('BB_param_Ka_Pr', 'Bethe Bloch parameterisation for Kaons and Protons; #beta; #LT Cluster size #GT #times #LT cos#lambda #GT')
    g_He = inFile_He.Get('clsize_vs_beta_He/clSizeCosL_vs_beta_He_sig_points')
    obj_setter(g_He, title='^{3}He', marker_color=kOrange, marker_style=20, marker_size=1)
    g_Pr = inFile_Pr.Get('clsize_vs_beta_Pr/clSizeCosL_vs_beta_Pr_sig_points')
    obj_setter(g_Pr, title='p', marker_color=kCyan-3, marker_style=21, marker_size=1)

    mg.Add(g_He)
    mg.Add(g_Pr)
    mg.GetXaxis().SetLimits(0.3, 1.0)

    BB_Pr = TF1('BB_Pr', BetheBloch, 0.4, 1.0, 5)
    obj_setter(BB_Pr, title='BB p', line_color=kBlue)
    BB_Pr.SetParameters(-1.87030e-01,
                        -8.28695e+00,
                        -7.54629e-01,
                         1.35783e+00,
                         8.49713e-01)
    BB_He = TF1('BB_He', BetheBloch, 0.3, 1.0, 5)
    obj_setter(BB_He, title='BB ^{3}He', line_color=kRed)
    BB_He.SetParameters(-2.05155e-01,
                        -7.91945e+00,
                        -8.12819e-01,
                         1.25875e+00,
                         1.09963e+00)

    c = TCanvas('c', 'c', 800, 600)
    hframe = c.DrawFrame(0.3, 0.0, 1.0, 12)
    mg.Draw('AP')
    BB_Pr.Draw('same')
    BB_He.Draw('same')

    c.BuildLegend(0.7, 0.7, 0.9, 0.9)
    c.SaveAs('clsize_vs_beta_Pr_He.pdf')