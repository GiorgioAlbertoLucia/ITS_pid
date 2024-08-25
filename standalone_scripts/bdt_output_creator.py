'''
    Function to train a BDT from xgboost
'''

import os
import logging
import numpy as np
import xgboost as xgb
import polars as pl
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from typing import Tuple, List, Dict
import sys
sys.path.append('..')
from core.dataset import DataHandler
from core.bdt import BDTRegressorTrainer, BDTClassifierTrainer
from framework.utils.terminal_colors import TerminalColors as tc
from framework.utils.timeit import timeit
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from framework.src.graph_handler import GraphHandler
from framework.utils.root_setter import obj_setter
from framework.utils.matplotlib_to_root import save_mpl_to_root
import pickle

from ROOT import TFile, TDirectory, TH1F, TH2F, TGraph, TCanvas, TLegend
from ROOT import kRed, kGreen, kBlue, kOrange, kCyan

def efficiency_purity_per_momentum_bin(hists: Dict[str, Dict[str, TH1F]], particles:List[str], momentum_bin:float, outfile:TDirectory, outfile_hists:TDirectory):
    '''
        Calculate efficiency and purity per momentum bin

        Parameters
        ----------
        hists (Dict[str, Dict[str, TH1F]]): histograms for BDT score -> [predicted_class][true_class]
    '''

    roc_data = pl.DataFrame({'particle': pl.Series([], dtype=str),
                             'momentum': pl.Series([], dtype=pl.Float32),
                             'threshold': pl.Series([], dtype=pl.Float32),
                             'efficiency': pl.Series([], dtype=pl.Float32),
                             'purity': pl.Series([], dtype=pl.Float32)})

    tot_Pr_Pr = hists['Pr']['Pr'].Integral()
    tot_Pr_Pi = hists['Pr']['Pi'].Integral()
    tot_Pr_Ka = hists['Pr']['Ka'].Integral()

    tot_Ka_Pi = hists['Ka']['Pi'].Integral()
    tot_Ka_Pr = hists['Ka']['Pr'].Integral()
    tot_Ka_Ka = hists['Ka']['Ka'].Integral()

    tot_Pi_Pr = hists['Pi']['Pr'].Integral()
    tot_Pi_Pi = hists['Pi']['Pi'].Integral()
    tot_Pi_Ka = hists['Pi']['Ka'].Integral()

    for th_bin in range(1, hists['Pr']['Pr'].GetNbinsX()+1):
    # Protons

        #tp_Pr, fp_Pr_Pi, fp_Pr_Ka, fn_Pr, tn_Pr_Pi, tn_Pr_Ka = 0., 0., 0., 0., 0., 0.
        #for ibin in range(th_bin, hists['Pr']['Pr'].GetNbinsX()+1):
        #    tp_Pr += hists['Pr']['Pr'].GetBinContent(ibin)
        #    fp_Pr_Pi += hists['Pr']['Pi'].GetBinContent(ibin)
        #    fp_Pr_Ka += hists['Pr']['Ka'].GetBinContent(ibin)
        #for ibin in range(1, th_bin):
        #    fn_Pr += hists['Pr']['Pr'].GetBinContent(ibin)
        #    tn_Pr_Pi += hists['Pr']['Pi'].GetBinContent(ibin)
        #    tn_Pr_Ka += hists['Pr']['Ka'].GetBinContent(ibin)
        tp_Pr = hists['Pr']['Pr'].Integral(th_bin, hists['Pr']['Pr'].GetNbinsX())
        fp_Pr_Pi = hists['Pr']['Pi'].Integral(th_bin, hists['Pr']['Pi'].GetNbinsX())
        fp_Pr_Ka = hists['Pr']['Ka'].Integral(th_bin, hists['Pr']['Ka'].GetNbinsX())
        fn_Pr = hists['Pr']['Pr'].Integral(1, th_bin)
        tn_Pr_Pi = hists['Pr']['Pi'].Integral(1, th_bin) 
        tn_Pr_Ka = hists['Pr']['Ka'].Integral(1, th_bin) 

        eff_Pr = tp_Pr / tot_Pr_Pr if tot_Pr_Pr > 0 else -1.
        pur_Pr = tp_Pr / tot_Pr_Pr / (tp_Pr / tot_Pr_Pr + fp_Pr_Pi / tot_Pr_Pi + fp_Pr_Ka / tot_Pr_Ka) if tot_Pr_Pi > 0  and tot_Pr_Ka > 0  and tot_Pr_Pr and tp_Pr > 0  else -1.

        roc_data_Pr = pl.DataFrame({'particle': pl.Series(['Pr'], dtype=str),
                                    'momentum': pl.Series([momentum_bin], dtype=pl.Float32),
                                    'threshold': pl.Series([hists['Pr']['Pr'].GetXaxis().GetBinCenter(th_bin)], dtype=pl.Float32),
                                    'efficiency': pl.Series([eff_Pr], dtype=pl.Float32),
                                    'purity': pl.Series([pur_Pr], dtype=pl.Float32)})

        # Kaons
        if True:
            tp_Ka = hists['Ka']['Ka'].Integral(th_bin, -1)
            fp_Ka_Pr = hists['Ka']['Pr'].Integral(th_bin, -1)
            fp_Ka_Pi = hists['Ka']['Pi'].Integral(th_bin, -1)
            fn_Ka = hists['Ka']['Ka'].Integral(1, th_bin)
            tn_Ka_Pr = hists['Ka']['Pr'].Integral(1, th_bin)
            tn_Ka_Pi = hists['Ka']['Pi'].Integral(1, th_bin)
        else:
            tp_Ka = hists['Pr']['Ka'].Integral(1, th_bin) + hists['Pi']['Ka'].Integral(1, th_bin)
            fp_Ka_Pr = hists['Pr']['Pr'].Integral(1, th_bin) + hists['Pi']['Pr'].Integral(1, th_bin)
            fp_Ka_Pi = hists['Pr']['Pi'].Integral(1, th_bin) + hists['Pi']['Pi'].Integral(1, th_bin)
            fn_Ka = hists['Pr']['Ka'].Integral(th_bin, -1) + hists['Pi']['Ka'].Integral(th_bin, -1)
            tn_Ka_Pr = hists['Pr']['Pr'].Integral(th_bin, -1) + hists['Pi']['Pr'].Integral(th_bin, -1)
            tn_Ka_Pi = hists['Pr']['Pi'].Integral(th_bin, -1) + hists['Pi']['Pi'].Integral(th_bin, -1)

            tot_Ka_Ka = hists['Pr']['Ka'].Integral() + hists['Pi']['Ka'].Integral()
            tot_Ka_Pr = hists['Pr']['Pr'].Integral() + hists['Pi']['Pr'].Integral()
            tot_Ka_Pi = hists['Pr']['Pi'].Integral() + hists['Pi']['Pi'].Integral()

        eff_Ka = tp_Ka / tot_Ka_Ka if tot_Ka_Ka > 0 else -1.
        pur_Ka = tp_Ka / tot_Ka_Ka / (tp_Ka / tot_Ka_Ka + fp_Ka_Pr / tot_Ka_Pr + fp_Ka_Pi / tot_Ka_Pi) if tot_Ka_Pi > 0 and tot_Ka_Ka > 0 and tot_Ka_Pr > 0  and tp_Ka > 0 else -1.

        #eff_Ka, pur_Ka = -1., -1.

        roc_data_Ka = pl.DataFrame({'particle': pl.Series(['Ka'], dtype=str),
                                    'momentum': pl.Series([momentum_bin], dtype=pl.Float32),
                                    'threshold': pl.Series([hists['Ka']['Ka'].GetXaxis().GetBinCenter(th_bin)], dtype=pl.Float32),
                                    'efficiency': pl.Series([eff_Ka], dtype=pl.Float32),
                                    'purity': pl.Series([pur_Ka], dtype=pl.Float32)})

        if len(roc_data) == 0:
            roc_data = pl.concat([roc_data_Pr, roc_data_Ka])
        else:
            roc_data = pl.concat([roc_data, roc_data_Pr, roc_data_Ka])

        # Pions
        tp_Pi = hists['Pi']['Pi'].Integral(th_bin, -1)
        fp_Pi_Pr = hists['Pi']['Pr'].Integral(th_bin, -1)
        fp_Pi_Ka = hists['Pi']['Ka'].Integral(th_bin, -1)
        fn_Pi = hists['Pi']['Pi'].Integral(1, th_bin)
        tn_Pi_Pr = hists['Pi']['Pr'].Integral(1, th_bin)
        tn_Pi_Ka = hists['Pi']['Ka'].Integral(1, th_bin)

        eff_Pi = tp_Pi / tot_Pi_Pi if tot_Pi_Pi > 0 else -1.
        pur_Pi = tp_Pi / tot_Pi_Pi / (tp_Pi / tot_Pi_Pi + fp_Pi_Pr / tot_Pi_Pr + fp_Pi_Ka / tot_Pi_Ka) if tot_Pi_Pi > 0 and tot_Pi_Ka > 0 and tot_Pi_Pr > 0 and tp_Pi > 0 else -1.

        roc_data_Pi = pl.DataFrame({'particle': pl.Series(['Pi'], dtype=str),
                                    'momentum': pl.Series([momentum_bin], dtype=pl.Float32),
                                    'threshold': pl.Series([hists['Pi']['Pi'].GetXaxis().GetBinCenter(th_bin)], dtype=pl.Float32),
                                    'efficiency': pl.Series([eff_Pi], dtype=pl.Float32),
                                    'purity': pl.Series([pur_Pi], dtype=pl.Float32)})

        if len(roc_data) == 0:
            roc_data = pl.concat([roc_data_Pr, roc_data_Pi])
        else:
            roc_data = pl.concat([roc_data, roc_data_Pr, roc_data_Pi])

    outfile_hists.cd()
    for pred_particle in particles:
        for true_particle in particles:
            hist = hists[pred_particle][true_particle].Clone()
            hist.Scale(1./hist.Integral())
            hist.Write(f'{pred_particle}_{true_particle}_{momentum_bin}')

    for pred_particle in particles:
        canvas = TCanvas(f'canvas_{pred_particle}_{momentum_bin}', f'canvas_{pred_particle}', 800, 600)
        colors = [2, 3, 4]
        names = ['#pi', 'K', 'p']
        tmp_hists = []
        for ipart, true_particle in enumerate(particles):
            hist = hists[pred_particle][true_particle].Clone()
            hist.Scale(1./hist.Integral())
            tmp_hists.append(hist)

        ymax = max([hist.GetMaximum() for hist in tmp_hists])
        hframe = canvas.DrawFrame(0., 0., 1., 1.1*ymax, f';{pred_particle} score; Normalized counts')
        for ihist, hist in enumerate(tmp_hists):
            obj_setter(hist, title=f'{true_particle}', fill_color=colors[ihist], fill_style=3013, fill_alpha=0.5)
            hist.Draw('hist same')
        legend = TLegend(0.7, 0.15, 0.9, 0.35)
        for ihist, hist in enumerate(tmp_hists):
            legend.AddEntry(hist, names[ihist], 'f')
        legend.Draw('same')
        canvas.Write()
        canvas.SaveAs(f'/home/galucia/ITS_pid/output/tmp.pdf')

    return roc_data

def efficiency_purity_vs_momentum(input_file:str, output_file:TDirectory):
    '''
        Calculate efficiency and purity per momentum bin

        Parameters
        ----------
        input_file (str): input file
        output_file (str): output file

    '''

    roc_data = pl.DataFrame({'particle': pl.Series(values=[], dtype=str),
                             'momentum': pl.Series(values=[], dtype=pl.Float32),
                             'threshold': pl.Series(values=[], dtype=pl.Float32),
                             'efficiency': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Pi': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Ka': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Pr': pl.Series(values=[], dtype=pl.Float32)})

    hist_names = {'Pr': {'Pr': 'class_scores_test_Pr_predas_Pr', 'Pi': 'class_scores_test_Pi_predas_Pr', 'Ka': 'class_scores_test_Ka_predas_Pr'},
                  'Pi': {'Pr': 'class_scores_test_Pr_predas_Pi', 'Pi': 'class_scores_test_Pi_predas_Pi', 'Ka': 'class_scores_test_Ka_predas_Pi'},
                  'Ka': {'Pr': 'class_scores_test_Pr_predas_Ka', 'Pi': 'class_scores_test_Pi_predas_Ka', 'Ka': 'class_scores_test_Ka_predas_Ka'}}

    hists = {'Pr': {'Pr': None, 'Pi': None, 'Ka': None},
             'Pi': {'Pr': None, 'Pi': None, 'Ka': None},
             'Ka': {'Pr': None, 'Pi': None, 'Ka': None}}

    input_file_root = TFile(input_file, 'READ')

    for pred_particle in ['Pr', 'Pi', 'Ka']:
        for true_particle in ['Pr', 'Pi', 'Ka']:
            hist = input_file_root.Get(hist_names[pred_particle][true_particle])
            hist.Rebin(5)
            hists[pred_particle][true_particle] = hist.Clone()


            del hist
    
    pmin = 0.2
    pmax = 2.5
    for momentum_bin in range(hists['Pr']['Pr'].GetXaxis().FindBin(pmin), hists['Pr']['Pr'].GetXaxis().FindBin(pmax)):
        hist_slices = {'Pr': {'Pr': None, 'Pi': None, 'Ka': None},
                       'Pi': {'Pr': None, 'Pi': None, 'Ka': None},
                       'Ka': {'Pr': None, 'Pi': None, 'Ka': None}}
        for pred_particle in ['Pr', 'Pi', 'Ka']:
            for true_particle in ['Pr', 'Pi', 'Ka']:
                hist = hists[pred_particle][true_particle]
                hist_slices[pred_particle][true_particle] = hist.ProjectionY(hist.GetName()+'_sliced', momentum_bin, momentum_bin)
                
        roc_data_momentum = efficiency_purity_per_momentum_bin(hist_slices, ['Pr', 'Pi', 'Ka'], hists['Pr']['Pr'].GetXaxis().GetBinCenter(momentum_bin), output_file)
        if len(roc_data) == 0:
            roc_data = roc_data_momentum
        else:
            roc_data = pl.concat([roc_data, roc_data_momentum])

    roc_data.write_csv('/home/galucia/ITS_pid/output/roc_data.csv')

    outdir = output_file.mkdir('ROC_Pr_Pi')

    for momentum_bin in roc_data['momentum'].unique():

        # Roc Pr
        roc_data_Pr = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Pr'))

        roc_data_Pr_Pi = roc_data_Pr.filter(pl.col('purity_Pi') > -0.5)
        roc_data_Pr_Pi = roc_data_Pr_Pi.filter(pl.col('efficiency') > -0.5)
        graph_handler_Pr_Pi = GraphHandler(roc_data_Pr_Pi)
        roc_Pr_Pi = graph_handler_Pr_Pi.createTGraph('purity_Pi', 'efficiency')
        if roc_Pr_Pi is not None:
            obj_setter(roc_Pr_Pi, name=f'roc_{momentum_bin:.2f}_Pr_Pi', title=f'ROC curve for Pr vs Pi at {momentum_bin:.2f} GeV/c; Purity; Efficiency', marker_color=4, marker_style=20)
            outdir.cd()
            roc_Pr_Pi.Write()
        roc_data_Pr_Ka = roc_data_Pr.filter(pl.col('efficiency') > -0.5)
        roc_data_Pr_Ka = roc_data_Pr_Ka.filter(pl.col('purity_Ka') > -0.5)
        graph_handler_Pr_Ka = GraphHandler(roc_data_Pr_Ka)
        roc_Pr_Ka = graph_handler_Pr_Ka.createTGraph('purity_Ka', 'efficiency')
        if roc_Pr_Ka is not None:
            obj_setter(roc_Pr_Ka, name=f'roc_{momentum_bin:.2f}_Pr_Ka', title=f'ROC curve for Pr vs Ka at {momentum_bin:.2f} GeV/c; Purity; Efficiency', marker_color=2, marker_style=21)
            outdir.cd()
            roc_Pr_Ka.Write()

        # Roc Pi
        roc_data_Pi = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Pi'))
        if (momentum_bin > 0.270 and momentum_bin < 0.280):
            print(roc_data_Pi)

        roc_data_Pi_Pr = roc_data_Pi.filter(pl.col('purity_Pr') > -0.5)
        roc_data_Pi_Pr = roc_data_Pi_Pr.filter(pl.col('efficiency') > -0.5)
        graph_handler_Pi_Pr = GraphHandler(roc_data_Pi_Pr)
        roc_Pi_Pr = graph_handler_Pi_Pr.createTGraph('purity_Pr', 'efficiency')
        if roc_Pi_Pr is not None:
            obj_setter(roc_Pi_Pr, name=f'roc_{momentum_bin:.2f}_Pi_Pr', title=f'ROC curve for Pi vs Pr at {momentum_bin} GeV/c; Purity; Efficiency', marker_color=4, marker_style=20)
            outdir.cd()
            roc_Pi_Pr.Write()

        roc_data_Pi_Ka = roc_data_Pi.filter(pl.col('efficiency') > -0.5)
        roc_data_Pi_Ka = roc_data_Pi_Ka.filter(pl.col('purity_Ka') > -0.5)
        graph_handler_Pi_Ka = GraphHandler(roc_data_Pi_Ka)
        roc_Pi_Ka = graph_handler_Pi_Ka.createTGraph('purity_Ka', 'efficiency')
        if roc_Pi_Ka is not None:
            obj_setter(roc_Pi_Ka, name=f'roc_{momentum_bin:.2f}_Pi_Ka', title=f'ROC curve for Pi vs Ka at {momentum_bin:.2f} GeV/c; Purity; Efficiency', marker_color=2, marker_style=21)
            outdir.cd()
            roc_Pi_Ka.Write()

def efficiency_purity_vs_momentum_ensemble(input_file:str, output_file:TDirectory):
    '''
        Calculate efficiency and purity per momentum bin

        Parameters
        ----------
        input_file (str): input file
        output_file (str): output file

    '''

    roc_data = pl.DataFrame({'particle': pl.Series(values=[], dtype=str),
                             'momentum': pl.Series(values=[], dtype=pl.Float32),
                             'threshold': pl.Series(values=[], dtype=pl.Float32),
                             'efficiency': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Pi': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Ka': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Pr': pl.Series(values=[], dtype=pl.Float32)})

    hist_names = {'Pr': {'Pr': 'class_scores_test_Pr_predas_Pr', 'Pi': 'class_scores_test_Pi_predas_Pr', 'Ka': 'class_scores_test_Ka_predas_Pr'},
                  'Pi': {'Pr': 'class_scores_test_Pr_predas_Pi', 'Pi': 'class_scores_test_Pi_predas_Pi', 'Ka': 'class_scores_test_Ka_predas_Pi'},
                  'Ka': {'Pr': 'class_scores_test_Pr_predas_Ka', 'Pi': 'class_scores_test_Pi_predas_Ka', 'Ka': 'class_scores_test_Ka_predas_Ka'}}

    hists = {'Pr': {'Pr': None, 'Pi': None, 'Ka': None},
             'Pi': {'Pr': None, 'Pi': None, 'Ka': None},
             'Ka': {'Pr': None, 'Pi': None, 'Ka': None}}

    input_file_root = TFile(input_file, 'READ')
    outdir_roc = output_file.mkdir('ROC')
    outdir_hists = output_file.mkdir('Hists')

    momentum_bins = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    
    for momentum_bin in range(len(momentum_bins)-1):
        hist_slices = {'Pr': {'Pr': None, 'Pi': None, 'Ka': None},
                       'Pi': {'Pr': None, 'Pi': None, 'Ka': None},
                       'Ka': {'Pr': None, 'Pi': None, 'Ka': None}}
        #indir = input_file_root.Get(f'bin_{momentum_bin}')
        for pred_particle in ['Pr', 'Pi', 'Ka']:
            for true_particle in ['Pr', 'Pi', 'Ka']:
                hist = input_file_root.Get(f'bin_{momentum_bin}/'+hist_names[pred_particle][true_particle])
                hist_slices[pred_particle][true_particle] = hist.Clone()
                del hist
                
        roc_data_momentum = efficiency_purity_per_momentum_bin(hist_slices, ['Pi', 'Ka', 'Pr'], (momentum_bins[momentum_bin] + momentum_bins[momentum_bin+1])/2., outdir_roc, outdir_hists)
        if len(roc_data) == 0:
            roc_data = roc_data_momentum
        else:
            roc_data = pl.concat([roc_data, roc_data_momentum])

    roc_data.write_csv('/home/galucia/ITS_pid/output/roc_data.csv')

    for momentum_bin in roc_data['momentum'].unique():

        # Roc Pr
        roc_data_Pr = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Pr'))
        roc_data_Pr = roc_data_Pr.filter(pl.col('purity') > -0.5)
        roc_data_Pr = roc_data_Pr.filter(pl.col('efficiency') > -0.5)
        graph_handler_Pr = GraphHandler(roc_data_Pr)
        roc_Pr = graph_handler_Pr.createTGraph('purity', 'efficiency')
        if roc_Pr is not None:
            obj_setter(roc_Pr, name=f'roc_{momentum_bin:.2f}_Pr', title=f'ROC curve for Pr at {momentum_bin:.2f} GeV/#it{{c}}; Purity; Efficiency', marker_color=4, marker_style=20)
            outdir_roc.cd()
            roc_Pr.Write()

        # Roc Ka
        roc_data_Ka = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Ka'))
        roc_data_Ka = roc_data_Ka.filter(pl.col('efficiency') > -0.5)
        roc_data_Ka = roc_data_Ka.filter(pl.col('purity') > -0.5)
        graph_handler_Ka = GraphHandler(roc_data_Ka)
        roc_Ka = graph_handler_Ka.createTGraph('purity', 'efficiency')
        if roc_Ka is not None:
            obj_setter(roc_Ka, name=f'roc_{momentum_bin:.2f}_Ka', title=f'ROC curve for Ka at {momentum_bin:.2f} GeV/#it{{c}}; Purity; Efficiency', marker_color=2, marker_style=21)
            outdir_roc.cd()
            roc_Ka.Write()

        # Roc Pi
        roc_data_Pi = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Pi'))
        roc_data_Pi = roc_data_Pi.filter(pl.col('purity') > -0.5)
        roc_data_Pi = roc_data_Pi.filter(pl.col('efficiency') > -0.5)
        graph_handler_Pi = GraphHandler(roc_data_Pi)
        roc_Pi = graph_handler_Pi.createTGraph('purity', 'efficiency')
        if roc_Pi is not None:
            obj_setter(roc_Pi, name=f'roc_{momentum_bin:.2f}_Pi', title=f'ROC curve for Pi at {momentum_bin} GeV/#it{{c}}; Purity; Efficiency', marker_color=4, marker_style=20)
            outdir_roc.cd()
            roc_Pi.Write()


def efficiency_purity_vs_momentum_he(input_file:str, output_file:TDirectory):
    '''
        Calculate efficiency and purity per momentum bin

        Parameters
        ----------
        input_file (str): input file
        output_file (str): output file

    '''

    roc_data = pl.DataFrame({'particle': pl.Series(values=[], dtype=str),
                             'momentum': pl.Series(values=[], dtype=pl.Float32),
                             'threshold': pl.Series(values=[], dtype=pl.Float32),
                             'efficiency': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Pi': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Ka': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Pr': pl.Series(values=[], dtype=pl.Float32),
                             'purity_De': pl.Series(values=[], dtype=pl.Float32),
                             'purity_He': pl.Series(values=[], dtype=pl.Float32)})

    particles = ['Pr', 'Pi', 'Ka', 'De', 'He']

    hist_names = {part: {part: f'class_scores_test_{part}_predas_{part}' for part in particles}}
    hists = {part: {part: None for part in particles}}

    input_file_root = TFile(input_file, 'READ')

    for pred_particle in particles:
        for true_particle in particles:
            hist = input_file_root.Get(hist_names[pred_particle][true_particle])
            hist.Rebin(5)
            hists[pred_particle][true_particle] = hist.Clone()


            del hist
    
    pmin = 0.2
    pmax = 2.5
    for momentum_bin in range(hists['Pr']['Pr'].GetXaxis().FindBin(pmin), hists['Pr']['Pr'].GetXaxis().FindBin(pmax)):
        hist_slices = {'Pr': {'Pr': None, 'Pi': None, 'Ka': None},
                       'Pi': {'Pr': None, 'Pi': None, 'Ka': None},
                       'Ka': {'Pr': None, 'Pi': None, 'Ka': None}}
        for pred_particle in ['Pr', 'Pi', 'Ka']:
            for true_particle in ['Pr', 'Pi', 'Ka']:
                hist = hists[pred_particle][true_particle]
                hist_slices[pred_particle][true_particle] = hist.ProjectionY(hist.GetName()+'_sliced', momentum_bin, momentum_bin)
                
        roc_data_momentum = efficiency_purity_per_momentum_bin(hist_slices, ['Pr', 'Pi', 'Ka'], hists['Pr']['Pr'].GetXaxis().GetBinCenter(momentum_bin), output_file)
        if len(roc_data) == 0:
            roc_data = roc_data_momentum
        else:
            roc_data = pl.concat([roc_data, roc_data_momentum])

    roc_data.write_csv('/home/galucia/ITS_pid/output/roc_data.csv')

    outdir = output_file.mkdir('ROC_Pr_Pi')

    for momentum_bin in roc_data['momentum'].unique():

        # Roc Pr
        roc_data_Pr = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Pr'))

        roc_data_Pr_Pi = roc_data_Pr.filter(pl.col('purity_Pi') > -0.5)
        roc_data_Pr_Pi = roc_data_Pr_Pi.filter(pl.col('efficiency') > -0.5)
        graph_handler_Pr_Pi = GraphHandler(roc_data_Pr_Pi)
        roc_Pr_Pi = graph_handler_Pr_Pi.createTGraph('purity_Pi', 'efficiency')
        if roc_Pr_Pi is not None:
            obj_setter(roc_Pr_Pi, name=f'roc_{momentum_bin:.2f}_Pr_Pi', title=f'ROC curve for Pr vs Pi at {momentum_bin:.2f} GeV/c; Purity; Efficiency', marker_color=4, marker_style=20)
            outdir.cd()
            roc_Pr_Pi.Write()
        roc_data_Pr_Ka = roc_data_Pr.filter(pl.col('efficiency') > -0.5)
        roc_data_Pr_Ka = roc_data_Pr_Ka.filter(pl.col('purity_Ka') > -0.5)
        graph_handler_Pr_Ka = GraphHandler(roc_data_Pr_Ka)
        roc_Pr_Ka = graph_handler_Pr_Ka.createTGraph('purity_Ka', 'efficiency')
        if roc_Pr_Ka is not None:
            obj_setter(roc_Pr_Ka, name=f'roc_{momentum_bin:.2f}_Pr_Ka', title=f'ROC curve for Pr vs Ka at {momentum_bin:.2f} GeV/c; Purity; Efficiency', marker_color=2, marker_style=21)
            outdir.cd()
            roc_Pr_Ka.Write()

        # Roc Pi
        roc_data_Pi = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Pi'))
        if (momentum_bin > 0.270 and momentum_bin < 0.280):
            print(roc_data_Pi)

        roc_data_Pi_Pr = roc_data_Pi.filter(pl.col('purity_Pr') > -0.5)
        roc_data_Pi_Pr = roc_data_Pi_Pr.filter(pl.col('efficiency') > -0.5)
        graph_handler_Pi_Pr = GraphHandler(roc_data_Pi_Pr)
        roc_Pi_Pr = graph_handler_Pi_Pr.createTGraph('purity_Pr', 'efficiency')
        if roc_Pi_Pr is not None:
            obj_setter(roc_Pi_Pr, name=f'roc_{momentum_bin:.2f}_Pi_Pr', title=f'ROC curve for Pi vs Pr at {momentum_bin} GeV/c; Purity; Efficiency', marker_color=4, marker_style=20)
            outdir.cd()
            roc_Pi_Pr.Write()

        roc_data_Pi_Ka = roc_data_Pi.filter(pl.col('efficiency') > -0.5)
        roc_data_Pi_Ka = roc_data_Pi_Ka.filter(pl.col('purity_Ka') > -0.5)
        graph_handler_Pi_Ka = GraphHandler(roc_data_Pi_Ka)
        roc_Pi_Ka = graph_handler_Pi_Ka.createTGraph('purity_Ka', 'efficiency')
        if roc_Pi_Ka is not None:
            obj_setter(roc_Pi_Ka, name=f'roc_{momentum_bin:.2f}_Pi_Ka', title=f'ROC curve for Pi vs Ka at {momentum_bin:.2f} GeV/c; Purity; Efficiency', marker_color=2, marker_style=21)
            outdir.cd()
            roc_Pi_Ka.Write()

def efficiency_purity_vs_momentum_ensemble_he(input_file:str, output_file:TDirectory):
    '''
        Calculate efficiency and purity per momentum bin

        Parameters
        ----------
        input_file (str): input file
        output_file (str): output file

    '''

    roc_data = pl.DataFrame({'particle': pl.Series(values=[], dtype=str),
                             'momentum': pl.Series(values=[], dtype=pl.Float32),
                             'threshold': pl.Series(values=[], dtype=pl.Float32),
                             'efficiency': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Pi': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Ka': pl.Series(values=[], dtype=pl.Float32),
                             'purity_Pr': pl.Series(values=[], dtype=pl.Float32)})

    hist_names = {'Pr': {'Pr': 'class_scores_test_Pr_predas_Pr', 'Pi': 'class_scores_test_Pi_predas_Pr', 'Ka': 'class_scores_test_Ka_predas_Pr'},
                  'Pi': {'Pr': 'class_scores_test_Pr_predas_Pi', 'Pi': 'class_scores_test_Pi_predas_Pi', 'Ka': 'class_scores_test_Ka_predas_Pi'},
                  'Ka': {'Pr': 'class_scores_test_Pr_predas_Ka', 'Pi': 'class_scores_test_Pi_predas_Ka', 'Ka': 'class_scores_test_Ka_predas_Ka'}}

    hists = {'Pr': {'Pr': None, 'Pi': None, 'Ka': None},
             'Pi': {'Pr': None, 'Pi': None, 'Ka': None},
             'Ka': {'Pr': None, 'Pi': None, 'Ka': None}}

    input_file_root = TFile(input_file, 'READ')
    outdir_roc = output_file.mkdir('ROC')
    outdir_hists = output_file.mkdir('Hists')

    momentum_bins = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    
    for momentum_bin in range(len(momentum_bins)-1):
        hist_slices = {'Pr': {'Pr': None, 'Pi': None, 'Ka': None},
                       'Pi': {'Pr': None, 'Pi': None, 'Ka': None},
                       'Ka': {'Pr': None, 'Pi': None, 'Ka': None}}
        #indir = input_file_root.Get(f'bin_{momentum_bin}')
        for pred_particle in ['Pr', 'Pi', 'Ka']:
            for true_particle in ['Pr', 'Pi', 'Ka']:
                hist = input_file_root.Get(f'bin_{momentum_bin}/'+hist_names[pred_particle][true_particle])
                hist_slices[pred_particle][true_particle] = hist.Clone()
                del hist
                
        roc_data_momentum = efficiency_purity_per_momentum_bin(hist_slices, ['Pi', 'Ka', 'Pr'], (momentum_bins[momentum_bin] + momentum_bins[momentum_bin+1])/2., outdir_roc, outdir_hists)
        if len(roc_data) == 0:
            roc_data = roc_data_momentum
        else:
            roc_data = pl.concat([roc_data, roc_data_momentum])

    roc_data.write_csv('/home/galucia/ITS_pid/output/roc_data.csv')

    for momentum_bin in roc_data['momentum'].unique():

        # Roc Pr
        roc_data_Pr = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Pr'))
        roc_data_Pr = roc_data_Pr.filter(pl.col('purity') > -0.5)
        roc_data_Pr = roc_data_Pr.filter(pl.col('efficiency') > -0.5)
        graph_handler_Pr = GraphHandler(roc_data_Pr)
        roc_Pr = graph_handler_Pr.createTGraph('purity', 'efficiency')
        if roc_Pr is not None:
            obj_setter(roc_Pr, name=f'roc_{momentum_bin:.2f}_Pr', title=f'ROC curve for Pr at {momentum_bin:.2f} GeV/#it{{c}}; Purity; Efficiency', marker_color=4, marker_style=20)
            outdir_roc.cd()
            roc_Pr.Write()

        # Roc Ka
        roc_data_Ka = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Ka'))
        roc_data_Ka = roc_data_Ka.filter(pl.col('efficiency') > -0.5)
        roc_data_Ka = roc_data_Ka.filter(pl.col('purity') > -0.5)
        graph_handler_Ka = GraphHandler(roc_data_Ka)
        roc_Ka = graph_handler_Ka.createTGraph('purity', 'efficiency')
        if roc_Ka is not None:
            obj_setter(roc_Ka, name=f'roc_{momentum_bin:.2f}_Ka', title=f'ROC curve for Ka at {momentum_bin:.2f} GeV/#it{{c}}; Purity; Efficiency', marker_color=2, marker_style=21)
            outdir_roc.cd()
            roc_Ka.Write()

        # Roc Pi
        roc_data_Pi = roc_data.filter(((pl.col('momentum') > momentum_bin-0.005) & (pl.col('momentum') < momentum_bin+0.005)) & (pl.col('particle') == 'Pi'))
        roc_data_Pi = roc_data_Pi.filter(pl.col('purity') > -0.5)
        roc_data_Pi = roc_data_Pi.filter(pl.col('efficiency') > -0.5)
        graph_handler_Pi = GraphHandler(roc_data_Pi)
        roc_Pi = graph_handler_Pi.createTGraph('purity', 'efficiency')
        if roc_Pi is not None:
            obj_setter(roc_Pi, name=f'roc_{momentum_bin:.2f}_Pi', title=f'ROC curve for Pi at {momentum_bin} GeV/#it{{c}}; Purity; Efficiency', marker_color=4, marker_style=20)
            outdir_roc.cd()
            roc_Pi.Write()


if __name__ == '__main__':

    # Configure logging
    os.remove("/home/galucia/ITS_pid/output/output_bdt.log")
    logging.basicConfig(filename="/home/galucia/ITS_pid/output/output_bdt.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    #input_files = ['../../data/0720/its_PIDStudy.root']
    input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = '../config/config_outputs.yml'
    
    #output_file = '../output/bdt_beta.root'
    #input_file = '../output/bdt_cls_20082024_35off.root'
    #input_file = '../output/bdt_cls_22082024.root'
    input_file = '../output/bdt_cls_22082024_he.root'
    #output_file = '../output/bdt_cls_output_35off.root'
    #output_file = '../output/bdt_cls_output_22082024.root'
    output_file = '../output/bdt_cls_output_22082024_he.root'
    output_file_root = TFile(output_file, 'RECREATE')
    
    #train_data = pl.read_parquet('/home/galucia/ITS_pid/output/bdt_cls_train.parquet')
    #test_data = pl.read_parquet('/home/galucia/ITS_pid/output/bdt_cls_test.parquet')
#
    ### Classifier
    #cfg_bdt_file = '../config/config_bdt_cls.yml'
    #bdt_classifier = BDTClassifierTrainer(cfg_bdt_file, output_file_root)
    #bdt_classifier.load_data(train_data, test_data)
    #bdt_classifier.prepare_for_plots()
#
    #bdt_classifier.draw_part_id_distribution()
    #bdt_classifier.draw_efficiency_purity_vs_momentum(0.3, 1.0, 0.1)
    #bdt_classifier.draw_confusion_matrix()

    #efficiency_purity_vs_momentum(input_file, output_file_root)
    efficiency_purity_vs_momentum_ensemble(input_file, output_file_root)

    #check(input_file, output_file_root)

    logging.info('closing output file') 
    output_file_root.Close()
