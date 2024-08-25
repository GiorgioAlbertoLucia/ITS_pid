import polars as pl
import numpy as np
from typing import List, Dict

from ROOT import TFile, TH1F, TDirectory, TGraphErrors, TGraph

import sys
sys.path.append('..')
from framework.src.graph_handler import GraphHandler
from framework.utils.root_setter import obj_setter


def compute_purity(fit_part:str, part2:str, part3:str, infile_path:str, outfile:TDirectory):

    purity = pl.DataFrame({'x': pl.Series(values=[], dtype=pl.Float64),
                           'purity': pl.Series(values=[], dtype=pl.Float64),
                           'sx': pl.Series(values=[], dtype=pl.Float64),
                           'spurity': pl.Series(values=[], dtype=pl.Float64)})

    infile = TFile(infile_path, 'READ')
    fit_h2 = infile.Get(f'clsize_vs_p_nsigma_{fit_part}/nsigma_vs_p_{fit_part}_nsigma')
    part2_h2 = infile.Get(f'clsize_vs_p_nsigma_{part2}/nsigma_vs_p_{part2}_nsigma')
    part3_h2 = infile.Get(f'clsize_vs_p_nsigma_{part3}/nsigma_vs_p_{part3}_nsigma')

    first_bin = fit_h2.GetXaxis().FindBin(0.3)
    last_bin = fit_h2.GetXaxis().FindBin(5)

    for ix in range(first_bin, last_bin + 1):

        print(f'Processing bin {ix}: p = ', fit_h2.GetXaxis().GetBinCenter(ix))
        fit_h1 = fit_h2.ProjectionY(f'fit_h1_{ix}', ix, ix)
        part2_h1 = part2_h2.ProjectionY(f'part2_h1_{ix}', ix, ix)
        part3_h1 = part3_h2.ProjectionY(f'part3_h1_{ix}', ix, ix)

        fit_int = fit_h1.Integral()
        part2_int = part2_h1.Integral()
        part3_int = part3_h1.Integral()
        fit_pos_int = fit_h1.Integral(fit_h1.FindBin(-2), fit_h1.FindBin(2))
        part2_pos_int = part2_h1.Integral(part2_h1.FindBin(-2), part2_h1.FindBin(2))
        part3_pos_int = part3_h1.Integral(part3_h1.FindBin(-2), part3_h1.FindBin(2))

        if fit_int == 0 or part2_int == 0 or part3_int == 0:
            continue
        
        a = fit_pos_int / fit_int
        b = part2_pos_int / part2_int
        c = part3_pos_int / part3_int
        sa = np.sqrt(a * (1 - a) / fit_int) if fit_int > 0 else 0
        sb = np.sqrt(b * (1 - b) / part2_int) if part2_int > 0 else 0
        sc = np.sqrt(c * (1 - c) / part3_int) if part3_int > 0 else 0

        purity_val = a / (a + b + c) if a > 0 and b > 0 and c > 0 else -1
        spurity_val = np.sqrt(sa**2 * (b + c)**2 + sb**2 * a**2 + sc**2 * a**2) / (a + b + c) if a + b + c > 0 else -1

        purity = pl.concat([purity, pl.DataFrame({'x': pl.Series(values=[fit_h2.GetXaxis().GetBinCenter(ix)], dtype=pl.Float64),
                                                  'purity': pl.Series(values=[purity_val], dtype=pl.Float64),
                                                  'sx': pl.Series(values=[0], dtype=pl.Float64),
                                                  'spurity': pl.Series(values=[0.], dtype=pl.Float64)})])
                                                  #'spurity': pl.Series(values=[np.sqrt(purity_val * (1 - purity_val) / (part2_pos_int / part2_int + fit_pos_int / fit_int))], dtype=pl.Float64)})])

    graph_handler = GraphHandler(purity)
    graph = graph_handler.createTGraphErrors('x', 'purity', 'sx', 'spurity')
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1)
    graph.SetMarkerColor(4)
    
    outfile.cd()
    graph.Write(f'purity_{fit_part}')

def compute_efficiency(fit_part:str, infile_path:str, outfile:TDirectory):

    efficiency = pl.DataFrame({'x': pl.Series(values=[], dtype=pl.Float64),
                           'efficiency': pl.Series(values=[], dtype=pl.Float64),
                           'sx': pl.Series(values=[], dtype=pl.Float64),
                           'sefficiency': pl.Series(values=[], dtype=pl.Float64)})

    infile = TFile(infile_path, 'READ')
    fit_h2 = infile.Get(f'clsize_vs_p_nsigma_{fit_part}/nsigma_vs_p_{fit_part}_nsigma')

    first_bin = fit_h2.GetXaxis().FindBin(0.3)
    last_bin = fit_h2.GetXaxis().FindBin(5)

    for ix in range(first_bin, last_bin + 1):

        print(f'Processing bin {ix}: p = ', fit_h2.GetXaxis().GetBinCenter(ix))
        fit_h1 = fit_h2.ProjectionY(f'fit_h1_{ix}', ix, ix)

        fit_int = fit_h1.Integral()
        fit_pos_int = fit_h1.Integral(fit_h1.FindBin(-2), fit_h1.FindBin(2))

        if fit_int == 0:
            continue
        efficiency_val = (fit_pos_int / fit_int)
        efficiency = pl.concat([efficiency, pl.DataFrame({'x': pl.Series(values=[fit_h2.GetXaxis().GetBinCenter(ix)], dtype=pl.Float64),
                                                  'efficiency': pl.Series(values=[efficiency_val], dtype=pl.Float64),
                                                  'sx': pl.Series(values=[0], dtype=pl.Float64),
                                                  'sefficiency': pl.Series(values=[np.sqrt(efficiency_val * (1 - efficiency_val) / fit_int)], dtype=pl.Float64)})])

    graph_handler = GraphHandler(efficiency)
    graph = graph_handler.createTGraphErrors('x', 'efficiency', 'sx', 'sefficiency')
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1)
    graph.SetMarkerColor(4)

    outfile.cd()
    graph.Write(f'efficiency_{fit_part}')


def compute_purity_efficiency_per_bin(hists:Dict[str, Dict[str, TH1F]], particles:List[str], momentum_bin:float):

    # Pions
    tot_Pi_Pi = hists['Pi']['Pi'].Integral()
    tot_Pi_Ka = hists['Pi']['Ka'].Integral()
    tot_Pi_Pr = hists['Pi']['Pr'].Integral()

    tp_Pi = hists['Pi']['Pi'].Integral(hists['Pi']['Pi'].FindBin(-1), hists['Pi']['Pi'].FindBin(1))
    fp_Pi_Ka = hists['Pi']['Ka'].Integral(hists['Pi']['Ka'].FindBin(-1), hists['Pi']['Ka'].FindBin(1))
    fp_Pi_Pr = hists['Pi']['Pr'].Integral(hists['Pi']['Pr'].FindBin(-1), hists['Pi']['Pr'].FindBin(1))

    eff_Pi = tp_Pi / tot_Pi_Pi if tot_Pi_Pi > 0 else -1
    seff_Pi = np.sqrt(eff_Pi * (1 - eff_Pi) / tot_Pi_Pi) if tot_Pi_Pi > 0 else -1
    pur_Pi = tp_Pi / tot_Pi_Pi / (tp_Pi / tot_Pi_Pi + fp_Pi_Ka / tot_Pi_Ka + fp_Pi_Pr / tot_Pi_Pr) if tot_Pi_Pi > 0 and tot_Pi_Ka > 0 and tot_Pi_Pr > 0 and tp_Pi > 0 else -1.
    spur_Pi = np.sqrt(pur_Pi * (1 - pur_Pi) / (tp_Pi / tot_Pi_Pi + fp_Pi_Ka / tot_Pi_Ka + fp_Pi_Pr / tot_Pi_Pr)) if tot_Pi_Pi > 0 and tot_Pi_Ka > 0 and tot_Pi_Pr > 0 and tp_Pi > 0 else -1.
    
    # Kaons
    tot_Ka_Pi = hists['Ka']['Pi'].Integral()
    tot_Ka_Ka = hists['Ka']['Ka'].Integral()
    tot_Ka_Pr = hists['Ka']['Pr'].Integral()

    tp_Ka = hists['Ka']['Ka'].Integral(hists['Ka']['Ka'].FindBin(-1), hists['Ka']['Ka'].FindBin(1))
    fp_Ka_Pi = hists['Ka']['Pi'].Integral(hists['Ka']['Pi'].FindBin(-1), hists['Ka']['Pi'].FindBin(1))
    fp_Ka_Pr = hists['Ka']['Pr'].Integral(hists['Ka']['Pr'].FindBin(-1), hists['Ka']['Pr'].FindBin(1))

    eff_Ka = tp_Ka / tot_Ka_Ka if tot_Ka_Ka > 0 else -1
    seff_Ka = np.sqrt(eff_Ka * (1 - eff_Ka) / tot_Ka_Ka) if tot_Ka_Ka > 0 else -1
    pur_Ka = tp_Ka / tot_Ka_Ka / (tp_Ka / tot_Ka_Ka + fp_Ka_Pi / tot_Ka_Pi + fp_Ka_Pr / tot_Ka_Pr) if tot_Ka_Ka > 0 and tot_Ka_Pi > 0 and tot_Ka_Pr > 0 and tp_Ka > 0 else -1.
    spur_Ka = np.sqrt(pur_Ka * (1 - pur_Ka) / (tp_Ka / tot_Ka_Ka + fp_Ka_Pi / tot_Ka_Pi + fp_Ka_Pr / tot_Ka_Pr)) if tot_Ka_Ka > 0 and tot_Ka_Pi > 0 and tot_Ka_Pr > 0 and tp_Ka > 0 else -1.

    # Protons
    tot_Pr_Pi = hists['Pr']['Pi'].Integral()
    tot_Pr_Ka = hists['Pr']['Ka'].Integral()
    tot_Pr_Pr = hists['Pr']['Pr'].Integral()

    tp_Pr = hists['Pr']['Pr'].Integral(hists['Pr']['Pr'].FindBin(-1), hists['Pr']['Pr'].FindBin(1))
    fp_Pr_Pi = hists['Pr']['Pi'].Integral(hists['Pr']['Pi'].FindBin(-1), hists['Pr']['Pi'].FindBin(1))
    fp_Pr_Ka = hists['Pr']['Ka'].Integral(hists['Pr']['Ka'].FindBin(-1), hists['Pr']['Ka'].FindBin(1))

    eff_Pr = tp_Pr / tot_Pr_Pr if tot_Pr_Pr > 0 else -1
    seff_Pr = np.sqrt(eff_Pr * (1 - eff_Pr) / tot_Pr_Pr) if tot_Pr_Pr > 0 else -1
    pur_Pr = tp_Pr / tot_Pr_Pr / (tp_Pr / tot_Pr_Pr + fp_Pr_Pi / tot_Pr_Pi + fp_Pr_Ka / tot_Pr_Ka) if tot_Pr_Pr > 0 and tot_Pr_Pi > 0 and tot_Pr_Ka > 0 and tp_Pr > 0 else -1.
    spur_Pr = np.sqrt(pur_Pr * (1 - pur_Pr) / (tp_Pr / tot_Pr_Pr + fp_Pr_Pi / tot_Pr_Pi + fp_Pr_Ka / tot_Pr_Ka)) if tot_Pr_Pr > 0 and tot_Pr_Pi > 0 and tot_Pr_Ka > 0 and tp_Pr > 0 else -1.

    return pl.DataFrame({'x': pl.Series(values=[momentum_bin, momentum_bin, momentum_bin], dtype=pl.Float64),
                                        'sx': pl.Series(values=[0, 0, 0], dtype=pl.Float64),
                                        'purity': pl.Series(values=[pur_Pi, pur_Ka, pur_Pr], dtype=pl.Float64),
                                        'spurity': pl.Series(values=[spur_Pi, spur_Ka, spur_Pr], dtype=pl.Float64),
                                        'efficiency': pl.Series(values=[eff_Pi, eff_Ka, eff_Pr], dtype=pl.Float64),
                                        'sefficiency': pl.Series(values=[seff_Pi, seff_Ka, seff_Pr], dtype=pl.Float64),
                                        'particle': pl.Series(values=['Pi', 'Ka', 'Pr'], dtype=str)})

def compute_efficiency_purity(infile_path:str, outfile:TFile):

    infile = TFile(infile_path, 'READ')
    purity_efficiency = pl.DataFrame({'x': pl.Series(values=[], dtype=pl.Float64),
                                      'sx': pl.Series(values=[], dtype=pl.Float64),
                                      'purity': pl.Series(values=[], dtype=pl.Float64),
                                      'spurity': pl.Series(values=[], dtype=pl.Float64),
                                      'efficiency': pl.Series(values=[], dtype=pl.Float64),
                                      'sefficiency': pl.Series(values=[], dtype=pl.Float64),
                                      'particle': pl.Series(values=[], dtype=str)})

    momentum_bins = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95] #, 1.05]
    hists = {
        'Pi': {
            'Pi': infile.Get('clsize_vs_p_nsigma_Pi/nsigma_vs_p_Pi_nsigma').RebinX(10),
            'Ka': infile.Get('clsize_vs_p_nsigma_Pi/nsigma_vs_p_Ka_nsigma').RebinX(10),
            'Pr': infile.Get('clsize_vs_p_nsigma_Pi/nsigma_vs_p_Pr_nsigma').RebinX(10)
        },
        'Ka': {
            'Pi': infile.Get('clsize_vs_p_nsigma_Ka/nsigma_vs_p_Pi_nsigma').RebinX(10),
            'Ka': infile.Get('clsize_vs_p_nsigma_Ka/nsigma_vs_p_Ka_nsigma').RebinX(10),
            'Pr': infile.Get('clsize_vs_p_nsigma_Ka/nsigma_vs_p_Pr_nsigma').RebinX(10)
        },
        'Pr': {
            'Pi': infile.Get('clsize_vs_p_nsigma_Pr/nsigma_vs_p_Pi_nsigma').RebinX(10),
            'Ka': infile.Get('clsize_vs_p_nsigma_Pr/nsigma_vs_p_Ka_nsigma').RebinX(10),
            'Pr': infile.Get('clsize_vs_p_nsigma_Pr/nsigma_vs_p_Pr_nsigma').RebinX(10)
        }
    }
    hist_slices = {
        'Pi': {
            'Pi': None,
            'Ka': None,
            'Pr': None
        },
        'Ka': {
            'Pi': None,
            'Ka': None,
            'Pr': None
        },
        'Pr': {
            'Pi': None,
            'Ka': None,
            'Pr': None
        }
    }

    for momentum_bin in momentum_bins:

        for pred_part in ['Pi', 'Ka', 'Pr']:
            for true_part in ['Pi', 'Ka', 'Pr']:
                hist_slices[pred_part][true_part] = hists[pred_part][true_part].ProjectionY(f'{pred_part}_{true_part}_{momentum_bin}', hists[pred_part][true_part].GetXaxis().FindBin(momentum_bin), hists[pred_part][true_part].GetXaxis().FindBin(momentum_bin))

        if len(purity_efficiency) == 0:
            purity_efficiency = compute_purity_efficiency_per_bin(hist_slices, ['Pi', 'Ka', 'Pr'], momentum_bin)
        purity_efficiency = pl.concat([purity_efficiency, compute_purity_efficiency_per_bin(hist_slices, ['Pi', 'Ka', 'Pr'], momentum_bin)])

        ds_graphs = { part: purity_efficiency.filter(((pl.col('x') - momentum_bin).abs() < 0.005) & (pl.col('particle') == part)) for part in ['Pi', 'Ka', 'Pr'] }
        
        outdir = outfile.mkdir(f'purity_efficiency_{momentum_bin}')
        outdir.cd()
        for part in ['Pi', 'Ka', 'Pr']:
            graph = TGraphErrors(1)
            graph.SetPoint(0, ds_graphs[part]['purity'][0], ds_graphs[part]['efficiency'][0]) #ds_graphs[part].sefficiency[0], ds_graphs[part].spurity[0])    
            obj_setter(graph, name=f'{part}_purity_efficiency', title=f'{part} purity vs efficiency;Purity;Efficiency', marker_style=20, marker_size=1, marker_color=4)
            graph.Write(f'{part}_purity_efficiency')


    purity_efficiency.write_csv('/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/purity_efficiency_nsigmaITS.csv')





if __name__ == '__main__':
    output_file = TFile('/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/nsigma_output.root', 'RECREATE')
    outdir = output_file.mkdir('vs_p')
    compute_purity('Pr', 'Ka', 'Pi', '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_nsigma.root', outdir)
    compute_purity('Ka', 'Pi', 'Pr', '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_nsigma.root', outdir)
    compute_purity('Pi', 'Ka', 'Pr', '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_nsigma.root', outdir)
    compute_efficiency('Pr', '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_nsigma.root', outdir)
    compute_efficiency('Pi', '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_nsigma.root', outdir)
    compute_efficiency('Ka', '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_nsigma.root', outdir)

    compute_efficiency_purity('/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_nsigma.root', output_file)