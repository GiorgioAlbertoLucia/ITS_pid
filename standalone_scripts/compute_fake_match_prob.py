import polars as pl
import numpy as np

from ROOT import TF1, TFile, TCanvas, TLine

import sys
sys.path.append('..')
from framework.utils.terminal_colors import TerminalColors as tc
from framework.src.graph_handler import GraphHandler

def compute_fake_match_prob(infile_path:str, outfile_path:str):


    data = pl.read_csv(infile_path)
    data = data.filter(data['fm_norm'] > 0)
    outfile = TFile(outfile_path, 'RECREATE')

    prob = pl.DataFrame({'x': pl.Series(values=[], dtype=pl.Float64),
                         'sx': pl.Series(values=[], dtype=pl.Float64),
                         'prob': pl.Series(values=[], dtype=pl.Float64),
                         'sprob': pl.Series(values=[], dtype=pl.Float64)})

    for idx in range(len(data)):

        signal = TF1('signal', 'gaus', 0, 15)
        signal.SetParameters(data['signal_norm'][idx], data['signal_mean'][idx], data['signal_sigma'][idx])
        signal.SetLineColor(2)
        fake = TF1('fake', 'gaus', 0, 15)
        fake.SetParameters(data['fm_norm'][idx], data['fm_mean'][idx], data['fm_sigma'][idx])
        fake.SetLineColor(4)

        low_edge = data['signal_mean'][idx] - 2*data['signal_sigma'][idx]
        high_edge = data['signal_mean'][idx] + 2*data['signal_sigma'][idx]

        low_line = TLine(low_edge, 0, low_edge, data['signal_norm'][idx])
        high_line = TLine(high_edge, 0, high_edge, data['signal_norm'][idx])

        sig_int = signal.Integral(low_edge, high_edge)
        fake_int = fake.Integral(low_edge, high_edge)
        fake_frac = fake_int/(sig_int+fake_int)
        print(tc.RED+'x: '+tc.RESET, data['x'][idx], tc.RED+'signal_int: '+tc.RESET, sig_int, tc.RED+'fake_int: '+tc.RESET, fake_int, tc.RED+'prob: '+tc.RESET, fake_int/sig_int)
        prob = pl.concat([prob, pl.DataFrame({'x': pl.Series(values=[data['x'][idx]], dtype=pl.Float64),
                                              'sx': pl.Series(values=[data['sx'][idx]], dtype=pl.Float64),
                                              'prob': pl.Series(values=[fake_frac], dtype=pl.Float64),
                                              'sprob': pl.Series(values=[np.sqrt(fake_frac*(1-fake_frac)/(sig_int+fake_int))], dtype=pl.Float64)
                                              })])

        c = TCanvas(f'c_{data["x"][idx]}', f'c_{data["x"][idx]}', 800, 600)
        signal.Draw()
        fake.Draw('same')
        low_line.Draw()
        high_line.Draw()
        c.BuildLegend()
        outfile.cd()
        c.Write()

    graph_handler = GraphHandler(prob)
    graph = graph_handler.createTGraphErrors('x', 'prob', 'sx', 'sprob')
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1)
    graph.SetMarkerColor(4) 
    outfile.cd()
    graph.Write('fake_match_prob')


if __name__ == '__main__':
    compute_fake_match_prob('/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/fit_rsults_Pr.csv', '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/fake_match_prob.root')