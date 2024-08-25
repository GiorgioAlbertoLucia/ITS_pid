#
#   Draw cluster size distribution for different layers of the ITS
#

import polars as pl
from typing import List
from ROOT import TCanvas, TLegend, TDirectory, TFile

import sys
sys.path.append('..')
from framework.src.hist_handler import HistHandler
from core.dataset import DataHandler
from framework.src.axis_spec import AxisSpec
from framework.utils.root_setter import obj_setter



def draw_clsize_layer_distribution(dataset: pl.DataFrame, outfile: TDirectory):
    """
    Draw cluster size distribution for different layers of the ITS

    Args:
        dataset (pl.DataFrame): dataset
        particles (List[str]): particles to consider
        layers (List[str]): layers to consider
        save (bool, optional): save the plot. Defaults to False.
        path (str, optional): path to save the plot. Defaults to None.
    """

    particles = ['Pi', 'Ka', 'Pr']
    colors = [2, 3, 4]

    hist_handlers = {
        'Pi': HistHandler.createInstance(dataset.filter(pl.col('fPartID') == 2)),
        'Ka': HistHandler.createInstance(dataset.filter(pl.col('fPartID') == 3)),
        'Pr': HistHandler.createInstance(dataset.filter(pl.col('fPartID') == 4))
    }

    for layer in range(7):
        canvas = TCanvas(f'clsize_layer_{layer}', f'Cluster size distribution for layer {layer}', 800, 600)
        hists = []
        max_y = 0
        for iparticle, particle in enumerate(particles):
            axis_spec_x = AxisSpec(16, 0, 16, f'{particle}_L{layer}', '; Cluster size L{layer}; Normalized counts')
            hist = hist_handlers[particle].buildTH1(f'fItsClusterSizeL{layer}', axis_spec_x)
            obj_setter(hist, fill_color=colors[iparticle], line_color=colors[iparticle], fill_alpha=0.5, fill_style=3013)
            hist.Scale(1.0 / hist.Integral())
            tmp_hist = hist.Clone()
            hists.append(tmp_hist)
            max_y = max(max_y, hist.GetMaximum())

        hframe = canvas.DrawFrame(0, 0, 16, max_y * 1.1, f'; Cluster size L{layer}; Normalized counts')
        legend = TLegend(0.7, 0.7, 0.9, 0.9)
        for hist in hists:
            hist.Draw('hist same')
            legend.AddEntry(hist, hist.GetName(), 'f')
            outfile.cd()
            hist.Write()

        legend.Draw('same')
        outfile.cd()
        canvas.Write()  

    outfile.Close()

if __name__ == '__main__':

    input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root']
    cfg_data_file = '../config/config_data.yml'
    output_file = '/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/clsize_layer_study.root'
    tree_name = 'O2clsttable'
    folder_name = 'DF_*' 

    data_handler = DataHandler(input_files, cfg_data_file, tree_name=tree_name, folder_name=folder_name, force_option='AO2D', rigidity_he=True)
    data_handler.dataset.filter(pl.col('fPAbs').is_between(0.95, 1.05))

    output_file_root = TFile(output_file, 'RECREATE')
    outdir = output_file_root.mkdir('clsize_layer_study')

    draw_clsize_layer_distribution(data_handler.dataset, outdir)