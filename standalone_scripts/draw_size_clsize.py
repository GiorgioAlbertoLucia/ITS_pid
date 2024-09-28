from ROOT import TFile, TF1, TCanvas, TGraphErrors, TMultiGraph, kRed, kGreen, kBlue, kOrange, kCyan, TLegend
import polars as pl
import sys
sys.path.append('..')
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from framework.utils.root_setter import obj_setter
from framework.utils.terminal_colors import TerminalColors as tc
from core.dataset import DataHandler

if __name__ == '__main__':

    infiles = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root']
    infiles_he = ['/data/galucia/its_pid/LHC23_pass4_skimmed/LHC23_pass4_skimmed.root']
    cfg_data_file = '../config/config_data.yml'
    output_dir = '../output'
    output_file = output_dir+'/cl_size_slice.root'
    tree_name = 'O2clsttable'
    folder_name = 'DF_*' 

    print(tc.GREEN+tc.BOLD+'Data uploading'+tc.RESET)
    data_handler = DataHandler(infiles, cfg_data_file, tree_name=tree_name, folder_name=folder_name, force_option='AO2D', rigidity_he=True)
    data_handler.dataset.filter(pl.col('fPartID') != 6)
    data_handler_he = DataHandler(infiles_he, cfg_data_file, tree_name=tree_name, folder_name=folder_name, force_option='AO2D', rigidity_he=False)
    data_handler_he.dataset = data_handler_he.dataset.filter(pl.col('fPartID') == 6)
    
    newcolumns = data_handler.dataset.columns
    newcolumns.remove('fItsClusterSize')
    data_handler.dataset = data_handler.dataset[newcolumns]
    data_handler_he.dataset = data_handler_he.dataset[newcolumns]

    print(data_handler.dataset[['fNClustersIts', 'fMeanItsClSize', 'fClSizeCosL', 'fMass']].describe())
    print(data_handler_he.dataset[['fNClustersIts', 'fMeanItsClSize', 'fClSizeCosL', 'fMass']].describe())
    print(data_handler.dataset.dtypes)
    print(data_handler_he.dataset.dtypes)

    for idx in [0, 4, 17]:
        data_handler_he.dataset = data_handler_he.dataset.with_columns(pl.col(data_handler_he.dataset.columns[idx]).cast(pl.Float64))

    print(data_handler.dataset.dtypes)
    print(data_handler_he.dataset.dtypes)
    
    data_handler.dataset = pl.concat([data_handler.dataset[newcolumns], data_handler_he.dataset[newcolumns]])

    hist_handler = HistHandler.createInstance(data_handler.dataset) 
    axis_spec_x = AxisSpec(200, 0, 5, 'All', '; #it{p} (GeV/#it{c}); #LT ITS Cluster size #GT #times #LT cos#lambda #GT; Counts')
    axis_spec_y = AxisSpec(70, 0, 15, 'All', '; #it{p} (GeV/#it{c}); #LT ITS Cluster size #GT #times #LT cos#lambda #GT; Counts')
    h2ClSizeCosL = hist_handler.buildTH2('fPAbs', 'fClSizeCosL', axis_spec_x, axis_spec_y)
    canvas = TCanvas('ch2', 'c', 800, 600)
    h2ClSizeCosL.Draw('colz')
    canvas.SaveAs('/home/galucia/ITS_pid/output/cl_size.pdf')
    
    data_handler.dataset = data_handler.dataset.filter(pl.col('fPAbs').is_between(1.15, 1.25))

    particles = ['Pi', 'Ka', 'Pr', 'De', 'He']
    names = ['#pi', 'K', 'p', 'd', '^{3}He']
    part_ids = [2, 3, 4, 5, 6]
    colors = [kRed, kGreen, kBlue, kOrange, kCyan]
    hists = []
    ymax = 0
    for name, color, part_id, part in zip(names, colors, part_ids, particles):
        axis_spec = AxisSpec(70, 0, 15, name, '; #LT ITS Cluster size #GT #times #LT cos#lambda #GT; Normalised counts')
        ds_part = data_handler.dataset.filter(pl.col('fPartID') == part_id)
        if len(ds_part) == 0:
            continue
        hist_handler = HistHandler.createInstance(ds_part)
        hist = hist_handler.buildTH1('fClSizeCosL', axis_spec)
        hist.Scale(1/hist.Integral())
        ymax = max(ymax, hist.GetMaximum())
        obj_setter(hist, line_color=color, fill_color=color, fill_alpha=0.5, fill_style=3013)
        hists.append(hist)

    canvas = TCanvas('c', 'c', 800, 600)
    hframe = canvas.DrawFrame(0, 0, 10, ymax*1.1, '; #LT ITS Cluster size #GT #times #LT cos#lambda #GT; Normalised counts')
    for hist in hists:
        hist.Draw('hist SAME')

    legend = TLegend(0.7, 0.7, 0.89, 0.89)
    for hist, name in zip(hists, names):
        legend.AddEntry(hist, name, 'f')
    legend.SetBorderSize(0)
    legend.SetNColumns(2)
    legend.Draw('SAME')

    canvas.SaveAs('/home/galucia/ITS_pid/output/cl_size_slice.pdf')



