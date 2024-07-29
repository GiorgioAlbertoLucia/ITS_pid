'''
    Script to determin a Bethe Bloch-like parametrisation for the cluster size distribution
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

lgging = logging.getLogger(__name__)

class BetheBlochParametrisation:

    def __init__(self, data_handler: DataHandler, config_file: str):
        self.data_handler = data_handler
        self._load_config(config_file)
        self.BetheBloch_params = {
            'kp1': -187.9255, 'kp2': -0.26878, 'kp3': 1.16252,
            'kp4': 1.15149, 'kp5': 2.47912
        }
        self.dir = None
        self.sig_points = [[], [], []]  # x, y, y_err
        self.fitted_particle = None

        self.h2 = None
        self.xs = None
        self.yerrs = None

    def _load_config(self, config_file: str) -> None:
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def _update_params(self, params: Dict[str, float]) -> None:
        for key, val in params.items():
            self.BetheBloch_params[key] = val

    def _create_axis_specs(self, cfg_plot: Dict[str, any], particle: str) -> Tuple[AxisSpec, AxisSpec]:
        ''' 
            Create axis specs for x and y variables from the configuration file
        '''
        axis_spec_x = AxisSpec(
            cfg_plot['nXBins'], cfg_plot['xMin'], cfg_plot['xMax'],
            f"{cfg_plot['name']}_{particle}", cfg_plot['title']
        )
        axis_spec_y = AxisSpec(
            cfg_plot['nYBins'], cfg_plot['yMin'], cfg_plot['yMax'],
            f"{cfg_plot['name']}_{particle}_y", f";{cfg_plot['yLabel']};counts"
        )
        return axis_spec_x, axis_spec_y
    
    def _set_output_dir(self, out_dir: TDirectory) -> None:
        self.dir = out_dir
    
    def _filter_data_by_particle(self, particle: str) -> pl.DataFrame:
        particle_id = self.config['species'].index(particle) + 1
        return self.data_handler.dataset.filter(pl.col('fPartID') == particle_id)

    def select_fit_particle(self, particle: str) -> None:
        self.fitted_particle = particle

    ########### Fits

    def generate_graph(self, cfg_label: str) -> None:
        '''
            Save graph to output file
        '''
        cfg = self.config[cfg_label]
        self._update_params(cfg['BBparams'])
        cfg_plot = cfg['plot']
        axis_spec_x, axis_spec_y = self._create_axis_specs(cfg_plot, self.fitted_particle)
        data = self._filter_data_by_particle(self.fitted_particle)

        hist_handler = HistHandler.createInstance(data)
        self.h2 = hist_handler.buildTH2(cfg_plot['xVariable'], cfg_plot['yVariable'], axis_spec_x, axis_spec_y)
        self.sig_points, fm_points, fm_prob = self.extract_fit_points(data, cfg, cfg_plot)

        self.create_and_save_graphs(cfg_plot, self.fitted_particle, fm_points, fm_prob)

    def _configure_fit_function(self, cfg: Dict[str, any], h1: TH1F, ibin: int) -> TF1:
        if ibin < self.h2.GetXaxis().FindBin(cfg['xMaxDoubleFit']):
            fit = TF1('fit', 'gaus + gaus(3)', cfg['yMinFit'], cfg['yMaxFit'])
            fit.SetParameters(h1.GetMaximum(), 5., h1.GetStdDev() / 3, h1.GetMaximum(), 2.1, h1.GetStdDev() / 3)
            fit.SetParLimits(0, h1.GetMaximum() * 0.001, h1.GetMaximum())
            fit.SetParLimits(1, 4., 12.)
            #fit.SetParLimits(3, h1.GetMaximum() * 0.01, h1.GetMaximum())
            fit.SetParLimits(4, 1.98, 2.58)
            #fit.FixParameter(4, 2.18)
            fit.FixParameter(5, 0.64)
        else:
            fit = TF1('fit', 'gaus', cfg['yMinFit'], cfg['yMaxFit'])
            fit.SetParameters(h1.GetMaximum(), h1.GetMean(), h1.GetStdDev())
        return fit

    def _fit_histogram_slice(self, data: pl.DataFrame, cfg_plot: Dict[str, any], cfg: Dict[str, any], ibin: int) -> Tuple[pl.DataFrame, TH1F, TF1, bool]:
        data_slice = data.filter((pl.col(cfg_plot['xVariable']) > self.h2.GetXaxis().GetBinLowEdge(ibin)) & (pl.col(cfg_plot['xVariable']) < self.h2.GetXaxis().GetBinUpEdge(ibin)))
        hist_handler_slice = HistHandler.createInstance(data_slice)
        h1 = hist_handler_slice.buildTH1(cfg_plot['yVariable'], self._create_axis_specs(cfg_plot, 'slice')[1])
        fit = self._configure_fit_function(cfg, h1, ibin)
        fit_status = h1.Fit(fit, 'RQSL')
        if ibin < self.h2.GetXaxis().FindBin(cfg['xMaxDoubleFit']):
            logging.info(f"{fit.GetChisquare():.2f} / {fit.GetNDF()}\t{fit.GetParameter(1):.2f}\t{fit.GetParameter(2):.2f}\t{fit.GetParameter(4):.2f}\t{fit.GetParameter(5):.2f}")
        else:
            logging.info(f"{fit.GetChisquare():.2f} / {fit.GetNDF()}\t{fit.GetParameter(1):.2f}\t{fit.GetParameter(2):.2f}")  
        return data_slice, h1, fit_status, fit  

    def _get_fit_bins(self, cfg: Dict[str, any]) -> Tuple[int, int, int]:
        first_bin = self.h2.GetXaxis().FindBin(cfg['xMinFit'])
        last_bin = self.h2.GetXaxis().FindBin(cfg['xMaxFit'])
        last_bin_double_fit = self.h2.GetXaxis().FindBin(cfg['xMaxDoubleFit'])
        return first_bin, last_bin, last_bin_double_fit 

    def extract_fit_points(self, data: pl.DataFrame, cfg: Dict[str, any], cfg_plot: Dict[str, any]) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        sig_points = [[], [], []]
        fm_points = [[], [], []]
        fm_prob = [[], [], []]

        first_bin, last_bin, last_bin_double_fit = self._get_fit_bins(cfg)
        logging.info(f"Fit results:")
        logging.info(f"chi2/ndf:\tmean:\tsigma:\tmean_fm:\tsigma_fm:")
        for ibin in range(first_bin, last_bin + 1):
            data_slice, h1, fit_status, fit = self._fit_histogram_slice(data, cfg_plot, cfg, ibin)
            if not fit_status.IsValid():
                logging.warning(f"Fit failed for bin range [{self.h2.GetXaxis().GetBinLowEdge(ibin)}, {self.h2.GetXaxis().GetBinUpEdge(ibin)}]")
                continue
            self._store_fit_results(sig_points, fm_points, fm_prob, fit, h1, ibin, last_bin_double_fit, cfg)
            self._save_fit_plot(h1, fit, ibin, last_bin_double_fit)
            del h1, data_slice
        return sig_points, fm_points, fm_prob

    def _store_fit_results(self, sig_points: List[List[float]], fm_points: List[List[float]], fm_prob: List[List[float]], fit: TF1, h1: TH1F, ibin: int, last_bin_double_fit: int, cfg: Dict[str, any]) -> None:
        sig_points[0].append(self.h2.GetXaxis().GetBinCenter(ibin))
        sig_points[1].append(fit.GetParameter(1))
        sig_points[2].append(fit.GetParameter(2))
        if ibin < last_bin_double_fit:
            fm_points[0].append(self.h2.GetXaxis().GetBinCenter(ibin))
            fm_points[1].append(fit.GetParameter(4))
            fm_points[2].append(fit.GetParameter(5))
            sig, fm = self.create_signal_and_fake_matching_functions(cfg, fit)
            fm_frac = self.calculate_fake_matching_fraction(cfg, sig, fm)
            fm_prob[0].append(self.h2.GetXaxis().GetBinCenter(ibin))
            fm_prob[1].append(fm_frac)
            fm_prob[2].append(0.01)

    def _save_fit_plot(self, h1: TH1F, fit: TF1, ibin: int, last_bin_double_fit: int) -> None:
        canvas = TCanvas()
        h1.SetTitle(f"fit {self.h2.GetXaxis().GetBinCenter(ibin)}")
        h1.Draw("hist e0")
        fit.SetLineColor(kRed)
        fit.Draw("same")
        if ibin < last_bin_double_fit:
            sig, fm = self.create_signal_and_fake_matching_functions(self.config['p'], fit)
            sig.SetLineColor(kOrange)
            sig.Draw("same")
            fm.SetLineColor(kGreen)
            fm.Draw("same")
        self.dir.cd()
        canvas.SetTitle(f"fit_{self.h2.GetXaxis().GetBinCenter(ibin)}")
        canvas.BuildLegend()
        canvas.Write(f"fit_{self.h2.GetXaxis().GetBinCenter(ibin)}", )

    ########### Fake matching

    def create_signal_and_fake_matching_functions(self, cfg: Dict[str, any], fit: TF1) -> Tuple[TF1, TF1]:
        sig = TF1('sig', 'gaus', cfg['yMinFit'], cfg['yMaxFit'])
        sig.SetParameters(fit.GetParameter(0), fit.GetParameter(1), fit.GetParameter(2))
        fm = TF1('fm', 'gaus', cfg['yMinFit'], cfg['yMaxFit'])
        fm.SetParameters(fit.GetParameter(3), fit.GetParameter(4), fit.GetParameter(5))
        return sig, fm

    def calculate_fake_matching_fraction(self, cfg: Dict[str, any], sig: TF1, fm: TF1) -> float:
        sig_center = sig.GetParameter(1)
        sig_sigma = sig.GetParameter(2)
        sig_frac = sig.Integral(sig_center - 2 * sig_sigma, sig_center + 2 * sig_sigma)
        fm_frac = fm.Integral(sig_center - 2 * sig_sigma, sig_center + 2 * sig_sigma)
        return fm_frac / (sig_frac + fm_frac)

    def create_and_save_graphs(self, cfg_plot: Dict[str, any], particle: str, fm_points: List[List[float]], fm_prob: List[List[float]]) -> None:
        sig_points_graph = TGraphErrors(len(self.sig_points[0]), np.array(self.sig_points[0]), np.array(self.sig_points[1]), np.zeros(len(self.sig_points[0])), np.array(self.sig_points[2]))
        obj_setter(sig_points_graph, name=f'{cfg_plot["name"]}_{particle}_sig_points', title=f'{cfg_plot["name"]}_{particle}_sig_points; {cfg_plot["xLabel"]}; {cfg_plot["yLabel"]}', marker_color=kRed, marker_style=20, marker_size=1)
        sig_sigma_points_graph = TGraphErrors(len(self.sig_points[0]), np.array(self.sig_points[0]), np.array(self.sig_points[2]), np.zeros(len(self.sig_points[0])), np.zeros(len(self.sig_points[0])))
        obj_setter(sig_sigma_points_graph, name=f'{cfg_plot["name"]}_{particle}_sig_sigma_points', title=f'{cfg_plot["name"]}_{particle}_sig_sigma_points; {cfg_plot["xLabel"]}; sigma', marker_color=kBlue, marker_style=20, marker_size=1)
        self.dir.cd()
        sig_points_graph.Write(f"{cfg_plot['name']}_{particle}_sig_points")
        sig_sigma_points_graph.Write(f"{cfg_plot['name']}_{particle}_sig_sigma_points")

        if fm_points[0]:
            fm_points_graph = TGraphErrors(len(fm_points[0]), np.array(fm_points[0]), np.array(fm_points[1]), np.zeros(len(fm_points[0])), np.array(fm_points[2]))
            obj_setter(fm_points_graph, name=f'{cfg_plot["name"]}_{particle}_fm_points', title=f'{cfg_plot["name"]}_{particle}_fm_points; {cfg_plot["xLabel"]}; {cfg_plot["yLabel"]}', marker_color=kGreen, marker_style=20, marker_size=1)
            self.dir.cd()
            fm_points_graph.Write(f"{cfg_plot['name']}_{particle}_fm_points")

            fm_prob_graph = TGraphErrors(len(fm_prob[0]), np.array(fm_prob[0]), np.array(fm_prob[1]), np.zeros(len(fm_prob[0])), np.array(fm_prob[2]))
            obj_setter(fm_prob_graph, name=f'{cfg_plot["name"]}_{particle}_fm_prob_points', title=f'{cfg_plot["name"]}_{particle}_fm_prob_points; {cfg_plot["xLabel"]}; fake matching probability', marker_color=kOrange, marker_style=20, marker_size=1)
            self.dir.cd()   
            fm_prob_graph.Write(f"{cfg_plot['name']}_{particle}_fm_prob_points")

    ########### Bethe-Bloch
          
    def _update_bethe_bloch_params(self, fit: TF1) -> None:
        logging.info("Updated Bethe-Bloch parameters:")
        for i, key in enumerate(self.BetheBloch_params.keys()):
            self.BetheBloch_params[key] = fit.GetParameter(key)
            logging.info(f"{key}: {fit.GetParameter(key)}")

    def fit_bethe_bloch(self, cfg_label: str) -> None:

        for key, val in self.config[cfg_label]['BBparams'].items():
            self.BetheBloch_params[key] = val
            logging.info(f"{key}: {val}")
        
        bethe_bloch_fit = TF1('bethe_bloch', BetheBloch, self.config[cfg_label]['xMinFit'], self.config[cfg_label]['xMaxFit'], 5)
        bethe_bloch_fit.SetParameters(self.BetheBloch_params['kp1'], self.BetheBloch_params['kp2'], self.BetheBloch_params['kp3'], self.BetheBloch_params['kp4'], self.BetheBloch_params['kp5'])
        bethe_bloch_fit.SetParNames("kp1", "kp2", "kp3", "kp4", "kp5")

        mg = TMultiGraph()
        sig_points_graph = TGraphErrors(len(self.sig_points[0]), np.array(self.sig_points[0]), np.array(self.sig_points[1]), np.zeros(len(self.sig_points[0])), np.array(self.sig_points[2]))
        sig_points_graph.Fit(bethe_bloch_fit, 'RSM+')
        self._update_bethe_bloch_params(bethe_bloch_fit)

        self._plot_bethe_bloch_fit(sig_points_graph, bethe_bloch_fit, mg, self.fitted_particle)

        del bethe_bloch_fit, sig_points_graph, mg

    def _plot_bethe_bloch_fit(self, sig_points_graph: TGraphErrors, fit: TF1, mg: TMultiGraph, particle: str) -> None:
        sig_points_graph.SetTitle(f"{particle} Bethe-Bloch Fit")
        sig_points_graph.SetMarkerStyle(20)
        sig_points_graph.SetMarkerSize(1)
        sig_points_graph.SetMarkerColor(kRed)
        mg.Add(sig_points_graph)

        canvas = TCanvas(f"cBB_{particle}", f"Bethe-Bloch fit for {particle}")
        mg.Draw("AP")
        fit.Draw("same")
        self.dir.cd()
        mg.Write(f"{particle}_BB_fit")

        h2_canvas = TCanvas(f"h2_cBB_{particle}", f"Bethe-Bloch fit for {particle}")
        self.h2.Draw()
        fit.SetRange(self.h2.GetXaxis().GetXmin(), self.h2.GetXaxis().GetXmax())
        fit.Draw("same")
        self.dir.cd()
        h2_canvas.Write()

        del canvas, h2_canvas
    
    ########### NSigma distribution

    def _BBfunc(self, x: float) -> float:
        logging.info(f"BBfunc: {self.BetheBloch_params}")
        x = np.abs(x)
        beta = x / np.sqrt(1 + x**2)
        aa = beta**self.BetheBloch_params['kp4']
        bb = (1/x)**self.BetheBloch_params['kp5']
        bb = np.log(self.BetheBloch_params['kp3'] + bb)
        return (self.BetheBloch_params['kp2'] - aa - bb) * self.BetheBloch_params['kp1'] / aa

    def _get_sigma(self, x: float) -> float:
        idx = np.abs(self.xs - np.abs(x)).argmin()
        return self.yerrs[idx]

    def _create_nsigma_distribution(self, cfg_label: str, particle: str) -> None:
        cfg_plot = self.config[cfg_label]['plot']
        col_name_exp = f"fExp{cfg_plot['yVariable'][1:]}"
        logging.info(f"Bethe Bloch function: {self.BetheBloch_params}")

        self.xs = np.array(self.sig_points[0], dtype=np.float32)
        self.yerrs = np.array(self.sig_points[2], dtype=np.float32)
        self.data_handler.dataset = self.data_handler.dataset.with_columns(self._BBfunc(pl.col(cfg_plot['xVariable'])).alias(col_name_exp))
        
        step = self.sig_points[0][1] - self.sig_points[0][0]
        xs = self.sig_points[0]
        xs[0] = self.data_handler.dataset[cfg_plot['xVariable']].min()
        xs[-1] = self.data_handler.dataset[cfg_plot['xVariable']].max()
        logging.info(f"xs: {xs}")
        logging.info(f"sigma: {self.sig_points[2]}")
        err_dict = {err: (pl.col(cfg_plot['xVariable']) > x-step).and_(pl.col(cfg_plot['xVariable']) < x+step) for err, x in zip(self.sig_points[2], xs)}
        #self.data_handler.dataset = self.data_handler.dataset.with_columns(pl.coalesce(pl.when(cond).then(val) for val, cond in err_dict.items()).alias(f"fSigma{cfg_plot['yVariable'][1:]}"))
        self.data_handler.dataset = self.data_handler.dataset.with_columns(pl.Series(np.ones(len(self.data_handler.dataset), dtype=np.float32)).alias(f"fSigma{cfg_plot['yVariable'][1:]}"))
        self.data_handler.dataset = self.data_handler.dataset.with_columns(((pl.col(cfg_plot['yVariable']) - pl.col(col_name_exp)) / pl.col(f"fSigma{cfg_plot['yVariable'][1:]}")).alias(f"fNSigma{self.fitted_particle}"))

    
    def draw_nsigma_distribution(self, cfg_label: str, particle: str) -> None:
        cfg_plot = self.config[cfg_label]['plot_nsigma']
        self._create_nsigma_distribution(cfg_label, particle)
        data = self._filter_data_by_particle(particle)
        axis_spec_x, axis_spec_y = self._create_axis_specs(cfg_plot, particle)
        sigma_col = f"fSigma{self.config[cfg_label]['plot']['yVariable'][1:]}"
        logging.info(f"Selected data\n{data.columns}\n{data[[cfg_plot['xVariable'], cfg_plot['yVariable'], sigma_col, self.config[cfg_label]['plot']['yVariable']]].describe()}")
        hist_handler = HistHandler.createInstance(data)
        hist = hist_handler.buildTH2(cfg_plot['xVariable'], cfg_plot['yVariable'], axis_spec_x, axis_spec_y)
        self.dir.cd()
        hist.Write(f"{cfg_plot['name']}_{particle}_nsigma")
        del hist, hist_handler, data

    def draw_expected_values(self, cfg_label: str) -> None:

        cfg = self.config[cfg_label]
        cfg_plot = self.config[cfg_label]['plot']
        col_name_exp = f"fExp{cfg_plot['yVariable'][1:]}"
        axis_spec_x, axis_spec_y = self._create_axis_specs(cfg_plot, 'Pr')
        hist_handler = HistHandler.createInstance(self.data_handler.dataset)
        hist = hist_handler.buildTH2(cfg_plot['xVariable'], col_name_exp, axis_spec_x, axis_spec_y)
        self.dir.cd()
        obj_setter(hist, name=f"{cfg_plot['name']}_expected", title=f"{cfg_plot['name']}_expected; {cfg_plot['xLabel']}; {cfg_plot['yLabel']}") 
        hist.Write(f"{cfg_plot['name']}_expected")
        del hist, hist_handler

    def _compute_purity_efficiency_in_slice(self, data: Dict[str, Dict[str, pl.DataFrame]], xlow: float, xup: float, nsigma_cut: float, cfg_label: str) -> Tuple[float, float, float, float]:
        ''''
            Compute purity and efficiency in a slice of the Bethe-Bloch curve.

            Parameters:
            data: Dict[str, Dict[str, pl.DataFrame]] - dictionary of datasets -> [predicted_label][true_label]
            xlow: float - lower bound of the slice
            xup: float - upper bound of the slice
            nsigma_cut: float - cut on the nsigma distribution

            Returns:
            purity: float - purity of the slice 
            purity_err: float - error on the purity
            efficiency: float - efficiency of the slice
            eff_err: float - error on the efficiency
        '''

        cfg = self.config[cfg_label]['plot']

        true_pos = len(data[self.fitted_particle][self.fitted_particle].filter((np.abs(pl.col(cfg['xVariable'])) > xlow) & (np.abs(pl.col(cfg['xVariable'])) < xup)))
        true_neg = len(data['all']['all'].filter((np.abs(pl.col(cfg['xVariable'])) > xlow) & (np.abs(pl.col(cfg['xVariable'])) < xup)))
        false_pos = len(data[self.fitted_particle]['all'].filter((np.abs(pl.col(cfg['xVariable'])) > xlow) & (np.abs(pl.col(cfg['xVariable'])) < xup)))
        false_neg = len(data['all'][self.fitted_particle].filter((np.abs(pl.col(cfg['xVariable'])) > xlow) & (np.abs(pl.col(cfg['xVariable'])) < xup)))

        purity = 0 if (true_pos+false_pos) == 0 else true_pos/(true_pos+false_pos)
        efficiency = 0 if (true_pos+false_neg) == 0 else true_pos/(true_pos+false_neg)
        purity_err = 0 if (true_pos+false_pos) == 0 else np.sqrt(purity*(1-purity)/(true_pos+false_pos))
        eff_err = 0 if (true_pos+false_neg) == 0 else np.sqrt(efficiency*(1-efficiency)/(true_pos+false_neg))

        return purity, purity_err, efficiency, eff_err

    def purity_efficiency(self, cfg_label:str, nsigma_cut:float, **kwargs) -> None:

        cfg_plot = self.config[cfg_label]['plot']

        step = (cfg_plot['xMax'] - cfg_plot['xMin'])/cfg_plot['nXBins']

        dss = {
            'all': self.data_handler.dataset.filter(pl.col('fPartID') != self.config['species'].index('Pr')+1),
            self.fitted_particle: self.data_handler.dataset.filter(pl.col('fPartID') == self.config['species'].index(self.fitted_particle)+1)
        }

        dss_pred = {
            'all': {key:ds.filter(np.abs(pl.col(f'fNSigma{self.fitted_particle}')) > nsigma_cut) for key, ds in dss.items()},
            self.fitted_particle: {key:ds.filter(np.abs(pl.col(f'fNSigma{self.fitted_particle}')) < nsigma_cut) for key, ds in dss.items()}
        }

        xs, purity, purity_err, efficiency, eff_err = [], [], [], [], []
        for ix in np.arange(cfg_plot['xMin'], cfg_plot['xMax'], step):
            p, p_err, e, e_err = self._compute_purity_efficiency_in_slice(dss_pred, ix, ix+step, nsigma_cut, cfg_label)
            xs.append(ix+step/2)
            purity.append(p)
            purity_err.append(p_err)
            efficiency.append(e)
            eff_err.append(e_err)

        eff_pur = TMultiGraph(f'eff_pur_{cfg_label}', f'Efficiency & Purity {self.fitted_particle}; {cfg_plot["xLabel"]}; Purity and Efficiency')
        g_purity = TGraphErrors(len(xs), np.array(xs, dtype=np.float32), np.array(purity, dtype=np.float32), np.zeros(len(xs), dtype=np.float32), np.array(purity_err, dtype=np.float32))
        obj_setter(g_purity, name=f'purity_{cfg_label}', title=f'Purity {self.fitted_particle}; {cfg_plot["xLabel"]}; Purity', marker_color=kCyan-3, marker_style=20, marker_size=1)
        g_efficiency = TGraphErrors(len(xs), np.array(xs, dtype=np.float32), np.array(efficiency, dtype=np.float32), np.zeros(len(xs), dtype=np.float32), np.array(eff_err, dtype=np.float32))
        obj_setter(g_efficiency, name=f'efficiency_{cfg_label}', title=f'Efficiency {self.fitted_particle}; {cfg_plot["xLabel"]}; Efficiency', marker_color=kOrange, marker_style=20, marker_size=1)

        eff_pur.Add(g_purity)
        eff_pur.Add(g_efficiency)

        canvas = TCanvas(f'c_eff_pur_{cfg_label}', f'Efficiency & Purity {self.fitted_particle}')
        canvas.cd()
        eff_pur.Draw('AP')
        canvas.BuildLegend()
        self.dir.cd()
        canvas.Write()



    ########### General
    
    @timeit
    def run_all(self, cfg_label, output_dir) -> None:

        self._set_output_dir(output_dir)
        
        self.select_fit_particle('Pr')
        self.generate_graph(cfg_label)
        self.fit_bethe_bloch(cfg_label)
        for part in self.config['species']:
            self.draw_nsigma_distribution(cfg_label, part)
        self.draw_expected_values(cfg_label)
        self.purity_efficiency(cfg_label, 2)
            

if __name__ == '__main__':

    # Configure logging
    os.remove("output.log")
    logging.basicConfig(filename="output.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(filename="output.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    input_files = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root']
    #input_files = ['/Users/glucia/Projects/ALICE/data/its_pid/LHC22o_pass7_minBias_small.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = 'bethe_bloch_parametrisation.yml'
    output_dir = '../output/LHC22o_pass6_minBias_slice'
    output_file = output_dir+'/bethe_bloch_parametrisation.root'
    tree_name = 'O2clsttable'
    folder_name = 'DF_*' 

    data_handler = DataHandler(input_files, cfg_data_file, tree_name=tree_name, folder_name=folder_name, force_option='AO2D')

    outFile = TFile(output_file, 'RECREATE')
    bb_param = BetheBlochParametrisation(data_handler, cfg_output_file)

    dir_p = outFile.mkdir('clsize_vs_p')
    bb_param.run_all('p', dir_p)
    
    outFile.Close()