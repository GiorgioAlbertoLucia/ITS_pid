'''
    Script to determin a Bethe Bloch-like parametrisation for the cluster size distribution
'''

import os
import yaml
import numpy as np
import polars as pl
from ROOT import TFile, TH1F, TH2F, TF1, TCanvas, gInterpreter, TObjArray, TDirectory, TGraphErrors, TMultiGraph
#from ROOT import RooRealVar, RooDataHist, RooPlot, RooGaussian, RooAddPdf, RooGenericPdf, RooAbsData
from ROOT import kRed, kGreen, kBlue, kOrange

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BETHEBLOCH_DIR = os.path.join(CURRENT_DIR, '..', 'include', 'BetheBloch.hh')
gInterpreter.ProcessLine(f'#include "{BETHEBLOCH_DIR}"')
from ROOT import BetheBloch

import sys
sys.path.append('..')
from framework.src.axis_spec import AxisSpec
from framework.src.hist_handler import HistHandler
from framework.utils.terminal_colors import TerminalColors as tc
from core.dataset import DataHandler

class BetheBlochParametrisation:

    def __init__(self, data_handler:DataHandler, config_file):
        self.data_handler = data_handler
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        self.BetheBloch_params = {'kp1':  -187.9255,
                                 'kp2':  -0.26878,
                                 'kp3':  1.16252,
                                 'kp4':  1.15149,
                                 'kp5':  2.47912,
                                 #'resolution':  0.09
                }
        self.dir = None

        self.sig_points = [[], [], []] # x, y, y_err

        self.fitted_particle = None

    def setOutputDir(self, out_dir:TDirectory) -> None:
        self.dir = out_dir

    def generateGraph(self, cfg_label:str, particle:str='Pr') -> None:

        cfg = self.config[cfg_label]
        for key, val in cfg['BBparams'].items():
            self.BetheBloch_params[key] = val

        cfg_plot = self.config[cfg_label]['plot']
        print(cfg_plot)
        
        axis_spec_x = AxisSpec(cfg_plot['nXBins'], cfg_plot['xMin'], cfg_plot['xMax'], cfg_plot['name']+f'_{particle}', cfg_plot['title'])
        axis_spec_y = AxisSpec(cfg_plot['nYBins'], cfg_plot['yMin'], cfg_plot['yMax'], cfg_plot['name']+f'_{particle}'+'_y', f';{cfg_plot["yLabel"]};counts')

        data = self.data_handler.dataset.filter(pl.col('fPartID') == self.config['species'].index(particle)+1)
        hist_handler = HistHandler.createInstance(data)
        self.h2 = hist_handler.buildTH2(cfg_plot['xVariable'], cfg_plot['yVariable'], axis_spec_x, axis_spec_y)
        self.h2.RebinX(2)

        self.sig_points = [[], [], []] # x, y, y_err
        fm_points = [[], [], []]  # x, y, y_err
        fm_prob = [[], [], []]        # x, y

        first_bin = self.h2.GetXaxis().FindBin(cfg['xMinFit'])
        last_bin = self.h2.GetXaxis().FindBin(cfg['xMaxFit'])
        last_bin_double_fit = self.h2.GetXaxis().FindBin(cfg['xMaxDoubleFit'])

        for ibin in range(first_bin, last_bin+1):
            
            data_slice = data.filter((pl.col(cfg_plot['xVariable']) > self.h2.GetXaxis().GetBinLowEdge(ibin)) & (pl.col(cfg_plot['xVariable']) < self.h2.GetXaxis().GetBinUpEdge(ibin)))
            hist_handler_slice = HistHandler.createInstance(data_slice)
            h1 = hist_handler_slice.buildTH1(cfg_plot['yVariable'], axis_spec_y)

            fit = TF1('fit', 'gaus', cfg['yMinFit'], cfg['yMaxFit'])
            fit.SetParameters(h1.GetMaximum(), h1.GetMean(), h1.GetStdDev())
            if ibin <= last_bin_double_fit:
                fit = TF1('fit', 'gaus + gaus(3)', cfg['yMinFit'], cfg['yMaxFit'])
                fit.SetParameters(h1.GetMaximum(), h1.GetMean(), h1.GetStdDev(), h1.GetMaximum()/10, h1.GetMean()/3., h1.GetStdDev()/3)
                fit.SetParLimits(3, h1.GetMaximum()*0.01, h1.GetMaximum())
                fit.SetParLimits(4, 0.2, h1.GetMean()-1*h1.GetStdDev())
                fit.SetParLimits(5, 0.0001, h1.GetStdDev())
            fit_status = h1.Fit(fit, 'RQSL')
            if not fit_status.IsValid():
                print(tc.BOLD+'range ='+tc.RESET, f'[{self.h2.GetXaxis().GetBinLowEdge(ibin)}, {self.h2.GetXaxis().GetBinUpEdge(ibin)}]')
                print(tc.RED+'Fit failed'+tc.RESET)
                continue

            print(tc.BOLD+'range ='+tc.RESET, f'[{self.h2.GetXaxis().GetBinLowEdge(ibin)}, {self.h2.GetXaxis().GetBinUpEdge(ibin)}]')
            print(tc.BOLD+'chi2 ='+tc.RESET, fit.GetChisquare())
            if ibin <= last_bin_double_fit:
                print(tc.BOLD+'NDF ='+tc.RESET, h1.GetNbinsX()-6)   
            else:    
                print(tc.BOLD+'NDF ='+tc.RESET, h1.GetNbinsX()-3)
            print(tc.BOLD+'chi2/NDF ='+tc.RESET, fit.GetChisquare()/(h1.GetNbinsX()-3))
            if ibin <= last_bin_double_fit:
                print(tc.BOLD+'mean fm ='+tc.RESET, fit.GetParameter(4))
                print(tc.BOLD+'sigma fm ='+tc.RESET, fit.GetParameter(5))
                print(tc.BOLD+'fake matching probability: '+tc.RESET, f'{fit.GetParameter(5)} ± {fit.GetParError(5)}')
            print()

            self.sig_points[0].append(self.h2.GetXaxis().GetBinCenter(ibin))
            self.sig_points[1].append(fit.GetParameter(1))
            self.sig_points[2].append(fit.GetParameter(2))
            if ibin <= last_bin_double_fit:
                fm_points[0].append(self.h2.GetXaxis().GetBinCenter(ibin))
                fm_points[1].append(fit.GetParameter(4))
                fm_points[2].append(fit.GetParameter(5))

                sig = TF1('sig', 'gaus', cfg['yMinFit'], cfg['yMaxFit'])
                sig.SetParameters(fit.GetParameter(0), fit.GetParameter(1), fit.GetParameter(2))
                fm = TF1('sig', 'gaus', cfg['yMinFit'], cfg['yMaxFit'])
                fm.SetParameters(fit.GetParameter(3), fit.GetParameter(4), fit.GetParameter(5))
                fm_frac = fm.Integral(cfg['yMinFit'], cfg['yMaxFit'])/(sig.Integral(cfg['yMinFit'], cfg['yMaxFit'])+fm.Integral(cfg['yMinFit'], cfg['yMaxFit']))
                fm_prob[0].append(self.h2.GetXaxis().GetBinCenter(ibin))
                fm_prob[1].append(fm_frac)
                fm_prob[2].append(np.sqrt(fm_frac*(1-fm_frac)/(sig.Integral(cfg['yMinFit'], cfg['yMaxFit'])+fm.Integral(cfg['yMinFit'], cfg['yMaxFit']))))

            if ibin <= last_bin_double_fit:
                    
                canvas = TCanvas(f'canvas_{ibin}', 'canvas')
                canvas.cd()
                h1.Draw('hist')
                sig.SetLineColor(kRed)
                sig.Draw('same')
                fm.SetLineColor(kGreen)
                fm.Draw('same')
                fit.SetLineColor(kBlue)
                fit.Draw('same')
                self.dir.cd()
                canvas.Write()

            del h1

        g_all = TMultiGraph(f'{cfg_plot["name"]}_all_{particle}', f'{cfg_plot["name"]}_all_{particle}; {cfg_plot["xLabel"]}; {cfg_plot["yLabel"]}')
        
        g_sig = TGraphErrors(len(self.sig_points[0]), np.array(self.sig_points[0], dtype=np.float32), np.array(self.sig_points[1], dtype=np.float32), np.zeros(len(self.sig_points[0]), dtype=np.float32), np.array(self.sig_points[2], dtype=np.float32))
        g_sig.SetName(f'{cfg_plot["name"]}_sig_{particle}')
        g_sig.SetTitle(f'{cfg_plot["name"]}_sig_{particle}; {cfg_plot["xLabel"]}; {cfg_plot["yLabel"]}')
        g_sig.SetMarkerColor(kBlue)
        g_fm = TGraphErrors(len(fm_points[0]), np.array(fm_points[0], dtype=np.float32), np.array(fm_points[1], dtype=np.float32), np.zeros(len(fm_points[0]), dtype=np.float32), np.array(fm_points[2], dtype=np.float32))
        g_fm.SetName(f'{cfg_plot["name"]}_fm_{particle}')
        g_fm.SetTitle(f'{cfg_plot["name"]}_fm_{particle}; {cfg_plot["xLabel"]}; {cfg_plot["yLabel"]}')
        g_fm.SetMarkerColor(kOrange)
        g_prob = TGraphErrors(len(fm_prob[0]), np.array(fm_prob[0], dtype=np.float32), np.array(fm_prob[1], dtype=np.float32), np.zeros(len(fm_prob[0]), dtype=np.float32), np.array(fm_prob[2], dtype=np.float32))
        g_prob.SetName(f'{cfg_plot["name"]}_prob_{particle}')
        g_prob.SetTitle(f'{cfg_plot["name"]}_prob_{particle}; {cfg_plot["xLabel"]}; fake matching probability')

        g_all.Add(g_sig)
        g_all.Add(g_fm)

        self.dir.cd()
        self.h2.Write()
        g_all.Write()
        g_sig.Write()
        g_fm.Write()
        g_prob.Write()

    def fitBetheBloch(self, cfg_label:str, particle:str='Pr') -> None:

        self.fitted_particle = particle

        cfg = self.config[cfg_label]
        for key, val in cfg['BBparams'].items():
            self.BetheBloch_params[key] = val

        cfg_plot = self.config[cfg_label]['plot']

        self.BBcurve = TF1(f'BetheBloch', BetheBloch, cfg_plot['xMin'], cfg_plot['xMax'], len(self.BetheBloch_params.values()))

        for i, (parName, param) in enumerate(self.BetheBloch_params.items()):    
            self.BBcurve.SetParameter(i, param)
            self.BBcurve.SetParName(i, parName)
        
        xs = np.array(self.sig_points[0], dtype=np.float32)
        ys = np.array(self.sig_points[1], dtype=np.float32)
        yerrs = np.array(self.sig_points[2], dtype=np.float32)
        self.hBB = TH1F(f'BetheBloch_{particle}_{cfg_label}', f'Bethe Bloch curve - {particle} - {cfg_label}; {cfg_plot["xLabel"]}; {cfg_plot["yLabel"]}', len(xs), xs[0], xs[-1])
        for ix in range(len(xs)):
            self.hBB.SetBinContent(ix+1, ys[ix])
            self.hBB.SetBinError(ix+1, 0.09*ys[ix]) # assume 9% resolution
        self.hBB.Fit(self.BBcurve, 'RM+')

        self.dir.cd()
    
        print(tc.BOLD+'Bethe Bloch parameters:'+tc.RESET)
        print(tc.GREEN+particle+tc.RESET)
        print(tc.GREEN+cfg_label+tc.RESET)
        for ipar, (parName, param) in enumerate(self.BetheBloch_params.items()):    
            self.BetheBloch_params[parName] = self.BBcurve.GetParameter(ipar)
            print(tc.RED+f'     {parName}:\t'+tc.RESET+f'{self.BBcurve.GetParameter(ipar)}')
        print(tc.BOLD+'     chi2/ndf:\t'+tc.RESET+str(self.BBcurve.GetChisquare())+'/'+str(self.BBcurve.GetNDF()))

        for ipar, (parName, param) in enumerate(self.BetheBloch_params.items()):    
            self.BetheBloch_params[parName] = self.BBcurve.GetParameter(ipar)

    def drawBetheBloch(self, cfg_label:str, particle:str='Pr') -> None:

        cfg = self.config[cfg_label]['plot']

        canvas = TCanvas(f'BB_{particle}', f'Bethe Bloch curve - {particle}')
        hframe = canvas.DrawFrame(cfg['xMin'], cfg['yMin'], cfg['xMax'], cfg['yMax'], f'Bethe Bloch {particle}; {cfg["xLabel"]}; {cfg["yLabel"]}')

        self.dir.cd()    
        self.hBB.Write()
        self.BBcurve.Write()
            
        canvas.cd()
        self.h2.Draw()
        self.BBcurve.Draw('same')
        self.dir.cd()
        canvas.Write()

    def drawNSigmaDistribution(self, cfg_label:str) -> None:

        cfg = self.config[cfg_label]
        cfg_prev_plot = self.config[cfg_label]['plot']
        cfg_plot = self.config[cfg_label]['plot_nsigma']
        
        col_name_exp = 'fExp' + cfg_prev_plot['yVariable'][1:]

        def BBfunc_vectorized(x):
            beta = x/(1 + x**2).sqrt()
            aa = beta**self.BetheBloch_params['kp4']
            bb = (1/x)**self.BetheBloch_params['kp5']
            bb = (self.BetheBloch_params['kp3'] + bb).log()
            return (self.BetheBloch_params['kp2'] - aa - bb) * self.BetheBloch_params['kp1'] / aa

        self.data_handler.dataset = self.data_handler.dataset.with_columns(BBfunc_vectorized(pl.col(cfg_plot['xVariable'])).alias(col_name_exp))
        self.data_handler.dataset = self.data_handler.dataset.with_columns(((pl.col(cfg_prev_plot['yVariable']) - pl.col(col_name_exp)) / (0.09 * pl.col(col_name_exp))).alias(f'fNSigma{self.fitted_particle}'))

        hist_handler = {'all': HistHandler.createInstance(data_handler.dataset)}
        for ipart, part in enumerate(self.config['species']):
            ds = self.data_handler.dataset.filter(pl.col('fPartID') == ipart+1)
            hist_handler[part] = HistHandler.createInstance(ds)

        self.dir.cd()
        for part in cfg_plot['particle']:
    
            title = cfg_plot['title'].split(';')[0] + f' ({part})' 
            for term in cfg_plot['title'].split(';')[1:]:
                title += ';' + term
            axisSpecX = AxisSpec(cfg_plot['nXBins'], cfg_plot['xMin'], cfg_plot['xMax'], cfg_plot['name']+f'_{part}', title)
            axisSpecY = AxisSpec(cfg_plot['nYBins'], cfg_plot['yMin'], cfg_plot['yMax'], cfg_plot['name']+f'_{part}', title)
            hist = hist_handler[part].buildTH2(cfg_plot['xVariable'], cfg_plot['yVariable'], axisSpecX, axisSpecY)

            hist.Write()

    def confusionMatrixPr(self, nsigma_cut:float) -> float:
        '''
            Evaluate the purity of the particle species
        '''

        parts = ['all', 'Pr']
        confus_matrix = np.zeros((len(parts), len(parts)))

        purity = 0
        dss = {
            'all': self.data_handler.dataset.filter(pl.col('fPartID') != self.config['species'].index('Pr')+1),
            'Pr': self.data_handler.dataset.filter(pl.col('fPartID') == self.config['species'].index('Pr')+1),
        }
        for idx, (part, ds) in enumerate(dss.items()):
            print('NSigma distribution for', part)
            print(ds[[f'fNSigma{self.fitted_particle}']].describe())
            
            ds_pr = ds.filter(np.abs(pl.col(f'fNSigma{self.fitted_particle}')) < nsigma_cut)
            ds_all = ds.filter(np.abs(pl.col(f'fNSigma{self.fitted_particle}')) > nsigma_cut)   
            
            print(ds_pr.shape, ds_all.shape)
            confus_matrix[idx][0] = ds_all.shape[0]
            confus_matrix[idx][1] = ds_pr.shape[0]

        # draw confusion matrix
        canvas = TCanvas(f'confusion_matrix', 'Confusion matrix')
        hconfus_matrix = TH2F('confusion_matrix', 'Confusion matrix; True; Predicted', 2, 0, 2, 2, 0, 2)
        hconfus_matrix.GetXaxis().SetBinLabel(1, 'all other')
        hconfus_matrix.GetXaxis().SetBinLabel(2, 'Pr')
        hconfus_matrix.GetYaxis().SetBinLabel(2, 'all other')
        hconfus_matrix.GetYaxis().SetBinLabel(1, 'Pr')
        tot = np.sum(confus_matrix)
        for i in range(2):
            for j in range(2):
                hconfus_matrix.SetBinContent(i+1, j+1, confus_matrix[i][1-j]/tot)

        self.dir.cd()
        hconfus_matrix.Write()
        print(tc.BOLD+'Confusion matrix:'+tc.RESET)
        print(confus_matrix)
        print(tc.BOLD+'Efficiency Pr:'+tc.RESET, confus_matrix[1][1]/(confus_matrix[1][0] + confus_matrix[1][1]))
        print(tc.BOLD+'Purity Pr:'+tc.RESET, confus_matrix[1][1]/(confus_matrix[0][1] + confus_matrix[1][1]))





            


        


        


if __name__ == '__main__':

    log = open("myprog.ansi", "w")
    sys.stdout = log

    input_files = ['/home/galucia/ITS_pid/o2/tree_creator/AO2D.root']
    cfg_data_file = '../config/config_data.yml'
    cfg_output_file = 'bethe_bloch_parametrisation.yml'
    output_file = '../output/bethe_bloch_parametrisation.root'
    tree_name = 'O2clsttableextra'
    folder_name = 'DF_*' 

    data_handler = DataHandler(input_files, cfg_data_file, tree_name=tree_name, folder_name=folder_name)

    outFile = TFile(output_file, 'RECREATE')
    bb_param = BetheBlochParametrisation(data_handler, cfg_output_file)

    dir_p = outFile.mkdir('clsize_vs_p')
    bb_param.setOutputDir(dir_p)
    bb_param.generateGraph('p', 'Pr')
    bb_param.fitBetheBloch('p', 'Pr')
    bb_param.drawBetheBloch('p', 'Pr')
    bb_param.drawNSigmaDistribution('p')
    bb_param.confusionMatrixPr(3)

    dir_p = outFile.mkdir('clsize_vs_beta')
    bb_param.setOutputDir(dir_p)
    bb_param.generateGraph('beta', 'Pr')
    bb_param.fitBetheBloch('beta', 'Pr')
    bb_param.drawBetheBloch('beta', 'Pr')
    bb_param.drawNSigmaDistribution('beta')

    outFile.Close()