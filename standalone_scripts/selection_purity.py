'''
    Study the purity of selected particles
'''

import os
import numpy as np
import uproot   
from ROOT import TFile, RooRealVar, RooGaussian, RooDataHist, RooGenericPdf, RooAddPdf, RooArgList, RooAbsData, RooChebychev, RooGExpModel, RooCrystalBall, RooConstVar
from ROOT import kRed, kCyan, kOrange, kBlue, kGreen, kMagenta
from ROOT import TF1, TCanvas, gInterpreter, gStyle, TLegend

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHEBYSHEV_DIR = os.path.join(CURRENT_DIR, '..', 'include', 'DoubleSidedCrystalBall.hh')
gInterpreter.ProcessLine(f'#include "{CHEBYSHEV_DIR}"')
from ROOT import DoubleSidedCrystalBall, DoubleSidedCrystalBallPol2, DoubleSidedCrystalBallExp  

import sys
sys.path.append('..')
from framework.utils.terminal_colors import TerminalColors as tc

def purityLambdaTF1(inFile:str, inHist:str, outFile:TFile):

    hInvMassLambdaPt = uproot.open(inFile)[inHist].to_pyroot()
    hInvMassLambda = hInvMassLambdaPt.ProjectionY('hInvMassLambda', 0, -1)
    for ibin in range(1, hInvMassLambda.GetNbinsX()+1):
        hInvMassLambda.SetBinError(ibin, np.sqrt(hInvMassLambda.GetBinContent(ibin)))
        print('mass:', hInvMassLambda.GetBinCenter(ibin), 'content:', hInvMassLambda.GetBinContent(ibin), 'error:', hInvMassLambda.GetBinError(ibin))

    #signal = TF1('signal', '[0]*ROOT::Math::crystalball_function(x, [3], [4], [2], [1])', 1.09, 1.15)
    signal = TF1('signal', DoubleSidedCrystalBall, 1.09, 1.15, 7)
    signal.SetParameters(1.5e7, 1.115683, 0.001, 1.5, 1.5) # norm, mean, sigma, alpha, n

    background = TF1('background', '[7] * exp(-[8]*x)', 1.09, 1.15)
    #background = TF1('background', '[7] + [8]*x + [9]*x*x', 1.09, 1.15)
    background.SetParameters(0, 0, 0, 0, 0, 1.115, 0.006, 10) # a0, a1

    #model = TF1('model', '[0]*ROOT::Math::crystalball_function(x, [3], [4], [2], [1]) + [5] * exp(-[6]*x)', 1.09, 1.15)
    #model = TF1('model', '[0]*ROOT::Math::crystalball_function(x, [3], [4], [2], [1]) + [5] + [6]*x + [7]*x*x', 1.09, 1.15)
    #model = TF1('model', DoubleSidedCrystalBallPol2, 1.09, 1.15, 10)
    model = TF1('model', DoubleSidedCrystalBallExp, 1.09, 1.15, 9)
    #model.SetParameters(1e7, 1.115683, 0.001, 1.5, 1.5, 5e4, 10, 10)
    #model.SetParameters(1e7, 1.115683, 0.001, 1.5, 1.5, 5e4, 10, 10)
    model.SetParameters(1e7, 1.115683, 0.001, 1.5, 1.5, 1.5, 1.5, 5e4, 1)
    model.SetParLimits(1, 1.1146, 1.1166)
    model.SetParLimits(2, 0.0008, 0.002)
    model.SetParLimits(3, 0.7, 5)
    model.SetParLimits(4, 0., 5)
    model.SetParLimits(5, 0.7, 5)
    model.SetParLimits(6, 0., 5)
    model.SetParLimits(7, 0, 1e6)
    model.SetParLimits(8, 0.1, 5)

    hInvMassLambda.Fit(model, 'RMSL+')
    signal.SetParameters(model.GetParameter(0), model.GetParameter(1), model.GetParameter(2), model.GetParameter(3), model.GetParameter(4), model.GetParameter(5), model.GetParameter(6))
    background.SetParameters(0, 0, 0, 0, 0, 0, 0, model.GetParameter(7), model.GetParameter(8))#, model.GetParameter(9))
    print('\n'+tc.BOLD+tc.BLUE+'chi2 ='+tc.RESET, model.GetChisquare())
    print(tc.BOLD+tc.BLUE+'NDF ='+tc.RESET, model.GetNDF())
    print(tc.BOLD+tc.BLUE+'chi2/NDF ='+tc.RESET, model.GetChisquare()/model.GetNDF())
    print()

    gStyle.SetOptStat(0)
    c = TCanvas('cInvMassLambda', 'cInvMassLambda', 800, 600)
    #c.SetLogy()

    hInvMassLambda.GetXaxis().SetRangeUser(1.09, 1.15)
    hInvMassLambda.SetFillColorAlpha(kCyan-3, 0.5)
    hInvMassLambda.SetFillStyle(3001)
    hInvMassLambda.SetLineColor(kCyan-3)
    hInvMassLambda.Draw('hist e0')
    model.SetLineColor(kBlue-3)
    model.Draw('same')
    signal.SetLineColor(kRed-3)
    signal.Draw('same')
    background.SetLineColor(kOrange-3)
    background.Draw('same')

    legend = TLegend(0.6, 0.6, 0.89, 0.89)
    legend.SetBorderSize(0)
    legend.AddEntry(hInvMassLambda, 'Invariant mass of #Lambda', 'f')
    legend.AddEntry(model, 'Signal + Background', 'l')
    legend.AddEntry(signal, 'Signal', 'l')
    legend.AddEntry(background, 'Background', 'l')
    legend.Draw('same')

    mean = model.GetParameter(1)
    sigma = model.GetParameter(2)
    intSignal = signal.Integral(mean - 3*sigma, mean + 3*sigma)
    intBackground = background.Integral(mean - 3*sigma, mean + 3*sigma)
    purity = intSignal / (intSignal + intBackground)
    print('\n'+tc.BOLD+tc.BLUE+'Signal component:'+tc.RESET+f'{intSignal:.4f}')
    print(tc.BOLD+tc.BLUE+'Background component:'+tc.RESET+f'{intBackground:.4f}')
    print(tc.BOLD+tc.BLUE+'Purity:'+tc.RESET+f'{purity:.4f}')

    outFile.cd()
    hInvMassLambda.Write()
    c.Write()
    c.SaveAs('cInvMassLambda.pdf')


def purityOmegaTF1(inFile:str, inHist:str, outFile:TFile):

    hInvMassOmegaPt = uproot.open(inFile)[inHist].to_pyroot()
    hInvMassOmega = hInvMassOmegaPt.ProjectionY('hInvMassOmega', 0, -1)
    for ibin in range(1, hInvMassOmega.GetNbinsX()+1):
        hInvMassOmega.SetBinError(ibin, np.sqrt(hInvMassOmega.GetBinContent(ibin)))

    signal = TF1('signal', '[0]*ROOT::Math::crystalball_function(x, [3], [4], [2], [1])', 1.66, 1.69)
    signal.SetParameters(1.5e7, 1.115683, 0.001, 1.5, 1.5) # norm, mean, sigma, alpha, n

    background = TF1('background', '[5]', 1.66, 1.69)
    background.SetParameters(0, 0, 0, 0, 0, 1.115) # a0, a1

    model = TF1('model', '[0]*ROOT::Math::crystalball_function(x, [3], [4], [2], [1]) + [5]', 1.66, 1.69)
    model.SetParameters(1800, 1.67245, 0.004, 1.5, 1.5, 42)
    signal.SetParLimits(1, 1.671, 1.674)
    signal.SetParLimits(2, 0.0001, 0.01)

    hInvMassOmega.Fit(model, 'RMSL+')
    signal.SetParameters(model.GetParameter(0), model.GetParameter(1), model.GetParameter(2), model.GetParameter(3), model.GetParameter(4))
    background.SetParameter(5, model.GetParameter(5))
    print('\n'+tc.BOLD+tc.BLUE+'chi2 ='+tc.RESET, model.GetChisquare())
    print(tc.BOLD+tc.BLUE+'NDF ='+tc.RESET, model.GetNDF())
    print(tc.BOLD+tc.BLUE+'chi2/NDF ='+tc.RESET, model.GetChisquare()/model.GetNDF())
    print()

    gStyle.SetOptStat(0)
    c = TCanvas('cInvMassOmega', 'cInvMassOmega', 800, 600)
    
    hInvMassOmega.GetXaxis().SetRangeUser(1.66, 1.69)
    hInvMassOmega.SetFillColorAlpha(kCyan-3, 0.5)
    hInvMassOmega.SetFillStyle(3001)
    hInvMassOmega.SetLineColor(kCyan-3)
    hInvMassOmega.Draw('hist')
    model.SetLineColor(kBlue-3)
    model.Draw('same')
    signal.SetLineColor(kRed-3)
    signal.Draw('same')
    background.SetLineColor(kOrange-3)
    background.Draw('same')

    legend = TLegend(0.6, 0.6, 0.89, 0.89)
    legend.SetBorderSize(0)
    legend.AddEntry(hInvMassOmega, 'Invariant mass of #Omega', 'f')
    legend.AddEntry(model, 'Signal + Background', 'l')
    legend.AddEntry(signal, 'Signal', 'l')
    legend.AddEntry(background, 'Background', 'l')
    legend.Draw('same')

    mean = model.GetParameter(1)
    sigma = model.GetParameter(2)
    intSignal = signal.Integral(mean - 3*sigma, mean + 3*sigma)
    intBackground = background.Integral(mean - 3*sigma, mean + 3*sigma)
    purity = intSignal / (intSignal + intBackground)
    print('\n'+tc.BOLD+tc.BLUE+'Signal component:'+tc.RESET+f'{intSignal:.4f}')
    print(tc.BOLD+tc.BLUE+'Background component:'+tc.RESET+f'{intBackground:.4f}')
    print(tc.BOLD+tc.BLUE+'Purity:'+tc.RESET+f'{purity:.4f}')


    outFile.cd()
    hInvMassOmega.Write()
    c.Write()
    c.SaveAs('cInvMassOmega.pdf')



def purityLambda(inFile:str, inHist:str, outFile:TFile):

    hInvMassLambdaPt = uproot.open(inFile)[inHist].to_pyroot()
    hInvMassLambda = hInvMassLambdaPt.ProjectionY('hInvMassLambda', 0, -1)
    for ibin in range(1, hInvMassLambda.GetNbinsX()+1):
        hInvMassLambda.SetBinError(ibin, np.sqrt(hInvMassLambda.GetBinContent(ibin)))

    print(type(hInvMassLambda))
    hInvMassLambda.Print()

    outFile.cd()
    hInvMassLambda.Write()

    x = RooRealVar('mass', 'mass (GeV/#it{c}^{2})', 1.11, 1.13)
    dataHist = RooDataHist('data_hist', 'data_hist', [x], Import=hInvMassLambda)

    frame = x.frame(Name='fInvMassLambda', Title='Invariant mass of #Lambda')
    dataHist.plotOn(frame)
    
    mean = RooRealVar('mean', 'mean', 1.115683, 1.1146, 1.1166)
    sigma = RooRealVar('sigma', 'sigma', 0.00001, 0.0008, 0.002)
    #gauss = RooGaussian('gauss', 'gauss', x, mean, sigma)
    alpha = RooRealVar('alpha', 'alpha', 1.5, -1e4, 1e4)
    n = RooRealVar('n', 'n', 1.5, -1e4, 1e4)
    signal = RooCrystalBall('signal', 'signal', x, mean, sigma, alpha, n)
    
    npars = 7
    a0 = RooRealVar('a0', 'a0', 1.115, 1.11, 1.12)
    a1 = RooRealVar('a1', 'a1', 0.006, 0.002, 0.01)
    a2 = RooRealVar('a2', 'a2', 2, 0, 1e2)
    #a3 = RooRealVar('a3', 'a3', 0, -1e3, 1e3)
    meanSF = RooConstVar('meanSF', 'meanSF', 1.)
    sigmaSF = RooConstVar('sigmaSF', 'sigmaSF', 1.)
    rlifeSF = RooConstVar('rlifeSF', 'rlifeSF', 1.)
    #background = RooGenericPdf('background', 'a0*exp(-a1*mass)', 'a0*exp(-a1*mass)', [x, a0, a1])
    #background = RooGenericPdf('background', '(a0+a1*mass)*exp(-a2*mass)', '(a0+a1*mass)*exp(-a2*mass)', [x, a0, a1, a2])
    #background = RooChebychev('background', 'background', x, RooArgList(a0, a1, a2)) #, a3))
    #background = RooGenericPdf('background', 'a0 + mass*a1 + mass*mass*a2', 'a0 + mass*a1 + mass*mass*a2', [x, a0, a1, a2])
    background = RooGExpModel('background', 'background', x, a0, a1, a2, meanSF, sigmaSF, rlifeSF)

    sigfrac = RooRealVar('sigfrac', 'sigfrac', 0.5, 0., 1.)
    bkgfrac = RooRealVar('bkgfrac', 'bkgfrac', 0.5, 0., 1.)
    model = RooAddPdf('model', 'model', [signal, background], [sigfrac, bkgfrac])
    model.fitTo(dataHist, PrintLevel=-1)
    chi2 = model.createChi2(dataHist, Range="fullRange", DataError=RooAbsData.Poisson)
    print('\n'+tc.BOLD+tc.BLUE+'chi2 ='+tc.RESET, chi2.getVal())
    print(tc.BOLD+tc.BLUE+'NDF ='+tc.RESET, hInvMassLambda.GetNbinsX()-(npars+1))
    print(tc.BOLD+tc.BLUE+'chi2/NDF ='+tc.RESET, chi2.getVal()/(hInvMassLambda.GetNbinsX()-(npars+1)))
    print()

    model.plotOn(frame, Components={background}, LineStyle="--", LineColor=kRed)
    model.plotOn(frame, Components={signal, background}, LineStyle=":", LineColor=kCyan+3)

    signal.paramOn(frame, Layout=(0.65, 0.9, 0.9))
    background.paramOn(frame, Layout=(0.65, 0.9, 0.7))

    model.Print("t")

    x.setRange('signal', mean.getValV() - 3*sigma.getValV(), mean.getValV() + 3*sigma.getValV())
    intSignal = signal.createIntegral({x}, NormSet={x}, Range='signal')
    intBackground = background.createIntegral({x}, NormSet={x}, Range='signal')
    sigComponent = intSignal.getVal() * sigfrac.getVal()
    bkgComponent = intBackground.getVal() * bkgfrac.getVal()
    purity = sigComponent / (sigComponent + bkgComponent)
    print('\n'+tc.BOLD+tc.BLUE+'Signal component:'+tc.RESET+f'{sigComponent:.4f}')
    print(tc.BOLD+tc.BLUE+'Background component:'+tc.RESET+f'{bkgComponent:.4f}')
    print(tc.BOLD+tc.BLUE+'Purity:'+tc.RESET+f'{purity:.4f}')

    outFile.cd()
    frame.Write()


def purityOmega(inFile:str, inHist:str, outFile:TFile):

    hInvMassOmegaPt = uproot.open(inFile)[inHist].to_pyroot()
    hInvMassOmega = hInvMassOmegaPt.ProjectionY('hInvMassOmega', 0, -1)
    for ibin in range(1, hInvMassOmega.GetNbinsX()+1):
        hInvMassOmega.SetBinError(ibin, np.sqrt(hInvMassOmega.GetBinContent(ibin)))

    print(type(hInvMassOmega))
    hInvMassOmega.Print()

    outFile.cd()
    hInvMassOmega.Write()

    x = RooRealVar('mass', 'mass (GeV/#it{c}^{2})', 1.66, 1.68)
    dataHist = RooDataHist('data_hist', 'data_hist', [x], Import=hInvMassOmega)

    frame = x.frame(Name='fInvMassOmega', Title='Invariant mass of #Omega')
    dataHist.plotOn(frame)
    
    mean = RooRealVar('mean', 'mean', 1.67245, 1.671, 1.674)
    sigma = RooRealVar('sigma', 'sigma', 0.0004, 0.0001, 0.01)
    signal = RooGaussian('gauss', 'gauss', x, mean, sigma)
    #alpha = RooRealVar('alpha', 'alpha', 1.5, -1e4, 1e4)
    #n = RooRealVar('n', 'n', 1.5, -1e4, 1e4)
    #signal = RooCrystalBall('signal', 'signal', x, mean, sigma, alpha, n)
    
    npars = 4
    a0 = RooRealVar('a0', 'a0', 40., 0., 1e2)
    a1 = RooRealVar('a1', 'a1', 1., -1e3, 1e3)
    #a0 = RooRealVar('a0', 'a0', 1.115, 1.11, 1.12)
    #a1 = RooRealVar('a1', 'a1', 0.006, 0.002, 0.01)
    a2 = RooRealVar('a2', 'a2', 2, -100, 100)
    #a3 = RooRealVar('a3', 'a3', 0, -1e3, 1e3)
    meanSF = RooConstVar('meanSF', 'meanSF', 1.)
    sigmaSF = RooConstVar('sigmaSF', 'sigmaSF', 1.)
    rlifeSF = RooConstVar('rlifeSF', 'rlifeSF', 1.)
    background = RooGenericPdf('background', 'a2 + a0*exp(-a1*mass)', 'a2 + a0*exp(-a1*mass)', [x, a0, a1, a2])
    #background = RooGenericPdf('background', '(a0+a1*mass)*exp(-a2*mass)', '(a0+a1*mass)*exp(-a2*mass)', [x, a0, a1, a2])
    #background = RooChebychev('background', 'background', x, RooArgList(a0, a1, a2)) #, a3))
    #background = RooGenericPdf('background', 'a0 + mass*a1 + mass*mass*a2', 'a0 + mass*a1 + mass*mass*a2', [x, a0, a1, a2])
    #background = RooGExpModel('background', 'background', x, a0, a1, a2, meanSF, sigmaSF, rlifeSF)

    sigfrac = RooRealVar('sigfrac', 'sigfrac', 0.5, 0., 1.)
    bkgfrac = RooRealVar('bkgfrac', 'bkgfrac', 0.5, 0., 1.)
    model = RooAddPdf('model', 'model', [signal, background], [sigfrac, bkgfrac])
    model.fitTo(dataHist, PrintLevel=-1)
    chi2 = model.createChi2(dataHist, Range="fullRange", DataError=RooAbsData.Poisson)
    print('\n'+tc.BOLD+tc.BLUE+'chi2 ='+tc.RESET, chi2.getVal())
    print(tc.BOLD+tc.BLUE+'NDF ='+tc.RESET, hInvMassOmega.GetNbinsX()-(npars+1))
    print(tc.BOLD+tc.BLUE+'chi2/NDF ='+tc.RESET, chi2.getVal()/(hInvMassOmega.GetNbinsX()-(npars+1)))
    print()

    model.plotOn(frame, Components={background}, LineStyle="--", LineColor=kRed)
    model.plotOn(frame, Components={signal, background}, LineStyle=":", LineColor=kCyan+3)

    signal.paramOn(frame, Layout=(0.65, 0.9, 0.9))
    background.paramOn(frame, Layout=(0.65, 0.9, 0.7))

    model.Print("t")

    x.setRange('signal', mean.getValV() - 3*sigma.getValV(), mean.getValV() + 3*sigma.getValV())
    intSignal = signal.createIntegral({x}, NormSet={x}, Range='signal')
    intBackground = background.createIntegral({x}, NormSet={x}, Range='signal')
    sigComponent = intSignal.getVal() * sigfrac.getVal()
    bkgComponent = intBackground.getVal() * bkgfrac.getVal()
    purity = sigComponent / (sigComponent + bkgComponent)


    print('\n'+tc.BOLD+tc.BLUE+'Signal component:'+tc.RESET+f'{sigComponent:.4f}')
    print(tc.BOLD+tc.BLUE+'Background component:'+tc.RESET+f'{bkgComponent:.4f}')
    print(tc.BOLD+tc.BLUE+'Purity:'+tc.RESET+f'{purity:.4f}')

    outFile.cd()
    frame.Write()










if __name__ == '__main__':

    output_dir = '../output/LHC22o_pass7_minBias_small'
    output_file = output_dir+'/purity_lambdaomega.root'
    outFile = TFile(output_file, 'RECREATE')
    purityLambda('/data/galucia/its_pid/pass7/AnalysisResults.root', 'lf-tree-creator-cluster-studies/LFTreeCreator/massLambda', outFile)
    purityOmega('/data/galucia/its_pid/pass7/AnalysisResults.root', 'lf-tree-creator-cluster-studies/LFTreeCreator/massOmega', outFile)
    #purity_Lambda('/data/galucia/its_pid/pass7/AnalysisResults.root', 'lf-tree-creator-cluster-studies/LFTreeCreator/massLambda', outFile)

    purityLambdaTF1('/data/galucia/its_pid/pass7/AnalysisResults.root', 'lf-tree-creator-cluster-studies/LFTreeCreator/massLambda', outFile)
    purityOmegaTF1('/data/galucia/its_pid/pass7/AnalysisResults.root', 'lf-tree-creator-cluster-studies/LFTreeCreator/massOmega', outFile)
