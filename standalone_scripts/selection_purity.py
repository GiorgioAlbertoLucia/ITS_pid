'''
    Study the purity of selected particles
'''

import os
import numpy as np
import uproot   
from ROOT import TFile, RooRealVar, RooGaussian, RooDataHist, RooGenericPdf, RooAddPdf, RooArgList, RooAbsData, RooChebychev, RooGExpModel, RooCrystalBall, RooConstVar
from ROOT import kRed, kCyan, kOrange
from ROOT import TF1, TCanvas, gInterpreter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHEBYSHEV_DIR = os.path.join(CURRENT_DIR, '..', 'include', 'ChebyshevPolynomial.hh')
gInterpreter.ProcessLine(f'#include "{CHEBYSHEV_DIR}"')
from ROOT import ChebyshevPolynomial 

import sys
sys.path.append('..')
from framework.utils.terminal_colors import TerminalColors as tc

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
