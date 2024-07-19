'''
    Study the purity of selected particles
'''

import numpy as np
import uproot   
from ROOT import TFile, RooRealVar, RooGaussian, RooDataHist, RooGenericPdf, RooAddPdf, RooArgList, RooAbsData
from ROOT import kRed, kCyan

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

    x = RooRealVar('mass', 'mass (GeV/#it{c}^{2})', hInvMassLambda.GetXaxis().GetXmin(), hInvMassLambda.GetXaxis().GetXmax())
    dataHist = RooDataHist('data_hist', 'data_hist', [x], Import=hInvMassLambda)

    frame = x.frame(Title='Invariant mass of #Lambda')
    dataHist.plotOn(frame)
    
    mean = RooRealVar('mean', 'mean', 1.115683, 1.115683-0.01, 1.115683+0.01)
    sigma = RooRealVar('sigma', 'sigma', 0.002, 0.0001, 0.01)
    gauss = RooGaussian('gauss', 'gauss', x, mean, sigma)
    
    npars = 2
    a0 = RooRealVar('a0', 'a0', 0.5, 0., 1.)
    a1 = RooRealVar('a1', 'a1', 0.5, 0., 1.)
    #a2 = RooRealVar('a2', 'a2', 0.5, 0., 1.)
    background = RooGenericPdf('background', 'a0 + mass*a1', 'a0 + mass*a1', [x, a0, a1])
    #background = RooGenericPdf('background', 'a0 + mass*a1 + mass*mass*a2', 'a0 + mass*a1 + mass*mass*a2', [x, a0, a1, a2])

    sigfrac = RooRealVar('sigfrac', 'sigfrac', 0.5, 0., 1.)
    bkgfrac = RooRealVar('bkgfrac', 'bkgfrac', 0.5, 0., 1.)
    model = RooAddPdf('model', 'model', [gauss, background], [sigfrac, bkgfrac])
    model.fitTo(dataHist, PrintLevel=-1)
    chi2 = model.createChi2(dataHist, Range="fullRange", DataError=RooAbsData.Poisson)
    print('\n'+tc.BOLD+tc.BLUE+'chi2 ='+tc.RESET, chi2.getVal())
    print(tc.BOLD+tc.BLUE+'NDF ='+tc.RESET, hInvMassLambda.GetNbinsX()-(npars+1))
    print(tc.BOLD+tc.BLUE+'chi2/NDF ='+tc.RESET, chi2.getVal()/(hInvMassLambda.GetNbinsX()-(npars+1)))
    print()

    model.plotOn(frame, Components={background}, LineStyle="--", LineColor=kRed)
    model.plotOn(frame, Components={gauss, background}, LineStyle=":", LineColor=kCyan+3)

    gauss.paramOn(frame, Layout=(0.65, 0.9, 0.9))
    background.paramOn(frame, Layout=(0.65, 0.9, 0.7))

    model.Print("t")

    x.setRange('signal', mean.getValV() - 3*sigma.getValV(), mean.getValV() + 3*sigma.getValV())
    intSignal = gauss.createIntegral({x}, NormSet={x}, Range='signal')
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

    outFile = TFile('/home/galucia/ITS_pid/output/purity.root', 'RECREATE')
    purityLambda('/home/galucia/ITS_pid/o2/tree_creator/AnalysisResults.root', 'lf-tree-creator-cluster-studies/LFTreeCreator/massLambda', outFile)
