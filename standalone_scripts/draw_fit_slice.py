from ROOT import TFile, TF1, TCanvas, TGraphErrors, TMultiGraph, kRed, kGreen, kBlue, kOrange, kCyan


if __name__ == '__main__':

    infile = TFile('/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_Pr.root', 'READ')
    outfile = TFile('/home/galucia/ITS_pid/output/fit_slice_Pr.root', 'RECREATE')

    h2 = infile.Get('clsize_vs_betagamma_Pr/h2_BB_Pr')
    h1 = h2.ProjectionY('h1', h2.GetXaxis().FindBin(0.5), h2.GetXaxis().FindBin(0.5))


    #sig_func = TF1('sig', 'x <= ([1] + [3]*[2]) ? [0] * exp( - ( (x - [1]) / (std::sqrt(2) * [2]) )^2 ) : [0] * exp( - (x - [1] - 0.5*[2]*[3]) * [3]/[2])', 0, 15)
    sig_func = TF1('sig', '[0] * exp(-0.5 * pow((x - [1]) /[2],2))', 0, 15)
    fm_func = TF1('fm', '[3] * exp(-0.5 * pow((x - [4]) /[5],2))', 0, 15)
    fit_func = TF1('fit', '[0] * exp(-0.5 * pow((x - [1]) /[2],2)) + [3] * exp(-0.5 * pow((x - [4]) /[5],2))', 0, 15)
    fit_func.SetParameters(32781.23, 5.29, 0.97, 22087.31, 2.22, 0.64)

    #h1.Fit(fit_func, 'RSML+')

    #sig_func.SetParameters(fit_func.GetParameter(0), fit_func.GetParameter(1), fit_func.GetParameter(2), fit_func.GetParameter(3))
    #fm_func.SetParameters(fit_func.GetParameter(4), fit_func.GetParameter(5), fit_func.GetParameter(6))
    sig_func.SetParameters(fit_func.GetParameter(0), fit_func.GetParameter(1), fit_func.GetParameter(2))
    fm_func.SetParameters(0,0,0,fit_func.GetParameter(3), fit_func.GetParameter(4), fit_func.GetParameter(5))

    c = TCanvas('c', 'c', 800, 600)
    h1.Draw()
    sig_func.SetLineColor(kRed)
    sig_func.Draw('SAME')
    fm_func.SetLineColor(kBlue)
    fm_func.Draw('SAME')
    fit_func.SetLineColor(kGreen)
    fit_func.Draw('SAME')

    outfile.cd()
    sig_func.Write()
    fm_func.Write()
    fit_func.Write()
    h1.Write()
    c.Write()
    outfile.Close()