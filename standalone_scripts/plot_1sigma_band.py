from ROOT import TGraph, TFile, TCanvas

if __name__ == '__main__':

    infile = TFile('/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation.root', 'READ')
    mean_graph = infile.Get('clsize_vs_betagamma_Pr/clSizeCosL_vs_beta_Pr_sig_points')
    sigma_graph = infile.Get('clsize_vs_betagamma_Pr/clSizeCosL_vs_beta_Pr_sig_sigma_points')
    
    x_points = mean_graph.GetX()
    mean_points = mean_graph.GetY()
    sigma_norm_points = sigma_graph.GetY()
    
    sigma_points = [sigma_norm_points[i] * mean_points[i] for i in range(len(mean_points))]
    n = len(x_points)
    band = TGraph(2 * n)
    low_band = TGraph(n)
    high_band = TGraph(n)
    
    for ix in range(n):
        band.SetPoint(ix, x_points[ix], mean_points[ix] + sigma_points[ix])
        band.SetPoint(n + ix, x_points[n - 1 - ix], mean_points[n - 1 - ix] - sigma_points[n - 1 - ix])
        low_band.SetPoint(ix, x_points[ix], mean_points[ix] - sigma_points[ix])
        high_band.SetPoint(ix, x_points[ix], mean_points[ix] + sigma_points[ix])

    band.SetFillColorAlpha(5, 0.3)
    band.SetLineColor(5)
    band.SetLineWidth(2)

    low_band.SetLineColor(5)
    high_band.SetLineColor(5)
    low_band.SetLineStyle(2)    
    high_band.SetLineStyle(2)
    low_band.SetLineWidth(2)
    high_band.SetLineWidth(2)

    canvas = TCanvas('band', 'band')
    band.Draw('AF')
    low_band.Draw('L')
    high_band.Draw('L')

    outfile = TFile('/home/galucia/ITS_pid/output/LHC22o_pass7_minBias_small/bethe_bloch_parametrisation_band.root', 'RECREATE')
    canvas.Write()
    band.Write('pr_band')
    low_band.Write('pr_low_band')
    high_band.Write('pr_high_band')
    outfile.Close()
    infile.Close()

