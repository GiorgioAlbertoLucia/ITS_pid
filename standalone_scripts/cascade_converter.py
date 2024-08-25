from ROOT import TFile, TGraphErrors, TH1F

if __name__ == '__main__':


    infile_Xi = TFile('/home/galucia/ITS_pid/output/cluster_size_xi.root', 'READ')
    infile_Omega = TFile('/home/galucia/ITS_pid/output/cluster_size_omega.root', 'READ')
    outfile = TFile('/home/galucia/ITS_pid/output/cluster_size_cascade.root', 'RECREATE')

    xi_h = infile_Xi.Get('mean_xi_pos')
    xi_g = TGraphErrors(xi_h.GetNbinsX())
    for ix in range(1, xi_h.GetNbinsX()+1):
        xi_g.SetPoint(ix-1, xi_h.GetXaxis().GetBinCenter(ix), xi_h.GetBinContent(ix))
        xi_g.SetPointError(ix-1, xi_h.GetBinWidth(ix)/2, xi_h.GetBinError(ix))

    omega_h = infile_Omega.Get('mean_omega_pos')
    omega_g = TGraphErrors(omega_h.GetNbinsX())
    for ix in range(1, omega_h.GetNbinsX()+1):
        omega_g.SetPoint(ix-1, omega_h.GetXaxis().GetBinCenter(ix), omega_h.GetBinContent(ix))
        omega_g.SetPointError(ix-1, omega_h.GetBinWidth(ix)/2, omega_h.GetBinError(ix))

    outfile.cd()    
    xi_g.Write('mean_xi_pos')
    omega_g.Write('mean_omega_pos')

    # vs momentum
    mass_xi = 1.32171
    mass_omega = 1.67245

    xi_h = infile_Xi.Get('mean_xi_pos')
    xi_g = TGraphErrors(xi_h.GetNbinsX())
    for ix in range(1, xi_h.GetNbinsX()+1):
        xi_g.SetPoint(ix-1, xi_h.GetXaxis().GetBinCenter(ix)*mass_xi, xi_h.GetBinContent(ix))
        xi_g.SetPointError(ix-1, xi_h.GetBinWidth(ix)/2*mass_xi, xi_h.GetBinError(ix))

    omega_h = infile_Omega.Get('mean_omega_pos')
    omega_g = TGraphErrors(omega_h.GetNbinsX())
    for ix in range(1, omega_h.GetNbinsX()+1):
        omega_g.SetPoint(ix-1, omega_h.GetXaxis().GetBinCenter(ix)*mass_omega, omega_h.GetBinContent(ix))
        omega_g.SetPointError(ix-1, omega_h.GetBinWidth(ix)/2*mass_omega, omega_h.GetBinError(ix))

    outfile.cd()    
    xi_g.Write('mean_xi_pos_vs_p')
    omega_g.Write('mean_omega_pos_vs_p')


    outfile.Close()
    