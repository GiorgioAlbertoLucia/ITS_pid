'''
    Class to manage and initialize it functions. Integrated with onfiguration files.
'''

import yaml
from typing import List
from ROOT import TF1
import numpy as np
from sklearn.cluster import KMeans


class Fitter:

    def __init__(self, data, func_names: List[str], cfg:str):
        '''
            Initialize the fitter

            Parameters
            ----------
            cfg_data_file (str): configuration file for the data
            cfg_output_file (str): configuration file for the output
        '''
        
        self.cfg = cfg
        
        self.data = data
        self.funcs = {func_name: TF1(func_name, self.cfg[func_name]['expr']) for func_name in func_names}
        self.fit = None
        self.params = {}
        for func_name in func_names:
            for iparam, param in self.cfg[func_name]['params'].items():
                self.params[iparam] = [param.get('init', 0.),
                                       param.get('opt', 'set'),
                                       param.get('limits', [0., 0.])]

    def fit_func(self, func_name:str, **kwargs):
        '''
            Fit the function

            Parameters
            ----------
            func_name (str): name of the function to fit
        '''

        fit_func = self.funcs[func_name]
        fit_range = kwargs.get('range', None) 
        if fit_range:
            fit_func.SetRange(fit_range[0], fit_range[1])
        self.data.Fit(fit_func, kwargs.get('fit_option', 'S'))
        
        for iparam, param in self.cfg[func_name]['params'].items():
            old_param = self.params[iparam]
            param_value = fit_func.GetParameter(iparam)
            param_opt = kwargs.get('param_opt', old_param[1])
            param_limits = old_param[2]
            self.params[iparam] = [param_value, param_opt, param_limits]

    def auto_initialise(self):
        '''
            Automatically initialise the parameters.
            Expects the function with lower mean to be the first in the list.
        '''

        data_points = []
        for ibin in range(1, self.data.GetNbinsX()+1):
            data_points.extend([self.data.GetBinCenter(ibin)] * int(self.data.GetBinContent(ibin)))

        data_points = np.array(data_points)
        n_components = len(self.funcs)

        if len(data_points) <= 0:
            print('No data points to fit')
            return

        kmeans = KMeans(n_clusters=n_components, init='k-means++', n_init='auto').fit(data_points.reshape(-1, 1))
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        covariances = []
        for icomp in range(n_components):
            comp_data = data_points[np.where(np.array(labels)==icomp)[0]]
            covariances.append(np.cov(comp_data.T))

        weights = np.array([np.sum(labels==icomp) for icomp in range(n_components)])
        max_bin_contents = [np.max(data_points[labels == icomp]) for icomp in range(n_components)]

        # Sort centers and get the sorted indices
        sorted_indices = np.argsort(centers.flatten())
        #sorted_indices = sorted_indices[::-1]  

        # Reorder centers, covariances, and weights based on the sorted indices
        centers = centers[sorted_indices]
        covariances = [covariances[i] for i in sorted_indices]
        weights = weights[sorted_indices]
        max_bin_contents = [np.max(data_points[labels == icomp]) for icomp in range(n_components)]


        print('DEBUG - centers: ', centers)

        for func_name, label in zip(self.funcs, set(labels)):
            mean = centers[label][0]
            std = np.sqrt(covariances[label])
            norm = max_bin_contents[label]
            print('DEBUG - norm: ', max_bin_contents[label])
            self.params[self.cfg[func_name]['mean_idx']] = [mean, 'limit', [mean-1.*std, mean+1.*std]]
            self.params[self.cfg[func_name]['sigma_idx']] = [std, 'fix', [0.9*std, 2*std]]
            self.params[self.cfg[func_name]['norm_idx']] = [norm, 'set', [0.8*norm, 1.2*norm]]
        print('DEBUG - params: ', self.params)
        
    def perform_fit(self, **kwargs):
        '''
            Fit the function
        '''
        
        self.fit = TF1('fit', '+'.join([self.funcs[func_name].GetExpFormula().Data() for func_name in self.funcs]), self.data.GetXaxis().GetXmin(), self.data.GetXaxis().GetXmax())
        for iparam, param in self.params.items():
            if kwargs.get('debug', False):  print('param: ', param)
            par_value, par_opt, par_limits = param
            if par_opt == 'fix':
                self.fit.FixParameter(iparam, par_value)
            elif par_opt == 'set':
                self.fit.SetParameter(iparam, par_value)
            elif par_opt == 'limit':
                self.fit.SetParameter(iparam, par_value)
                self.fit.SetParLimits(iparam, par_limits[0], par_limits[1])
            else:
                raise ValueError('Invalid parameter option')
        
        fit_status = self.data.Fit(self.fit, kwargs.get('fit_option', 'RMS+'))
        return fit_status, self.fit



