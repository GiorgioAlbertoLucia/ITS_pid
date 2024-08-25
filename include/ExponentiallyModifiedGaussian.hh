//
// Exponentially Modified Gaussian (EMG) distribution
//

#pragma once

#include <cmath>

double ExponentiallyModifiedGaussian(double *x, double *par)
{
    double N = par[0];
    double mu = par[1];
    double sigma = par[2];
    double lambda = par[3];

    double t = (x[0] - mu) / sigma;
    if (lambda < 0)
    {
        return 0;
    }
    else
    {
        return N * 0.5 * lambda * std::exp(0.5 * lambda * (2 * mu + lambda * sigma * sigma - 2 * x[0])) * std::erfc((mu + lambda * sigma * sigma - x[0]) / (std::sqrt(2) * sigma));
    }
}