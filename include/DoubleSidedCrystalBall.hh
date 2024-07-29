/**
 * @file DoubleSidedCrystalBall.hh
 * 
 * Implementation for a double-sided Crystal Ball function.
*/

#pragma once
#include <cmath>

double DoubleSidedCrystalBall(double *x, double *par)
{
    double N = par[0];
    double mu = par[1];
    double sigma = par[2];
    double alphaL = par[3];
    double nL = par[4];
    double alphaR = par[5];
    double nR = par[6];

    double t = (x[0] - mu) / sigma;
    if (alphaL < 0 && alphaR < 0)
    {
        if (t < -alphaL)
        {
            return N * std::exp(-0.5 * alphaL * alphaL) * std::pow(nL / std::fabs(alphaL), nL);
        }
        else if (t > alphaR)
        {
            return N * std::exp(-0.5 * alphaR * alphaR) * std::pow(nR / std::fabs(alphaR), nR);
        }
        else
        {
            return N * std::exp(-0.5 * t * t);
        }
    }
    else
    {
        double absAlphaL = std::fabs(alphaL);
        double absAlphaR = std::fabs(alphaR);
        if (t < -absAlphaL)
        {
            return N * std::exp(-0.5 * absAlphaL * absAlphaL) * std::pow(nL / absAlphaL, nL);
        }
        else if (t > absAlphaR)
        {
            return N * std::exp(-0.5 * absAlphaR * absAlphaR) * std::pow(nR / absAlphaR, nR);
        }
        else
        {
            return N * std::exp(-0.5 * t * t);
        }
    }
}
