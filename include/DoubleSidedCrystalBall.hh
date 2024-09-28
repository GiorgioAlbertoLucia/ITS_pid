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

double DoubleSidedCrystalBallPol2(double *x, double *par)
{
    double N = par[0];
    double mu = par[1];
    double sigma = par[2];
    double alphaL = par[3];
    double nL = par[4];
    double alphaR = par[5];
    double nR = par[6];
    
    double kp0 = par[7];
    double kp1 = par[8];
    double kp2 = par[9];

    double t = (x[0] - mu) / sigma;
    if (alphaL < 0 && alphaR < 0)
    {
        if (t < -alphaL)
        {
            return N * std::exp(-0.5 * alphaL * alphaL) * std::pow(nL / std::fabs(alphaL), nL) + kp0 + kp1*x[0] + kp2*x[0]*x[0];
        }
        else if (t > alphaR)
        {
            return N * std::exp(-0.5 * alphaR * alphaR) * std::pow(nR / std::fabs(alphaR), nR) + kp0 + kp1*x[0] + kp2*x[0]*x[0];
        }
        else
        {
            return N * std::exp(-0.5 * t * t) + kp0 + kp1*x[0] + kp2*x[0]*x[0];
        }
    }
    else
    {
        double absAlphaL = std::fabs(alphaL);
        double absAlphaR = std::fabs(alphaR);
        if (t < -absAlphaL)
        {
            return N * std::exp(-0.5 * absAlphaL * absAlphaL) * std::pow(nL / absAlphaL, nL) + kp0 + kp1*x[0] + kp2*x[0]*x[0];
        }
        else if (t > absAlphaR)
        {
            return N * std::exp(-0.5 * absAlphaR * absAlphaR) * std::pow(nR / absAlphaR, nR) + kp0 + kp1*x[0] + kp2*x[0]*x[0];
        }
        else
        {
            return N * std::exp(-0.5 * t * t) + kp0 + kp1*x[0] + kp2*x[0]*x[0];
        }
    }
}

double DoubleSidedCrystalBallExp(double *x, double *par)
{
    double N = par[0];
    double mu = par[1];
    double sigma = par[2];
    double alphaL = par[3];
    double nL = par[4];
    double alphaR = par[5];
    double nR = par[6];
    
    double kp0 = par[7];
    double kp1 = par[8];

    double t = (x[0] - mu) / sigma;
    if (alphaL < 0 && alphaR < 0)
    {
        if (t < -alphaL)
        {
            return N * std::exp(-0.5 * alphaL * alphaL) * std::pow(nL / std::fabs(alphaL), nL) + kp0 * std::exp(-kp1*x[0]);
        }
        else if (t > alphaR)
        {
            return N * std::exp(-0.5 * alphaR * alphaR) * std::pow(nR / std::fabs(alphaR), nR) + kp0 * std::exp(-kp1*x[0]);
        }
        else
        {
            return N * std::exp(-0.5 * t * t) + kp0 * std::exp(-kp1*x[0]);
        }
    }
    else
    {
        double absAlphaL = std::fabs(alphaL);
        double absAlphaR = std::fabs(alphaR);
        if (t < -absAlphaL)
        {
            return N * std::exp(-0.5 * absAlphaL * absAlphaL) * std::pow(nL / absAlphaL, nL) + kp0 * std::exp(-kp1*x[0]);
        }
        else if (t > absAlphaR)
        {
            return N * std::exp(-0.5 * absAlphaR * absAlphaR) * std::pow(nR / absAlphaR, nR) + kp0 * std::exp(-kp1*x[0]);
        }
        else
        {
            return N * std::exp(-0.5 * t * t) + kp0 * std::exp(-kp1*x[0]);
        }
    }
}