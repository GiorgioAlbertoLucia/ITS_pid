#include <TF1.h>
#include <TH1.h>

#include <vector>


// Function to calculate the Chebyshev polynomial of the first kind, T_n(x)
double ChebyshevPolynomial(double *x, double *par) {
    int n = (int)par[0]; // The order of the Chebyshev polynomial
    double xVal = x[0];

    if (n == 0) return 1;
    if (n == 1) return xVal;

    double Tn_minus_2 = 1;       // T_0(x)
    double Tn_minus_1 = xVal;    // T_1(x)
    double Tn = 0;

    for (int i = 2; i <= n; i++) {
        Tn = 2 * xVal * Tn_minus_1 - Tn_minus_2;
        Tn_minus_2 = Tn_minus_1;
        Tn_minus_1 = Tn;
    }

    return Tn;
}
/**
 * Sum of Chebyshev polynomials of the first kind, T_n(x), and a Gaussian function
*/
double ChebyshevPolynomialGaussian(double *x, double *par) {
    int n = (int)par[0]; // The order of the Chebyshev polynomial
    double xVal = x[0];

    if (n == 0) return 1;
    if (n == 1) return xVal;

    double Tn_minus_2 = 1;       // T_0(x)
    double Tn_minus_1 = xVal;    // T_1(x)
    double Tn = 0;

    for (int i = 2; i <= n; i++) {
        Tn = 2 * xVal * Tn_minus_1 - Tn_minus_2;
        Tn_minus_2 = Tn_minus_1;
        Tn_minus_1 = Tn;
    }

    return Tn + par[1] * TMath::Gaus(xVal, par[2], par[3]);
}