import numpy as np
import math
from scipy import special
from numba import njit

###########################################################
# L-moments GEV fitting
# https://github.com/xiaoganghe/python-climate-visuals
###########################################################


# Calculate samples L-moments
def samlmom3_numpy(sample):
    """
    samlmom3 returns the first three L-moments of samples
    sample is the 1-d array
    n is the total number of the samples, j is the j_th sample
    """
    n = len(sample)
    sample = np.sort(sample.reshape(n))[::-1]
    b0 = np.mean(sample)
    b1 = np.array([(n - j - 1) * sample[j] / n / (n - 1) for j in range(n)]).sum()
    b2 = np.array(
        [
            (n - j - 1) * (n - j - 2) * sample[j] / n / (n - 1) / (n - 2)
            for j in range(n - 1)
        ]
    ).sum()
    lmom1 = b0
    lmom2 = 2 * b1 - b0
    lmom3 = 6 * (b2 - b1) + b0

    return np.array([lmom1, lmom2, lmom3])


# Estimate GEV parameters using numerical approximations
def pargev_numpy(lmom):
    """
    pargev returns the parameters of the Generalized Extreme Value
    distribution given the L-moments of samples
    """
    lmom_ratios = [lmom[0], lmom[1], lmom[2] / lmom[1]]

    SMALL = 1e-5
    eps = 1e-6
    maxit = 20

    # EU IS EULER'S CONSTANT
    EU = 0.57721566
    DL2 = math.log(2)
    DL3 = math.log(3)

    # COEFFICIENTS OF RATIONAL-FUNCTION APPROXIMATIONS FOR XI
    A0 = 0.28377530
    A1 = -1.21096399
    A2 = -2.50728214
    A3 = -1.13455566
    A4 = -0.07138022
    B1 = 2.06189696
    B2 = 1.31912239
    B3 = 0.25077104
    C1 = 1.59921491
    C2 = -0.48832213
    C3 = 0.01573152
    D1 = -0.64363929
    D2 = 0.08985247

    T3 = lmom_ratios[2]
    if lmom_ratios[1] <= 0 or abs(T3) >= 1:
        raise ValueError("Invalid L-Moments")

    if T3 <= 0:
        G = (A0 + T3 * (A1 + T3 * (A2 + T3 * (A3 + T3 * A4)))) / (
            1 + T3 * (B1 + T3 * (B2 + T3 * B3))
        )

        if T3 >= -0.8:
            para3 = G
            GAM = math.exp(special.gammaln(1 + G))
            para2 = lmom_ratios[1] * G / (GAM * (1 - 2**-G))
            para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
            return para1, para2, para3
        elif T3 <= -0.97:
            G = 1 - math.log(1 + T3) / DL2

        T0 = (T3 + 3) * 0.5
        for IT in range(1, maxit):
            X2 = 2**-G
            X3 = 3**-G
            XX2 = 1 - X2
            XX3 = 1 - X3
            T = XX3 / XX2
            DERIV = (XX2 * X3 * DL3 - XX3 * X2 * DL2) / (XX2**2)
            GOLD = G
            G -= (T - T0) / DERIV

            if abs(G - GOLD) <= eps * G:
                para3 = G
                GAM = math.exp(special.gammaln(1 + G))
                para2 = lmom_ratios[1] * G / (GAM * (1 - 2**-G))
                para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
                return para1, para2, para3
        raise Exception("Iteration has not converged")
    else:
        Z = 1 - T3
        G = (-1 + Z * (C1 + Z * (C2 + Z * C3))) / (1 + Z * (D1 + Z * D2))
        if abs(G) < SMALL:
            para2 = lmom_ratios[1] / DL2
            para1 = lmom_ratios[0] - EU * para2
            para3 = 0
        else:
            para3 = G
            GAM = math.exp(special.gammaln(1 + G))
            para2 = lmom_ratios[1] * G / (GAM * (1 - 2**-G))
            para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
        return np.array([para1, para2, para3])


###########################
# Numba version
###########################
@njit
def samlmom3_numba(sample):
    """
    samlmom3 returns the first three L-moments of samples
    sample is the 1-d array
    n is the total number of the samples, j is the j_th sample
    """
    # Sort in descending order
    sorted_sample = np.sort(sample)[::-1]
    n = len(sorted_sample)

    # Calculate mean directly
    b0 = np.sum(sorted_sample) / n

    # Pre-calculate the coefficients for b1
    b1 = 0.0
    for j in range(n):
        b1_coef = (n - j - 1) / (n * (n - 1))
        b1 += b1_coef * sorted_sample[j]

    # Pre-calculate the coefficients for b2
    b2 = 0.0
    for j in range(n - 1):  # Note: upper bound is n-1 for b2
        b2_coef = (n - j - 1) * (n - j - 2) / (n * (n - 1) * (n - 2))
        b2 += b2_coef * sorted_sample[j]

    # Calculate L-moments
    lmom1 = b0
    lmom2 = 2 * b1 - b0
    lmom3 = 6 * (b2 - b1) + b0

    return np.array([lmom1, lmom2, lmom3])


@njit
def pargev_numba(lmom):
    """
    pargev returns the parameters of the Generalized Extreme Value
    distribution given the L-moments of samples
    """
    t3 = lmom[2] / lmom[1]
    # Don't create a new array here - just use the values directly
    lmom_0 = lmom[0]
    lmom_1 = lmom[1]

    SMALL = 1e-5
    eps = 1e-6
    maxit = 20
    # Constants
    EU = 0.57721566
    DL2 = math.log(2)
    DL3 = math.log(3)
    # Coefficients for rational-function approximations
    A0 = 0.28377530
    A1 = -1.21096399
    A2 = -2.50728214
    A3 = -1.13455566
    A4 = -0.07138022
    B1 = 2.06189696
    B2 = 1.31912239
    B3 = 0.25077104
    C1 = 1.59921491
    C2 = -0.48832213
    C3 = 0.01573152
    D1 = -0.64363929
    D2 = 0.08985247

    # Check for valid L-moments
    if lmom_1 <= 0 or abs(t3) >= 1:
        # Create a new array for the result rather than modifying an input
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    para1 = 0.0
    para2 = 0.0
    para3 = 0.0

    if t3 <= 0:
        G = (A0 + t3 * (A1 + t3 * (A2 + t3 * (A3 + t3 * A4)))) / (
            1 + t3 * (B1 + t3 * (B2 + t3 * B3))
        )
        if t3 >= -0.8:
            para3 = G
            GAM = math.exp(math.lgamma(1 + G))
            para2 = lmom_1 * G / (GAM * (1 - 2**-G))
            para1 = lmom_0 - para2 * (1 - GAM) / G
            return np.array([para1, para2, para3], dtype=np.float64)

        elif t3 <= -0.97:
            G = 1 - math.log(1 + t3) / DL2
        T0 = (t3 + 3) * 0.5

        # Iteration loop
        for IT in range(maxit):
            X2 = 2**-G
            X3 = 3**-G
            XX2 = 1 - X2
            XX3 = 1 - X3
            T = XX3 / XX2
            DERIV = (XX2 * X3 * DL3 - XX3 * X2 * DL2) / (XX2**2)
            GOLD = G
            G -= (T - T0) / DERIV
            if abs(G - GOLD) <= eps * G:
                para3 = G
                GAM = math.exp(math.lgamma(1 + G))
                para2 = lmom_1 * G / (GAM * (1 - 2**-G))
                para1 = lmom_0 - para2 * (1 - GAM) / G
                return np.array([para1, para2, para3], dtype=np.float64)

        # If iteration doesn't converge, return NaN values
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    else:
        Z = 1 - t3
        G = (-1 + Z * (C1 + Z * (C2 + Z * C3))) / (1 + Z * (D1 + Z * D2))

        if abs(G) < SMALL:
            para2 = lmom_1 / DL2
            para1 = lmom_0 - EU * para2
            para3 = 0
        else:
            para3 = G
            GAM = math.exp(math.lgamma(1 + G))
            para2 = lmom_1 * G / (GAM * (1 - 2**-G))
            para1 = lmom_0 - para2 * (1 - GAM) / G

        return np.array([para1, para2, para3], dtype=np.float64)
