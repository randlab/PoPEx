# -*- coding: utf-8 -*-
""" `isample.neff` is a module that implements the most common importance
sampling diagnostics used in the PoPEx procedure. Namely these are

    - :meth:`ne`            Effective number of weights for estimating the
      expectation
    - :meth:`ne_var`        Effective number of weights for estimating the
      variance
    - :meth:`ne_gamma`      Effective number of weights for estimating the
      skewness
    - :meth:`alpha`         Optimization for finding the weight correction
      exponent
    - :meth:`correct_w`     Computes the set of corrected weights
"""

# -------------------------------------------------------------------------
#   Authors: Christoph Jaeggli, Julien Straubhaar and Philippe Renard
#   Year: 2018
#   Institut: University of Neuchatel
#
#   Copyright (c) 2018 Christoph Jaeggli
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

import warnings
import numpy as np
from scipy import optimize
import popex.popex_messages as msg
from popex.cnsts import NP_MIN_TOL, ALPHA_MIN, ALPHA_MAX


def ne(weights):
    """ `ne` computes the effective number of weights.

    Kish's effective number of weights is computed as

        `n_e(w_1, ..., w_n) = ( \sum w_i )^2 / ( \sum w_i^2 )`.


    Parameters
    ----------
    weights : ndarray, shape=(n,)
        Set of weights.

    Returns
    -------
    float
        Effective number of weights.

    """
    # Raise numpy over-/underflow errors
    old_settings = np.seterr(over='raise', under='raise')

    # Treat extreme values
    w = np.array(weights)
    try:
        w_sq = w ** 2
    except FloatingPointError as err:
        first_word = str(err).split()[0]
        if first_word == 'overflow':
            w[w >= NP_MIN_TOL] /= np.max(w)
            w[w < NP_MIN_TOL] = 0
        elif first_word == 'underflow':
            w[w < NP_MIN_TOL] = 0
        else:
            raise FloatingPointError(err)
        w_sq = w ** 2

    # Treat zero division
    if np.sum(w_sq) < NP_MIN_TOL:
        w = np.ones_like(w)
        w_sq = w
        warnings.warn(msg.warn1001)

    # Set numpy settings back
    np.seterr(**old_settings)

    # Return effective number of weights
    return (np.sum(w) ** 2) / np.sum(w_sq)


def ne_var(weights):
    """ `ne_var` computes the effective number of weights for estimating the
    variance.

    The effective number of weights for an empirical estimation of the variance
    is computed as:

        `n_e(w_1, ..., w_n) = ( \sum w_i^2 )^2 / ( \sum w_i^4 )`.


    Parameters
    ----------
    weights : ndarray, shape=(n,)
        Set of weights.

    Returns
    -------
    float
        Effective number of weights (for the variance).

    """
    # Raise numpy overflow errors
    old_settings = np.seterr(over='raise', under='raise')

    w = np.array(weights)
    if np.sum(w ** 2) < NP_MIN_TOL:
        w = np.ones_like(w)
        warnings.warn(msg.warn1001)

    # Set numpy settings back
    np.seterr(**old_settings)

    # Return effective number of weights
    return (np.sum(w ** 2) ** 2) / np.sum(w ** 4)


def ne_gamma(weights):
    """ `ne_gamma` computes the effective number of weights for the skewness.

    The effective number of weights for estimating the skewness is computed as:

        `n_e(w_1, ..., w_n) = ( \sum w_i^2 )^3 / ( ( \sum w_i^3 )^2 )`.


    Parameters
    ----------
    weights : ndarray, shape=(n,)
        Set of weights.

    Returns
    -------
    float
        Effective number of weights (for the skewness).

    """
    # Raise numpy overflow errors
    old_settings = np.seterr(over='raise', under='raise')

    w = np.array(weights)
    if np.sum(w ** 2) < NP_MIN_TOL:
        w = np.ones_like(w)
        warnings.warn(msg.warn1001)

    # Set numpy settings back
    np.seterr(**old_settings)

    # Return effective number of weights
    return (np.sum(w ** 2) ** 3) / (np.sum(w ** 3) ** 2)


def alpha(weights, theta, a_init=1):
    """ `alpha` computes the best `alpha` for lowering the skewness of the
    weights.


    Let `k_1` be the number of zero weights and `k_2` is the number of weights
    that attain the maximum value in 'weights'. For a given theta in the
    interval `(k_2, n-k_1)`, this functions finds the unique `alpha` such that

        `neff.ne(weights ** alpha) = theta`.

    The alpha is found by using the 'L-BFGS-B' optimization algorithm of
    `scipy.optimize` with initial value equal to `a_init`. We set upper and
    lower bounds to `ALPHA_MIN` and `ALPHA_MAX`, respectively. The maximum
    number of iterations is 15.

    Parameters
    ----------
    weights : ndarray, shape=(n,)
        Set of weights
    theta : float
        A positive value for the effective number of weights
    a_init : float
        Initial guess for the optimization problem


    Returns
    -------
    float
        `alpha` value

    """
    # Set numpy error behaviour
    old_settings = np.seterr(over='raise', under='raise')
    w = np.array(weights)
    # w[w < NP_MIN_TOL] = 0

    # Define optimization function
    def opt_fun(a):
        try:
            w_a = w ** a
        except FloatingPointError:
            if a < 1:
                return (np.sum(w > 0) - theta) ** 2
            else:
                return (np.sum(w == np.max(w)) - theta) ** 2
        return (ne(w_a) - theta) ** 2

    # Minimize optimization function
    nnz = np.sum(w > 0)
    nmax = np.sum(w == np.max(w))
    if theta > nmax:
        if theta < nnz:
            opt_obj = optimize.minimize(opt_fun, np.array([a_init]),
                                        method='L-BFGS-B',
                                        bounds=[(ALPHA_MIN, ALPHA_MAX)],
                                        options={'disp': False,
                                                 'maxiter': 10})

            a_ret = opt_obj['x'][0]
        else:
            a_ret = ALPHA_MIN
    else:
        a_ret = ALPHA_MAX
    # Set back to old settings
    np.seterr(**old_settings)
    return a_ret


def correct_w(weights, ne_w_corr):
    """ `correct_w` is a function that lowers the skewness of the weights.

    In addition to the method :meth:`alpha` it computes whether the correction
    is possible and treats the exceptions accordingly.


    Parameters
    ----------
    weights : ndarray, shape=(n,)
        Set of weights
    ne_w_corr : float
        A positive value for the effective number of weights


    Returns
    -------
    w_corr : ndarray, shape=(n,)
        Corrected set of weights that is also normalized

    """
    a_val = alpha(weights, ne_w_corr, a_init=1)

    if np.abs(a_val - ALPHA_MIN) < NP_MIN_TOL:
        # Set all positive value to 1
        w_corr = np.zeros_like(weights)
        w_corr[weights >= NP_MIN_TOL] = 1

    elif np.abs(a_val - ALPHA_MAX) < NP_MIN_TOL:
        # Set all values that attain the max to 1
        w_corr = np.zeros_like(weights)
        w_corr[weights == np.max(weights)] = 1

    else:
        # Compute the power weights
        w_corr = weights ** a_val

    # Check for division error
    if np.sum(w_corr) < NP_MIN_TOL:
        w_corr = np.ones_like(w_corr)
    w_corr /= np.sum(w_corr)

    return w_corr
