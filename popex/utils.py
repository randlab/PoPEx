# -*- coding: utf-8 -*-
""" `utils.py` contains utilities for performing a PoPEx sampling and computing
predictions:

Category probabilities and kld maps
    - :meth:`compute_cat_prob`:   Computes probability maps according to
      categories
    - :meth:`update_cat_prob`:    Updates probability maps according to
      categories
    - :meth:`compute_entropy`:    Computes entropy of a probability map
    - :meth:`compute_kld`:        Computes kld of two probability maps

Hard conditioning data
    - :meth:`generate_hd`:        Computes the new hard conditioning data
    - :meth:`merge_hd`:           Merges prior and new hard conditioning data
    - :meth:`compute_ncmod`:      Computes the number of conditioning points per
      model type
    - :meth:`compute_w_lik`:      Computes the likelihood weights (used for the
      hard conditioning maps)

Generic functions
    - :meth:`compute_w_pred`:     Computes the weights for the predictions
    - :meth:`compute_subset_ind`: Computes the smallest number of indices that
      cover a given percentage of a total weight
    - :meth:`write_hd_info`:      Writes/saves hd information about each model
    - :meth:`write_run_info`:     Appends information about models to run info
      file
"""

# -------------------------------------------------------------------------
#   Authors: Christoph Jaeggli, Julien Straubhaar and Philippe Renard
#   Year: 2019
#   Institut: University of Neuchâtel
#
#   Copyright (c) 2019 Christoph Jaeggli
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# General imports
import os
import errno
import warnings
import numpy as np
import pickle
from copy import deepcopy

# Package imports
import popex.isampl as isampl
import popex.popex_messages as msg
from popex.popex_objects import PoPEx, CatMType, ContParam, CatProb
from popex.cnsts import NP_MIN_TOL


# Category probabilities and kld maps
def compute_cat_prob(popex, weights, start=-1, stop=-1):
    """ `compute_cat_prob` computes the weighted category probabilities.

    The models are obtained from `popex.model` and weighted by `weights`.

    Parameters
    ----------
    popex : PoPEx
        PoPEx main structure (cf `popex_objects.PoPEx`)
    weights : ndarray, shape=(nmod,)
        Relative weights of the models
    start : int
        Defines the first model to take into account:

        - -1: For starting at 0
        - N: For starting at `max(N, 0)`
    stop : int
        Defines the last model to take into account

        - -1: For stopping at `popex.nmod`
        - N: For stopping at `min(N, popex.nmo)`

    Returns
    -------
    m-tuple
        A tuple of instances that describes the category probabilities for all
        categorical model types.

        If a model type i is not a subclass of ``CatMType``, the corresponding
        map is set to ``None``. If a model is given by (`CatModel_1`, ...,
        `CatModel_m`) and the model values in `CatModel_i` are subdivided into
        `ncat_i` categories, then the return value is a (`CatProb_1`, ...,
        `CatProb_m`) tuple where `return[i].param_val` is a ``ndarray`` with
        `shape=(nparam_i, ncat_i)`.

    """
    nmtype = popex.nmtype

    # Evaluate starting and ending point
    if start == -1:
        first = 0
    else:
        first = max(start, 0)
    if stop == -1:
        last = popex.nmod
    else:
        last = min(stop, popex.nmod)
    if first >= last:
        raise ValueError(msg.err1006(first, last))

    # Normalize weights
    if np.any(weights[first:last] < -NP_MIN_TOL):
        raise ValueError(msg.err1001)
    if np.sum(weights[first:last]) >= NP_MIN_TOL:
        w_nrm = weights[first:last] / np.sum(weights[first:last])
    else:
        w_nrm = np.ones(last - first) / (last - first)
        warnings.warn(msg.warn1001)
    if np.abs(np.sum(w_nrm[first:last]) - 1) > NP_MIN_TOL:
        raise ValueError(msg.err1007(first, last))

    # Instantiate p_cat
    mod = deepcopy(popex.model[first])
    if isinstance(mod, str):
        with open(popex.path_res + mod, 'rb') as mfile:
            mod = pickle.load(mfile)
    p_cat = [None for _ in range(nmtype)]
    cat_mtype = [imtype for imtype in range(nmtype)
                 if isinstance(mod[imtype], CatMType)]
    for imtype in cat_mtype:
        p_cat[imtype] = CatProb(param_val=np.zeros((mod[imtype].nparam,
                                                    mod[imtype].ncat)),
                                categories=mod[imtype].categories)

    # Compute category probabilities
    for imod in range(first, last):
        if imod > first:
            mod = deepcopy(popex.model[imod])
            if isinstance(mod, str):
                with open(popex.path_res + mod, 'rb') as mfile:
                    mod = pickle.load(mfile)
        for imtype in cat_mtype:
            for icat in range(mod[imtype].ncat):
                p_cat[imtype].param_val[:, icat] += \
                    (mod[imtype].param_cat == icat) * w_nrm[imod]
    return tuple(p_cat)


def update_cat_prob(p_cat, m_new, w_new, sum_w_old):
    """ `update_cat_prob` updates (in place) the category probabilities.

    If `p_cat_old` represents the old category probability maps, then we have

        `p_cat_new = [sum_w_old*p_cat_old + \sum_i w_new_i*1(m_new_i)]
        / [sum_w_old + sum(w_new)]`

    where `1(m_new_i)` is the categorical indicator of the model `i`.


    Parameters
    ----------
    p_cat : m-tuple
        Tuple of categorical probability maps (cf. :meth:`compute_cat_prob`)
    m_new : list
        List of m-tuples defining a set of `nmod` models
    w_new : ndarray, shape=(nmod,)
        Weights associated to the new models
    sum_w_old : float
        Old weight normalization constant


    Returns
    -------
    None

    """

    # Check input types
    if not isinstance(m_new, list):
        raise ValueError(msg.err1011)
    if not isinstance(w_new, np.ndarray):
        raise ValueError(msg.err1012)
    elif not len(m_new) == w_new.size:
        raise ValueError(msg.err1013)

    # Check positivity of weights
    if any(w < 0 for w in w_new):
        raise ValueError(msg.err1001)

    # p_cat stays unchanged if the new weight is zero
    sum_w_new = np.sum(w_new)
    nmtype = len(p_cat)
    if sum_w_new >= NP_MIN_TOL:
        # Generate new p_cat instance (based on new models)
        popex_new = PoPEx(model=m_new, nmtype=nmtype)
        p_cat_new = compute_cat_prob(popex_new, w_new)

        # Update weights
        w_p_cat = (sum_w_old / (sum_w_old+sum_w_new))
        w_p_cat_new = (sum_w_new / (sum_w_old+sum_w_new))

        # p_cat[imtype] may be 'None' (for unconditioned model types)
        cat_mtype = [imtype for imtype in range(nmtype)
                     if not p_cat[imtype] is None]

        # Update p_cat
        for imtype in cat_mtype:
            p_cat[imtype].param_val = p_cat[imtype].param_val*w_p_cat \
                                      + p_cat_new[imtype].param_val*w_p_cat_new
    return None


def compute_entropy(p_cat):
    """ `compute_entropy` computes the entropy of `p_cat`.

    The entropy of a discrete probability distributions `p = (p_1, ...,p_s)` is

        `H(p) = -\sum_{i=1}^s p_i log( p_i )`.

    Therefore, if the probability map `p_cat` is a ``m-tuple`` such that
    `p_cat[i].param_val` is an ``ndarray`` of `shape=(nparam_i, nfac_i)`, the
    entropy is also an ``m-tuple`` where `H[i].param_val` being an ``ndarray``
    of `shape=(nparam,)`.

    Notes
    -----
    Note that `t*log(t) -> 0` as `t -> 0`. Therefore, `H(x) = 0` wherever
    `p_i(x) = 0`.


    Parameters
    ----------
    p_cat : m-tuple
        Tuple of ``CatProb`` instances with `p_cat[i].param_val` being an
        ``ndarray`` of `shape=(nparam_i, nfac_i)`


    Returns
    -------
    m-tuple
        Tuple of entropy maps

        `return[i]` : ``None`` or ``ContParam``
            Return value `i` is ``None`` if `p_cat[i]` is ``None``, otherwise
            it is an instance of ``ContParam``

    """
    nmtype = len(p_cat)

    # Compute H only not-None probability maps
    H = [None for _ in range(nmtype)]
    cat_mtype = [imtype for imtype in range(nmtype)
                 if not p_cat[imtype] is None]

    # Iteratively compte H for each mtype
    for imtype in cat_mtype:
        H[imtype] = ContParam()
        p_i = p_cat[imtype].param_val

        # Compute KLD
        nparam, ncat = p_cat[imtype].param_val.shape
        H[imtype].param_val = np.zeros((nparam,))
        for icat in range(ncat):
            nnz_ind = p_i[:, icat] > NP_MIN_TOL
            H[imtype].param_val[nnz_ind]\
                -= p_i[nnz_ind, icat] * np.log(p_i[nnz_ind, icat])

    return tuple(H)


def compute_kld(p_cat, q_cat):
    """ `compute_kld` computes the Kullback-Leibler divergence (KLD) between
    two category probability maps `p_cat` and `q_cat`.

    The KLD between two discrete probability distributions `p = (p_1, ...,p_s)`
    and `q = (q_1, ...,q_s)` is

        `KLD(p||q) = \sum_{i=1}^s p_i log( p_i / q_i)`.

    Therefore, if the probability maps `p_cat` and `q_cat` are ``m-tuples`` such
    that `p_cat[i].param_val` and `q_cat[i].param_val` are ``ndarrays`` of
    `shape=(nparam_i, nfac_i)`, the Kullback-Leibler divergence is also an
    ``m-tuple`` where `kld[i].param_val` is an ``ndarray`` of `shape=(nparam,)`.


    Notes
    -----
    Note that `t*log(t/a) -> 0` as `t -> 0`. Therefore, we require that `q_i(x)
    = 0` implies `p_i(x) = 0` in which case we can put `kld(x) = 0`. However,
    due to the (inaccurate) numerical representation of the probability maps, it
    is possible that `q_i(x) = 0` and `p_i(x) > 0` (f.e. when `q` has been
    approximated from a relative small set of models). In this case we enforce
    `q_i(x) = p_i(x)` what leads to `kld(x) = 0`.


    Parameters
    ----------
    p_cat : m-tuple
        Tuple of ``CatProb`` instances with `p_cat[i].param_val` being an
        ``ndarray`` of `shape=(nparam_i x nfac_i)`
    q_cat : m-tuple
        Tuple of ``CatProb`` instances with `q_cat[i].param_val` being an
        ``ndarray`` of `shape=(nparam_i, nfac_i)`


    Returns
    -------
    m-tuple
        Tuple of kld maps

        `return[i]` : ``None`` or ``ContParam``
            Return value `i` is ``None`` if `p_cat[i]`` is ``None``, otherwise
            it is an instance of ``ContParam``
    """
    nmtype = len(p_cat)

    # Compute kld only for not-None probability maps
    kld = [None for _ in range(nmtype)]
    cat_mtype = [imtype for imtype in range(nmtype)
                 if not p_cat[imtype] is None]

    # Iteratively compute kld for each mtype
    for imtype in cat_mtype:
        kld[imtype] = ContParam()
        p_i = p_cat[imtype].param_val
        q_i = q_cat[imtype].param_val
        # Check for "corrupted" prior probability map q
        corr_ind = np.where(
            np.logical_and(q_i <= NP_MIN_TOL, p_i > NP_MIN_TOL))[0]
        if corr_ind.size > 0:
            q_i[corr_ind, :] = p_i[corr_ind, :]
            warnings.warn(msg.warn1002(imtype))

        # Compute KLD
        nparam, ncat = p_cat[imtype].param_val.shape
        kld[imtype].param_val = np.zeros((nparam,))
        for icat in range(ncat):
            nnz_ind = p_i[:, icat] > NP_MIN_TOL
            kld[imtype].param_val[nnz_ind]\
                += p_i[nnz_ind, icat] *\
                   np.log(p_i[nnz_ind, icat] / q_i[nnz_ind, icat])

        # Correct (theoretically impossible) negative kld values
        kld[imtype].param_val[np.abs(kld[imtype].param_val) < NP_MIN_TOL] = 0
        if np.any(kld[imtype].param_val < -NP_MIN_TOL):
            ind = np.where(kld[imtype].param_val < -NP_MIN_TOL)[0]
            raise ValueError(msg.err1002(imtype, ind))
    return tuple(kld)


# Hard conditioning data
def generate_hd(popex, meth_w_hd, ncmod, kld, p_cat, q_cat):
    """ `generate_hd` generates the hard conditioning data set that is used
    to sample a new model.

    This set of hard conditioning data does NOT include prior hard conditioning.
    For each model type `(imtype)`, every hard conditioning is obtained by the
    following 2-steps:

        (a) Sample a location `[j]` according to the values in the Kullback-
            Leibler divergence map (i.e. the values in `kld[imtype].param_val`)
        (b) Sample a model `[k]` according to the weights from
            :meth:`compute_w_lik` and directly extract the hard conditioning
            value from `popex.model[k][imtype].param_val[j]`.

    In addition to the hard conditioning, this function also extracts
    probability values from `q_cat` and `p_cat` at the conditioning location.
    These values represent the prior/weighted category probability of the
    category that corresponds to `popex.model[k][imtype].param_val[j]`. They
    can be useful to compute the sampling weight ratio.

    Notes
    -----
    There are two important things to note:

        (1) The two objects `hd_prior` and `hd_generation` are the corresponding
            prior and weighted probability values of the hard conditioning
            CATEGORY that corresponds to the values in `hd_param_val`.
            Therefore, if they are used in the computation of the sampling
            weight ratio, one uses CATEGORY probabilities and NOT value
            probabilities.

        (2) Numerical imperfections (for example in the computation of 'q_cat')
            can cause locations where `p_cat > 0` but `q_cat = 0`. In the
            computation of the Kullback-Leibler divergence we did put
            corresponding `kld` values to `0` (by enforcing `q_i(x) = p_i(x)`)
            and therefore it is impossible to sample and condition such
            locations.


    Parameters
    ----------
    popex : PoPEx
        PoPEx main structure
    meth_w_hd : dict
        Method to compute hard conditioning weights (cf. :meth:`compute_w_lik`)
    ncmod : m-tuple
        Number of conditioning points per model type
    kld : m-tuple
        Tuple of ``ContParam`` instances defining the Kullback-Leibler
        divergence
    p_cat : m-tuple
        Tuple of ``CatProb`` instances defining the weighted category
        probabilities with `p_cat[i].param_val` being an ``ndarray`` of
        `shape=(nparam_i, nfac_i)`
    q_cat : m-tuple
        Tuple of ``CatProb`` instances defining the weighted category
        probabilities with `q_cat[i].param_val` being an ``ndarray`` of
        `shape=(nparam_i x nfac_i)`


    Returns
    -------
    hd_param_ind : m-tuple
        Tuple of hard conditioning indices where hard conditioning values are
        imposed. If there is no hard conditioning for a model type `i`, then
        `hd_ind[i]` is ``None`` otherwise it is an ``ndarray`` of
        `shape=(ncmod[i], ndarray)`.
    hd_param_val : m-tuple
        Tuple of hard conditioning values that are imposed at the hard
        conditioning indices. If there is no hard conditioning for a model type
        `i`, then `hd_val[i]` is ``None`` otherwise it is an ``ndarray`` of
        `shape=(ncmod[i], ndarray)`.
    hd_prior :  m-tuple
        Tuple of probability values according to the prior probability maps in
        `q_cat`. Each value corresponds to the prior probability of the category
        that contains the extracted hard conditioning value.  If there is no
        hard conditioning for a model type `i`, then `hd_val[i]` is ``None``
        otherwise it is an ``ndarray`` of `shape=(ncmod[i], ndarray)`.
    hd_generation : m-tuple
        Tuple of probability values according to the sampling probability maps
        in `p_cat`. Each value corresponds to the sampling probability of the
        category that contains the extracted hard conditioning values. If there
        is no hard conditioning for a model type `i`, then `hd_val[i]` is
        ``None`` otherwise it is an ``ndarray`` of `shape=(ncmod[i], ndarray)`.
    """
    # Raise all numpy errors
    old_settings = np.seterr(all='raise')

    # Initialisation
    nmtype = popex.nmtype
    ncmod = list(ncmod)
    hd_mtype = [imtype for imtype, nc_i in enumerate(ncmod) if nc_i > 0]
    hd_param_ind = [None for _ in range(nmtype)]
    hd_param_val = [None for _ in range(nmtype)]
    hd_prior = [None for _ in range(nmtype)]
    hd_generation = [None for _ in range(nmtype)]

    # STEP (a) (see remark in docstring)
    # Choose hard conditioning locations
    for imtype in hd_mtype:
        # Test KLD values
        if np.any(kld[imtype].param_val < -NP_MIN_TOL):
            ind = np.where(kld[imtype].param_val < -NP_MIN_TOL)[0]
            raise ValueError(msg.err1002(imtype, ind))
        if np.sum(kld[imtype].param_val) == 0:
            ncmod[imtype] = 0

        # Normalize kld values
        try:
            kld_val = kld[imtype].param_val
            nnz_ind = kld[imtype].param_val >= NP_MIN_TOL

            kld_nrm = np.zeros_like(kld_val)
            kld_nrm[nnz_ind] = kld_val[nnz_ind] / np.sum(kld_val[nnz_ind])
        except (ZeroDivisionError, FloatingPointError,
                RuntimeWarning, RuntimeError):
            kld_nrm = np.ones_like(kld[imtype].param_val)\
                       / kld[imtype].nparam
            warnings.warn(msg.warn1001)

        # Compute conditioning location indices
        hd_param_ind[imtype] = \
            np.unique(
                np.random.choice(kld[imtype].nparam,
                                 size=ncmod[imtype],
                                 replace=True,
                                 p=kld_nrm))
        # We put 'replace=True' in order to allow nc_i to be larger than
        # nparam. This can be convenient for the client when nparam is very
        # small. However, multiple indices are removed by using 'np.unique'
        ncmod[imtype] = hd_param_ind[imtype].size

    # Adapt the variables
    hd_mtype = [imtype for imtype, nc_i in enumerate(ncmod) if nc_i > 0]
    hd_param_ind = tuple(hd_param_ind)

    # STEP (b) (see remark in docstring)
    # Compute normalized w_hd
    w_hd_nrm = compute_w_lik(popex=popex, meth=meth_w_hd)

    # Choose hard conditioning values
    for imtype in hd_mtype:
        # Initialize
        hd_param_val[imtype] = []
        hd_prior[imtype] = []
        hd_generation[imtype] = []

        # Select conditional value sequentially
        nc_i = ncmod[imtype]
        if not nc_i == hd_param_ind[imtype].size:
            raise ValueError(msg.err1009)
        for ihd in range(nc_i):
            # Select model according to the 'w_hd' values
            imod = int(np.random.choice(popex.nmod, 1, p=w_hd_nrm))
            hd_ind = hd_param_ind[imtype][ihd]

            # Load pickled model and extract hd value
            path_mod = popex.path_res + popex.model[imod]
            with open(path_mod, 'rb') as mfile:
                model = pickle.load(mfile)
            hd_param_val[imtype].append(model[imtype].param_val[hd_ind])

            # Extract the corresponding category probability value
            icat = model[imtype].param_cat[hd_ind]
            hd_prior[imtype].append(q_cat[imtype].param_val[hd_ind, icat])
            hd_generation[imtype].append(
                p_cat[imtype].param_val[hd_ind, icat])

        # Transform ...[imtype] into ndarray
        hd_param_val[imtype] = np.array(hd_param_val[imtype])
        hd_prior[imtype] = np.array(hd_prior[imtype])
        hd_generation[imtype] = np.array(hd_generation[imtype])

        # Test proability values
        if np.any(hd_prior[imtype] == 0)\
                or np.any(hd_generation[imtype] == 0):
            raise ValueError(msg.err1008)

    # Transform into m-tuples
    hd_param_val = tuple(hd_param_val)
    hd_prior = tuple(hd_prior)
    hd_generation = tuple(hd_generation)

    # Set numpy settings back
    np.seterr(**old_settings)
    return hd_param_ind, hd_param_val, hd_prior, hd_generation


def merge_hd(hd_param_ind_1, hd_param_ind_2, hd_param_val_1, hd_param_val_2):
    """ `merge_hd` used for merging two sets of hard conditioning data.

    It is assumed that `hd_param_ind_i[imtype]` is ``None`` if and only if
    `hd_param_val_i[imtype]` is ``None``.


    Parameters
    ----------
    hd_param_ind_1 : m-tuple
        First set of hard conditioning indices
    hd_param_ind_2 : m-tuple
        Second set of hard conditioning indices
    hd_param_val_1 : m-tuple
        First set of hard conditioning values
    hd_param_val_2 : m-tuple
        Second set of hard conditioning values


    Returns
    -------
    hd_ind : m-tuple
        Merged set of hard conditioning indices
    hd_par : m-tuple
        Merged set of hard conditioning values

    """
    nmtype = len(hd_param_ind_1)

    # Initialisation
    hd_param_ind = [None for _ in range(nmtype)]
    hd_param_val = [None for _ in range(nmtype)]

    # Merge data for each model type
    for imtype in range(nmtype):
        if hd_param_ind_1[imtype] is not None \
                and hd_param_ind_2[imtype] is not None:
            hd_param_ind[imtype] = np.append(hd_param_ind_1[imtype],
                                             hd_param_ind_2[imtype])
            hd_param_val[imtype] = np.append(hd_param_val_1[imtype],
                                             hd_param_val_2[imtype])
        elif hd_param_ind_1[imtype] is None \
                and hd_param_ind_2[imtype] is not None:
            hd_param_ind[imtype] = hd_param_ind_2[imtype]
            hd_param_val[imtype] = hd_param_val_2[imtype]
        elif hd_param_ind_1[imtype] is not None \
                and hd_param_ind_2[imtype] is None:
            hd_param_ind[imtype] = hd_param_ind_1[imtype]
            hd_param_val[imtype] = hd_param_val_1[imtype]
    return tuple(hd_param_ind), tuple(hd_param_val)


def compute_ncmod(popex, meth_w_hd=None):
    """ `compute_ncmod` computes, for each model type, the number of
    conditioning points.


    It is assumed that the number of hard data is restricted model type-wise.
    Therefore, the number of conditioning points is also computed model
    type-wise by sampling from an uniform random variable
    `~U(0, popex.ncmax[imtype])`.


    Notes
    -----
    Note that if the total sum of the likelihood values in `popex.p_lik` is
    zero, `ncmod` is set to zero for each model type.


    Parameters
    ----------
    popex : PoPEx
        PoPEx main structure
    meth_w_hd : dict
        Method to compute hard conditioning weights (cf. :meth:`compute_w_lik`)


    Returns
    -------
    m-tuple
        Number of conditioning points per model type
    """

    # Weights according to 'hd_meth'
    w_hd = compute_w_lik(popex=popex, meth=meth_w_hd)

    # Set all ncmod equal to zero for too small weights
    if np.sum(w_hd) >= NP_MIN_TOL:
        nc_bnd = popex.ncmax
    else:
        nc_bnd = (0,) * popex.nmtype
    ncmod = tuple(np.random.randint(nc_bnd[imtype] + 1, size=1)[0]
                  for imtype in range(popex.nmtype))
    return ncmod


def compute_w_lik(popex, meth=None):
    """ `compute_w_lik` returns the set of normalized likelihood values.

    In practice, when the likelihood values must be represented by a floating
    point number, it might be advantageous to compute approximations of `L(m)`.

    There are several approximation possibilities that are implemented in this
    version (specified in `meth`):

        (a) No approximation (`meth={'name': 'exact'}` or `meth=None`):

            `L(m) = exp( 'log_p_lik' )`

        (b) Sqrt-unskewed (`meth={'name': 'exp_sqrt_log'}`)

            `L(m) ~ exp( -sqrt(-'log_p_lik' )`

        (c) K-unskewed (`meth={'name': 'exp_sqrt_log', 'pow': k}`)

            `L(m) ~ exp( - (-'log_p_lik')^k )`

        (d) Inverse log (`meth={'name': 'inv_log'}`)

            `L(m) ~ 1 / ( 1-'log_p_lik' )`

        (e) Inverse sqrt-log (`meth={'name': 'inv_sqrt_log'}`)

            `L(m) ~ 1 / ( 1+sqrt(-'log_p_lik') )`.

    As mentioned above, these techniques aim to unskew the likelihood values.


    Notes
    -----
    This function is used in two different locations (with possibly two
    different approximation techniques): for the learning
    scheme in the PoPEx sampling and for computing predictions. While in first
    case any approximation technique can be used, the latter choice might bias
    the computation weights.


    Parameters
    ----------
    popex : PoPEx
        PoPEx main structure
    meth : dict
        Defines the approximation method to be used. Fields are

            - ``'name'`` : Name of the method (`str`)
            - ``'pow'`` : Power for method (c) (`float`)


    Returns
    -------
    ndarray, shape=(nmod,)
        Array of normalized weights

    """

    log_w_lik = np.array(popex.log_p_lik)
    if log_w_lik.size == 0:
        return np.zeros_like(log_w_lik)

    if meth is None:
        name = 'exact'
    else:
        name = meth['name']

    # Weight quantities according to 'p_lik'
    if name == 'exact':
        # Shift log-weights (possible because of the normalization at the end)
        log_w_lik -= np.max(log_w_lik)

        # Compute exp for sufficiently large log-values
        w_lik = np.zeros_like(log_w_lik)
        nnz_ind = log_w_lik >= np.log(NP_MIN_TOL)
        w_lik[nnz_ind] = np.exp(log_w_lik[nnz_ind])

    elif name == 'exp_sqrt_log':
        # Shift log-weights (possible because of the normalization at the end)
        log_w_lik -= np.max(log_w_lik)

        # Bring the weights "closer together"
        log_w_lik = -np.sqrt(-log_w_lik)

        # Compute exp for sufficiently large log-values
        w_lik = np.zeros_like(log_w_lik)
        nnz_ind = log_w_lik >= np.log(NP_MIN_TOL)
        w_lik[nnz_ind] = np.exp(log_w_lik[nnz_ind])

    elif name == 'exp_pow_log':
        # Shift log-weights (possible because of the normalization at the end)
        log_w_lik -= np.max(log_w_lik)

        # Bring the weights "closer together"
        log_w_lik = -((-log_w_lik)**(meth['pow']))

        # Compute exp for sufficiently large log-values
        w_lik = np.zeros_like(log_w_lik)
        nnz_ind = log_w_lik >= np.log(NP_MIN_TOL)
        w_lik[nnz_ind] = np.exp(log_w_lik[nnz_ind])

    # Weight quantities according to 'log_p_lik'
    elif name == 'inv_log':
        # Check for all-negative log(w)
        pos_ind = log_w_lik >= NP_MIN_TOL
        if np.any(pos_ind):
            ind = np.where(pos_ind)
            raise ValueError(msg.err1010(ind))

        # Compute 1/(1-log(w))
        w_lik = 1 / (1-log_w_lik)

    # Weight quantities according to sqrt(-'log_p_lik')
    elif name == 'inv_sqrt_log':
        # Check for all-negative log(w)
        pos_ind = log_w_lik >= NP_MIN_TOL
        if np.any(pos_ind):
            ind = np.where(pos_ind)
            raise ValueError(msg.err1010(ind))

        # Compute 1/( 1-log( abs(w)^1/2 ) )
        w_lik = 1 / (1+np.sqrt(-log_w_lik))

    else:
        raise ValueError('\nUnknown method "{}" for computing weights'
                         .format(meth))

    # Remove numerical inaccuracy
    w_lik[w_lik < NP_MIN_TOL] = 0

    # Check weights for non-negativity
    if np.any(w_lik < -NP_MIN_TOL):
        ind = np.where(w_lik < -NP_MIN_TOL)[0]
        raise ValueError(msg.err1003(ind))

    # Weights must be normalized, because
    #   - of the shift when using meth['name'] = 'exact' (see above)
    #   - 'utils.generate_hd' assumes normalized weights
    try:
        nnz_ind = w_lik >= NP_MIN_TOL
        w_lik_nrm = np.zeros_like(w_lik)
        w_lik_nrm[nnz_ind] = w_lik[nnz_ind] / np.sum(w_lik[nnz_ind])
    except (ZeroDivisionError, FloatingPointError,
            RuntimeWarning, RuntimeError):
        w_lik_nrm = np.ones_like(popex.log_p_lik) / popex.nmod
        warnings.warn(msg.warn1004)

    return w_lik_nrm


# Generic functions
def compute_w_pred(popex, nw_min=0, ibnd=-1, meth=None):
    """ `compute_w_pred` returns the set of normalized predictive weights.

    For assuring a minimum number of effective weights, they are computed such
    that

        `ne(w_pred) = min(nw_min, ne(w))`

    where `w` contains the weights associated to the models and `ne(w)` denotes
    the number of effective weights. This quantity can be modified by replacing
    `w` with `w^\alpha`, where `alpha > 0`. A `1-d` optimisation problem is used
    to compute the optimal `\alpha` value.


    Parameters
    ----------
    popex : PoPEx
        PoPEx main structure
    nw_min : int
        Mininum number of effective weights `(= l_0)`
    ibnd : int
        Length of the weight array
    meth : dict
        Defines the approximation method to be used (cf. :meth:`compute_w_lik`).
        Fields are

            - ``'name'`` : Name of the method (`str`)
            - ``'pow'`` : Power for method (c) (`float`)


    Returns
    -------
    ndarray, shape=(nmod,)
        Array of predictive weights

    """

    # Compute weight array
    w_lik = compute_w_lik(popex=popex, meth=meth)
    ratio = np.exp(popex.log_p_pri-popex.log_p_gen)
    weights = w_lik * ratio

    # Adapt the weights by the power method
    if ibnd > 0:
        weights = weights[:ibnd]

    if np.sum(weights > NP_MIN_TOL) > 0:
        ne_w = isampl.ne(weights)
        # ne_w_pred = nw_min + ne_w
        ne_w_pred = np.max((nw_min, ne_w))
        w_pred = isampl.correct_w(weights, ne_w_pred)
    else:
        w_pred = np.ones_like(weights) / weights.size

    return w_pred


def compute_subset_ind(p_frac, weights):
    """ `compute_subset_ind` computes the smallest index set that covers a given
    percentage.

    This means that the subset indices `ind` are such that

        `np.sum(weights[ind]) >= p_frac * np.sum(weights)`,

    or in other words `weights[ind]` covers at least a fraction of `p_frac` of
    the total some of `weights`.


    Parameters
    ----------
    p_frac : float
        Coverage fraction in `(0, 1]`
    weights : ndarray, shape=(nw,)
        Non-negative weights


    Returns
    -------
    list
        Subset of indices

    """
    # Compute cumsum with largest weights first
    w_sorted = np.sort(weights)[::-1]
    w_cs = np.cumsum(w_sorted)

    # Compute treshold weight value (for p_frac)
    val = w_sorted[np.where(w_cs / w_cs[-1] - p_frac >= -NP_MIN_TOL)[0][0]]

    return [ind for ind, w in enumerate(weights) if w >= val]


def write_hd_info(popex, imod, hd_param_ind, hd_param_val):
    """ `write_hd_info` writes the hard conditioning that has been deduced
    for creating a specific model to a text file.

    | The text file is saved at `popex.path_res` with the following structure:
    |   <popex.path_res>$
    |       └-- hd
    |           └-- hd_modXXXXXX.txt


    Parameters
    ----------
    popex : PoPEx
        PoPEx main structure
    imod : int
        Model index
    hd_param_ind : m-tuple
        Hard conditioning indices
    hd_param_val : m-tuple
        Hard conditioning values


    Returns
    -------
    None

    """
    # Path of the result location
    path_res = popex.path_res
    nmtype = popex.nmtype

    # Make hd directory
    path_hd = path_res + 'hd/'
    try:
        os.mkdir(path_hd)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # For each model, write the precise hard conditioning
    with open('{}hd_mod{:06}.txt'.format(path_hd, imod), 'w+') as file:
        for imtype in range(nmtype):
            file.write('mtype {:02}\n'.format(imtype) + '-'*8 + '\n')
            if hd_param_ind[imtype] is not None:
                nc_i = hd_param_ind[imtype].size
                file.write('nc = {:3d}\n'.format(nc_i))
                file.write('Index   Value\n')
                mat = np.concatenate((hd_param_ind[imtype].reshape(-1, 1),
                                      hd_param_val[imtype].reshape(-1, 1)),
                                     axis=1)
                for li in mat:
                    val, pos = str(li)[1:-1].split()
                    file.write(' {:<7} {}'.format(val, pos) + '\n')
            else:
                file.write('None\n')
            file.write('\n')
    return None


def write_run_info(pb, popex, imod, log_p_lik, cmp_log_p_lik,
                   log_p_pri, log_p_gen, ncmod):
    """ `write_run_info` writes some algorithm specific information to a text
    file.

    | The text file is save at `popex.path_res` with the following structure:
    |   <popex.path_res>$
    |       └-- run_info.txt


    Parameters
    ----------
    pb : Problem
        Defines the problem functions and parameters
    popex : PoPEx
        PoPEx main structure
    imod : int
        Model index
    log_p_lik : float
        Log-likelihood value of the model
    cmp_log_p_lik : bool
        Indicates if likelihood has been computed (True) or predicted (False)
    log_p_pri : float
        Prior log-probability of the model
    log_p_gen : float
        Sampling log-probability of the model
    ncmod : m-tuple
        Model type specific number of conditioning points used


    Returns
    -------
    None

    """
    # Path of the result location
    path_res = popex.path_res
    nmtype = popex.nmtype

    # Compute n_e diagnostics
    with warnings.catch_warnings(record=True) as _:
        ne_l    = isampl.ne(compute_w_lik(popex=popex,
                                          meth={'name': 'exact'}))
        ne_w_hd = isampl.ne(compute_w_lik(popex=popex,
                                          meth=pb.meth_w_hd))
        ne_w_l  = isampl.ne(compute_w_pred(popex=popex,
                                           nw_min=0,
                                           meth={'name': 'exact'}))

    # Write into a file 'iteration_info.txt'
    if write_run_info.__first_call:
        mode = 'w+'
        write_run_info.__first_call = False
    else:
        mode = 'a'
    with open('{}/run_info.txt'.format(path_res), mode) as file:

        # Write model details
        file.write('Model {:6d}\n'.format(imod) + '-' * 12 + '\n')
        file.write('\n')
        file.write('  Hard conditioning\n')
        for imtype in range(nmtype):
            file.write('    mtype {:2d}:  nc = {:3d}\n'
                       .format(imtype, ncmod[imtype]))
        file.write('\n')
        file.write('  Computations\n')
        file.write('    log(p_lik)  = {:17.7f} (cmp = {!r})\n'
                   .format(log_p_lik, cmp_log_p_lik))
        file.write('    log(p_pri)  = {:17.7f}\n'.format(log_p_pri))
        file.write('    log(p_gen)  = {:17.7f}\n'.format(log_p_gen))
        file.write('    weight      = {:17.7f}\n'
                   .format(np.exp(log_p_lik + log_p_pri - log_p_gen)))

        # Write n_e
        file.write('\n')
        file.write('  Diagnostics\n')
        file.write('    ne(L)       = {:11.1f}\n'.format(ne_l))
        file.write('    ne(w_hd)    = {:11.1f}\n'.format(ne_w_hd))
        file.write('    ne(w_pred)  = {:11.1f}\n'.format(ne_w_l))
        file.write('\n\n')
write_run_info.__first_call = True
