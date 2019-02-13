# -*- coding: utf-8 -*-
""" `algorithm.py` contains the main implementation of the parallel (multiple-
chain) PoPEx algorithm. Mainly this concerns the two functions

    - :meth:`run_popex_mp`: Main implementation for PoPEx runs
    - :meth:`pred_popex_mp`: Main implementation for PoPEx predictions

that are used to sample models and predict results. A common file structure for
a PoPEx run together with some predictions is the following:

    | <path_res>$
    |  ├-- hd
    |     └-- hd_modXXXXXX.txt
    |  ├-- model
    |     └-- modXXXXXX.mod
    |  ├-- solution
    |      ├-- run_<name>_modXXXXXX
    |      └-- pred_<name>_modXXXXXX
    |  ├-- run_info.txt
    |  ├-- run_progress.txt
    |  ├-- pred_progress.txt
    |  ├-- popex.pop
    |  └-- problem.pb

The content of the different files is:

    - `hd_modXXXXXX.txt`:             Hard conditioning (without prior hd)
    - `modXXXXXX.mod`:                Pickled model object
    - `run_<name>_modXXXXXX.txt`:     Info from forward operator
    - `pred_<name>_modXXXXXX.txt`:    Info from prediction operator
    - `run_info.txt`:                 Info about the sampling
    - `run_progress.txt`:             Progress summary about the sampling
    - `pred_progress.txt`:            Progress summary about the predictions
    - `popex.pop`:                    Pickled PoPEx object
    - `problem.pb`:                   Pickled Problem object
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
import warnings
import pickle
import time
import errno
import os
import numpy as np
import multiprocessing
from copy import deepcopy
from collections import deque
import bisect

# Package imports
import popex.isampl as isampl
import popex.popex_messages as msg
import popex.utils as utl
from popex.popex_objects import PoPEx, CatProb
from popex.cnsts import NP_MIN_TOL, RECMP_P_CAT_FREQ


def run_popex_mp(pb, path_res, path_q_cat,
                 ncmax=(20,), nmp=1, nmax=1000,
                 upd_hdmap_freq=1, upd_ls_freq=-1, si_freq=-1):
    """ `run_popex_mp` is the main implementation of the PoPEx algorithm.

    The algorithm expands a set of models until the defined stopping condition
    is fulfilled. A pre-requirement of the method is that we know a prior
    probability map that corresponds to a given set of categories (c.f.
    `q_cat`).


    Notes
    -----
    The prior and generation probability values should NEVER be considered as
    true probability values according to the true distribution. They should
    only be used in combination (as ratio r):

            `ratio(m) = rho(m) / phi(m)`.

    This ratio is important in the importance sampling framework where we want
    to compute weighted expectation values according to a set of generated
    models.


    Parameters
    ----------
    pb : Problem
        Defines the problem functions and parameters
    path_res : str
        Path to the 'results' folder
    path_q_cat : str
        Path to the prior probability maps `q_cat` such that the tuple of
        `nmtype` objects can be loaded under '<path_q_cat>q_cat.prob'
    ncmax : m-tuple
        Maximal number of conditioning points for each model type
    nmp : int
        Number of parallel processes to use
    nmax : int
        Number of maximal models (stopping condition)
    upd_hdmap_freq : int
        Defines the frequency for updating the HD maps (`kld` and `p_cat`)
    upd_ls_freq : int
         Defines the frequency for updating the learning scheme (-1 for no
         update)
    si_freq : int
         Defines the frequency of saving intermediate states (-1 for no
         intermediate saves)


    Returns
    -------
    None

    """

    # Initialization
    nmtype = pb.nmtype
    tst_run = time.time()
    popex = PoPEx(ncmax=ncmax, nmtype=nmtype, path_res=path_res)
    hd_prior_param_ind, hd_prior_param_val = pb.get_hd_pri()
    cond_mtype = [imtype for imtype in range(nmtype)
                  if ncmax[imtype] > 0]

    # Write run information to standard output
    print("\n  START 'RUN_POPEX_MP'\n")
    print('    ncmax   = {!r}'.format(ncmax))
    print('    nmax    = {:>6d}'.format(nmax))
    print('    nmp     = {:>6d}'.format(nmp))
    print("    hd_meth = '{}'\n".format(pb.meth_w_hd['name']))

    # Generate 'model' folder
    try:
        os.mkdir('{}model/'.format(path_res))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # The map 'q_cat' must be a nmtype-tuple of 'CatProb' or 'None' objects
    print('    > load q_cat...', end='')
    with open('{}q_cat.prob'.format(path_q_cat), 'rb') as file:
        q_cat = pickle.load(file)
    for imtype in cond_mtype:
        if not isinstance(q_cat[imtype], CatProb):
            raise TypeError(imtype, type(q_cat[imtype]))
    print('done')

    # The kld map is zero initially (no conditioning data is selected)
    p_cat = deepcopy(q_cat)
    kld = utl.compute_kld(p_cat, q_cat)

    # Main PoPEx initialisations
    stop = False
    isim = 0
    sim_id_not_upd = []
    m_not_upd = []
    proc_mngr = deque([], maxlen=nmp)
    sim_id_mngr = []
    print('    > sample space...', end='')

    # Start PoPEx sampling by a pool of working processes
    with multiprocessing.Pool(processes=nmp, maxtasksperchild=100) as pool:
        while not stop:
            nmngr = len(proc_mngr)
            if nmngr < nmp and popex.nmod + nmngr < nmax:
                # Create and start a new working process
                args = (deepcopy(pb), deepcopy(popex), deepcopy(isim),
                        hd_prior_param_ind, hd_prior_param_val,
                        deepcopy(kld), deepcopy(p_cat), q_cat)
                proc_mngr.append(pool.apply_async(_run_process, args))
                isim += 1

            # Check if the process at the head of the queue has finished
            if proc_mngr[0].ready():
                # Remove process and get results
                proc = proc_mngr.popleft()
                (imod, model, ncmod, log_p_lik,
                 cmp_log_p_lik, log_p_pri, log_p_gen) = proc.get()

                # Extend control arrays for the update of 'p_cat'
                loc_upd = bisect.bisect(sim_id_not_upd, imod)
                sim_id_not_upd.insert(loc_upd, imod)
                m_not_upd.append(deepcopy(model))

                # Insert the model in the PoPEx structure
                loc_mngr = bisect.bisect(sim_id_mngr, imod)
                sim_id_mngr.insert(loc_mngr, imod)
                popex.insert_model(loc_mngr, imod, model, log_p_lik,
                                   cmp_log_p_lik, log_p_pri, log_p_gen, ncmod)

                # Write some information
                t_run = time.time() - tst_run
                t_mod = t_run / popex.nmod
                utl.write_run_info(pb, popex, imod, log_p_lik, cmp_log_p_lik,
                                   log_p_pri, log_p_gen, ncmod)
                _write_run_sum(pb, popex, nmax, t_run, t_mod)

                # Update 'kld' and 'p_cat' (if required)
                if len(cond_mtype) > 0:

                    if np.mod(popex.nmod, RECMP_P_CAT_FREQ) == 0:
                        # Frequently renew 'p_cat' entirely
                        w_hd = utl.compute_w_lik(popex=popex,
                                                 meth=pb.meth_w_hd)
                        p_cat = utl.compute_cat_prob(popex, w_hd)

                        # Compute the kld map with the new 'p_cat' map
                        kld = utl.compute_kld(p_cat, q_cat)

                        # 'p_cat' is "up to date" so we clean the arrays
                        sim_id_not_upd = []
                        m_not_upd = []

                    elif np.mod(popex.nmod, upd_hdmap_freq) == 0:
                        # Update 'p_cat' (which is much faster)
                        w_hd = utl.compute_w_lik(popex=popex,
                                                 meth=pb.meth_w_hd)
                        ind_not_upd = \
                            [i for i, id in enumerate(sim_id_mngr)
                             if id in sim_id_not_upd]
                        w_not_upd = w_hd[ind_not_upd]
                        sum_w_old = np.sum(w_hd) - np.sum(w_not_upd)
                        utl.update_cat_prob(p_cat, m_not_upd,
                                            w_not_upd, sum_w_old)

                        # Compute the kld map with the updated 'p_cat' map
                        kld = utl.compute_kld(p_cat, q_cat)

                        # 'p_cat' is "up to date" so we clean the arrays
                        sim_id_not_upd = []
                        m_not_upd = []

                # Update 'learning_scheme' (if required)
                if pb.learning_scheme and cmp_log_p_lik:
                    n_cmp = np.sum(popex.cmp_log_p_lik)
                    if all((upd_ls_freq > 0, np.mod(n_cmp, upd_ls_freq)==0)):
                        pb.learning_scheme.train(popex)

                # Update stopping condition
                if popex.nmod >= nmax:
                    stop = True

                # Save intermediate PoPEx result
                if all([si_freq > 0,
                        np.mod(popex.nmod, si_freq) == 0,
                        stop is False]):
                    with open('{}popex.pop'.format(popex.path_res), 'wb')\
                            as file:
                        pickle.dump(popex, file)

            # Otherwise send it to the back of the queue
            else:
                proc_mngr.rotate(-1)
    print('done')

    # Save final PoPEx result
    print('    > save PoPEx set...', end='')
    with open('{}popex.pop'.format(popex.path_res), 'wb') as file:
        pickle.dump(popex, file)
    print('done')

    # Save Problem
    print('    > save problem...', end='')
    with open('{}problem.pb'.format(popex.path_res), 'wb') as file:
        pickle.dump(pb, file)
    print('done')

    print("\n  END 'RUN_POPEX_MP'\n")


def _run_process(pb, popex, imod,
                 hd_prior_param_ind, hd_prior_param_val,
                 kld, p_cat, q_cat):
    """ `_run_process` is the method that can be used to sample a new model in
    the PoPEx procedure. It mainly generates a new model and computes the
    corresponding log-likelihood value.


    Parameters
    ----------
    pb : Problem
        Problem definition
    popex : PoPEx
        PoPEx main structure
    imod : int
        Model index
    hd_prior_param_ind : m-tuple
        Prior hard conditioning indices (cf. :meth:`pb.get_hd_pri`)
    hd_prior_param_val : m-tuple
        Prior hard conditioning values (cf. :meth:`pb.get_hd_pri`)
    kld : m-tuple
        Kullback-Leibler divergence map (cf. :meth:`pb.compute_kld`)
    p_cat : m-tuple
        Weighted category probabilities
    q_cat : m-tuple
        Prior category probabilities


    Returns
    -------
    imod : int
        Model index
    model : m-tuple
        Tuple of ``Mtype`` instances that define the new model
    ncmod : m-tuple
        Number of hard conditioning for each model type
    log_p_lik : float
        Log-likelihood value
    cmp_log_p_lik : bool
        Log-likelihood was computed (True) or predicted (False)
    log_p_pri : float
        Log-prior probability of the model
    log_p_gen : float
        Log-sampling probability of the model

    """

    np.random.seed(pb.seed + imod)
    meth_w_hd = pb.meth_w_hd

    # Number of conditioning (is zero everywhere if sum(w_hd) = 0)
    ncmod_tmp = utl.compute_ncmod(popex, meth_w_hd)

    # Hard conditioning
    hd_i_param_ind, hd_i_param_val, hd_prior, hd_generation = \
        utl.generate_hd(popex, meth_w_hd, ncmod_tmp, kld, p_cat, q_cat)
    hd_param_ind, hd_param_val = \
        utl.merge_hd(hd_prior_param_ind, hd_i_param_ind,
                     hd_prior_param_val, hd_i_param_val)

    # The number of hard conditioning data may has changed (duplicated loc.)
    ncmod = tuple([len(hd_ind) if hd_ind is not None else 0
                   for hd_ind in hd_i_param_ind])

    # Generate model
    model = pb.generate_m(hd_param_ind, hd_param_val, imod)

    # Write hd information (!!! without prior hd !!!)
    utl.write_hd_info(popex, imod, hd_i_param_ind, hd_i_param_val)

    # Compute log-likelihood (if learning_scheme is None or with p_eval)
    if not pb.learning_scheme or \
            np.random.rand(1) < pb.learning_scheme.compute_p_eval_for(model):
        log_p_lik = pb.compute_log_p_lik(model, imod)
        cmp_log_p_lik = True
    # Predict likelihood (otherwise)
    else:
        log_p_lik = pb.learning_scheme.learn_value_of(model)
        cmp_log_p_lik = False

    # Test positivity of likelihood value
    if np.exp(log_p_lik) < -NP_MIN_TOL:
        raise ValueError(msg.err4001(list(imod)))

    # Compute sampling ratio
    log_p_pri = pb.compute_log_p_pri(model, hd_prior, hd_i_param_ind)
    log_p_gen = pb.compute_log_p_gen(model, hd_generation, hd_param_ind)

    return imod, model, ncmod, log_p_lik, cmp_log_p_lik, log_p_pri, log_p_gen


def _write_run_sum(pb, popex, nmax, t_popex, t_mod):
    """ `_write_run_sum` writes a document that summarized the overall
    progress of the popex run at 'popex.path_res'.


    The file structure is as followes:
        | <popex.path_res>$
        |    └- run_progress.txt


    Parameters
    ----------
    pb : Problem
        Problem definition
    popex : PoPEx
        PoPEx main structure
    nmax : int
        Number of maximal models
    t_popex : float
        Total time elapsed
    t_mod : float
        Average time per model


    Returns
    -------
    None

    """

    # Generic constants
    bar_width = 40

    # Path of the result location
    path_res = popex.path_res

    # Compute n_e diagnostics
    with warnings.catch_warnings(record=True) as _:
        ne_l    = isampl.ne(utl.compute_w_lik(popex=popex,
                                              meth={'name': 'exact'}))
        ne_w_hd = isampl.ne(utl.compute_w_lik(popex=popex,
                                              meth=pb.meth_w_hd))
        ne_w_l  = isampl.ne(utl.compute_w_pred(popex=popex,
                                               nw_min=0,
                                               meth={'name': 'exact'}))

    # Compute time information (t_popex, t_mod, t_est)
    progress = popex.nmod / nmax
    m_popex, s_popex = divmod(t_popex, 60)
    h_popex, m_popex = divmod(m_popex, 60)
    m_mod, s_mod = divmod(t_mod, 60)
    h_mod, m_mod = divmod(m_mod, 60)
    t_est = t_popex / progress
    m_est, s_est = divmod(t_est, 60)
    h_est, m_est = divmod(m_est, 60)

    # Write into a file 'iteration_info.txt'
    with open('{}/run_progress.txt'.format(path_res), 'w+') as file:
        # Start with empty line
        file.write('\n')

        # Write title
        file.write('RUN PROGRESS SUMMARY\n')
        file.write('--------------------\n')
        file.write('\n')

        # Print status bar
        nfull = int(bar_width * progress)
        nempty = bar_width - nfull
        file.write('  Status\n')
        file.write('    n_mod       = {:7d} / {:7d}\n'.format(popex.nmod, nmax))
        file.write('    [{:s}{:s}]'.format('-' * nfull, ' ' * nempty))
        file.write(' {:3.0f}%\n'.format(progress * 100))
        file.write('\n')

        # Write n_e
        file.write('  Diagnostics\n')
        file.write('    ne(L)       = {:9.1f}\n'.format(ne_l))
        file.write('    ne(w_hd)    = {:9.1f}\n'.format(ne_w_hd))
        file.write('    ne(w_pred)  = {:9.1f}\n'.format(ne_w_l))
        file.write('\n')

        # Write compute/predict stats
        n_comp = np.sum(popex.cmp_log_p_lik)
        n_pred = popex.nmod - n_comp
        file.write('  Computed or Predicted\n')
        file.write('    n_cmp       = {:7d}\n'.format(n_comp))
        file.write('    n_prd       = {:7d}\n'.format(n_pred))
        file.write('\n')

        # Write times
        file.write('  Time\n')
        file.write('    T_el        = {:4.0f}[h] {:2.0f}[m] {:5.2f}[s]\n'
                   .format(h_popex, m_popex, s_popex))
        file.write('    T / mod     = {:4.0f}[h] {:2.0f}[m] {:5.2f}[s]\n'
                   .format(h_mod, m_mod, s_mod))
        file.write('    T_tot       = {:4.0f}[h] {:2.0f}[m] {:5.2f}[s]\n'
                   .format(h_est, m_est, s_est))

        # End with empty line
        file.write('\n')


def pred_popex_mp(pred, path_res, nmp=1):
    """ `pred_popex_mp` is the main implementation for computing predictions
    from a PoPEx sampling.


    Notes
    -----
    The computations are supposed to be independent, such that they can be
    computed in parallel.


    Parameters
    ----------
    pred : Prediction
        Defines the prediction functions and parameters
    path_res : str
        Path for loading the PoPEx results
    nmp : int
        Number of parallel processes


    Returns
    -------
    None

    """
    # Start PoPEx predictions
    tst_pred = time.time()

    # Load PoPEx results
    with open(path_res + 'popex.pop', 'rb') as file:
        popex = pickle.load(file)  # type: PoPEx

    # Compute prediction indices
    w_pred = utl.compute_w_pred(popex=popex,
                                nw_min=pred.nw_min,
                                meth=pred.meth_w_pred)
    ind_pred = utl.compute_subset_ind(pred.wfrac_pred, w_pred)
    npred = len(ind_pred)

    # Write some information
    print("\n  START 'PRED_POPEX_MP'\n")
    print('    nmp        = {:>6}'.format(nmp))
    print('    npred      = {:>6}'.format(npred))
    print("    pred_meth  = '{}'\n".format(pred.meth_w_pred['name']))

    # Main prediction loop
    stop = False
    ipred = 0
    ndone = 0
    proc_mngr = deque([], maxlen=nmp)
    print('    > predict data...', end='')
    with multiprocessing.Pool(processes=nmp, maxtasksperchild=100) as pool:
        while not stop:
            if len(proc_mngr) < nmp and ipred < npred:
                # Create new process
                imod = ind_pred[ipred]
                args = (deepcopy(pred), deepcopy(popex), imod)
                proc_mngr.append(pool.apply_async(_pred_process, args))
                ipred += 1

            # Check if the process at the head of the queue has finished
            if proc_mngr[0].ready():
                # Remove process
                proc = proc_mngr.popleft()
                ipred_proc = proc.get()
                ndone += 1

                # Write some information
                t_pred = time.time() - tst_pred
                _write_pred_sum(path_res, ipred_proc, ndone, npred, t_pred)

                # Update stopping condition
                if len(proc_mngr) == 0 and ipred >= npred:
                    stop = True
            else:
                # Send process to the back of the queue
                proc_mngr.rotate(-1)
    print('done')

    # Save Prediction
    print('    > save prediction...', end='')
    with open('{}pred.pred'.format(popex.path_res), 'wb') as file:
        pickle.dump(pred, file)
    print('done')

    print("\n  END 'PRED_POPEX_MP'\n")


def _pred_process(pred, popex, imod):
    """ `_pred_process` is the method that can be used to compute a prediction
    from one model in a PoPEx sampling.


    Parameters
    ----------
    pred : Prediction
        Defines the prediction functions and parameters
    popex : PoPEx
        PoPEx main structure
    imod : int
        Model index


    Returns
    -------
    None
    """
    pred.compute_pred(popex, imod)
    return imod


def _write_pred_sum(path_res, ipred, ndone, ntot, t_pred):
    """ `_write_pred_sum` writes a text file that summarized the overall
    progress of the predictions at 'popex.path_res'.


    The file structure is as followes:
        | <popex.path_res>$
        |    └- pred_progress.txt


    Parameters
    ----------
    path_res : str
        Path to where the results should be saved
    ipred : int
        Index of last predictions
    ndone : int
        Number of computed predictions
    ntot : int
        Number of total predictions
    t_pred : float
        Total time elapsed


    Returns
    -------
    None
    """
    # Generic constants
    bar_width = 40

    # Compute time information (t_pred, t_mod, t_est)
    progress = ndone / ntot
    t_mod = t_pred / ndone
    m_pred, s_pred = divmod(t_pred, 60)
    h_pred, m_pred = divmod(m_pred, 60)
    m_mod, s_mod = divmod(t_mod, 60)
    h_mod, m_mod = divmod(m_mod, 60)
    t_est = t_pred / progress
    m_est, s_est = divmod(t_est, 60)
    h_est, m_est = divmod(m_est, 60)

    # Write into a file 'iteration_info.txt'
    with open('{}pred_progress.txt'.format(path_res), 'w+') as file:
        # Write title
        file.write('PREDICTION PROGRESS SUMMARY\n')
        file.write('---------------------------\n')
        file.write('\n')

        # Print status bar
        nfull = int(bar_width * progress)
        nempty = bar_width - nfull
        file.write('  Status\n')
        file.write('    n_mod    = {:6d} / {:6d}    (last: {:6d})\n'
                   .format(ndone, ntot, ipred))
        file.write('    [{:s}{:s}]'.format('-' * nfull, ' ' * nempty))
        file.write(' {:3.0f}%\n'.format(progress * 100))
        file.write('\n')

        # Write times
        file.write('  Time\n')
        file.write('    T_el     = {:4.0f}[h] {:2.0f}[m] {:5.2f}[s]\n'
                   .format(h_pred, m_pred, s_pred))
        file.write('    T / mod  = {:4.0f}[h] {:2.0f}[m] {:5.2f}[s]\n'
                   .format(h_mod, m_mod, s_mod))
        file.write('    T_tot    = {:4.0f}[h] {:2.0f}[m] {:5.2f}[s]\n'
                   .format(h_est, m_est, s_est))
