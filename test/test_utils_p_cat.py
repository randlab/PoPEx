""" TEST FILE
Some tests are run to check the behaviour of the functions
    - 'utils.compute_p_cat'
    - 'utils.update_p_cat'.
A toy problem is set up and each function is run multiple times. The tests are
all formulated in terms of the 'assert' method.
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

# General imports
import unittest
import numpy as np
import os

# # Package imports
from popex.popex_objects import PoPEx, CatParam, ContParam, CatProb, MType
import popex.utils as utl
from popex.cnsts import NP_MIN_TOL

PATH_TEST = os.path.dirname(os.path.dirname(__file__)) + '/'\
               + 'test/test_dir/'


# Set up toy functions for 'Problem'
def generate_m():
    categories1 = [[(0.00, 0.75), (2.25, 3.00)],
                   [(0.75, 1.0)],
                   [(1.0, 2.25)]]
    categories2 = [[(0.00, 0.75), (2.25, 3.00)],
                   [(0.75, 2.25)]]
    m1 = CatParam(param_val=np.random.rand(3) * 3,
                  categories=categories1)
    m2 = CatParam(param_val=np.random.rand(6) * 3,
                  categories=categories2)
    m3 = ContParam(param_val=np.random.rand(4) * 3)
    return m1, m2, m3


def compute_log_p_lik(model):
    return sum([np.sum(mod.param_val) for mod in model]) / 3


def compute_log_p_pri(model):
    return -max([np.sum(mod.param_val) for mod in model])


def compute_log_p_gen(model):
    return -min([np.sum(mod.param_val) for mod in model])


def get_hd_pri():
    hd_prior_param_ind = tuple(None for _ in range(3))
    hd_prior_param_val = tuple(None for _ in range(3))
    return hd_prior_param_ind, hd_prior_param_val


# Unit test class
class TestPCat(unittest.TestCase):

    def setUp(self):
        self.ncmax = (1, 2, 0)
        self.nmtype = 3
        self.nmod_init = 100
        self.nmod_sup = 10
        self.ncond_mtype = [i for i, n in enumerate(self.ncmax) if n == 0]
        self.cond_mtype = [i for i, n in enumerate(self.ncmax) if n > 0]

    def test_compute_p_cat(self):
        popex = PoPEx(ncmax=self.ncmax,
                      nmtype=self.nmtype,
                      path_res=PATH_TEST)

        for imod in range(self.nmod_init):
            mod       = generate_m()
            p_lik     = compute_log_p_lik(mod)
            log_p_pri = compute_log_p_pri(mod)
            log_p_gen = compute_log_p_gen(mod)
            popex.add_model(0, mod, p_lik, True, log_p_pri, log_p_gen, (2, 5, 0))

        w_pred = utl.compute_w_pred(popex)
        p_cat = utl.compute_cat_prob(popex, w_pred)

        for imtype in self.ncond_mtype:
            self.assertTrue(p_cat[imtype] is None)
        for imtype in self.cond_mtype:
            self.assertTrue(isinstance(p_cat[imtype], MType))
            self.assertTrue(isinstance(p_cat[imtype], CatProb))
            self.assertTrue(
                p_cat[imtype].param_val.shape[0] == p_cat[imtype].nparam)
            self.assertTrue(
                p_cat[imtype].param_val.shape[1] == p_cat[imtype].ncat)

    def test_update_p_cat(self):
        popex = PoPEx(ncmax=self.ncmax,
                      nmtype=self.nmtype,
                      path_res=PATH_TEST)

        for imod in range(self.nmod_init):
            mod = generate_m()
            p_lik = compute_log_p_lik(mod)
            log_p_pri = compute_log_p_pri(mod)
            log_p_gen = compute_log_p_gen(mod)
            popex.add_model(imod, mod, p_lik, True, log_p_pri, log_p_gen,
                            ncmod=(0, 0, 0))

        w_pred = utl.compute_w_pred(popex)
        p_cat_upd = utl.compute_cat_prob(popex, w_pred)

        m_new = []
        for imod in range(self.nmod_sup):
            mod = generate_m()
            m_new.append(mod)
            p_lik = compute_log_p_lik(mod)
            log_p_pri = compute_log_p_pri(mod)
            log_p_gen = compute_log_p_gen(mod)
            popex.add_model(imod+self.nmod_init, mod,
                            p_lik, True, log_p_pri, log_p_gen,
                            ncmod=(0, 0, 0))

        w_pred = utl.compute_w_pred(popex)
        p_cat_cmp = utl.compute_cat_prob(popex, w_pred)

        sum_w_old = np.sum(w_pred[:self.nmod_init])
        w_new = w_pred[self.nmod_init:]
        utl.update_cat_prob(p_cat_upd, m_new, w_new, sum_w_old)

        for imtype in self.ncond_mtype:
            self.assertTrue(p_cat_upd[imtype] is None)
        for imtype in self.cond_mtype:
            self.assertTrue(isinstance(p_cat_upd[imtype], MType))
            self.assertTrue(isinstance(p_cat_upd[imtype], CatProb))
            self.assertTrue(
                p_cat_upd[imtype].param_val.shape[0] == p_cat_upd[imtype].nparam)
            self.assertTrue(
                p_cat_upd[imtype].param_val.shape[1] == p_cat_upd[imtype].ncat)

        for imtype in self.cond_mtype:
            diff = p_cat_cmp[imtype].param_val - p_cat_upd[imtype].param_val
            self.assertTrue(np.all(np.abs(diff) < NP_MIN_TOL))


if __name__ == '__main__':
    unittest.main(verbosity=2)