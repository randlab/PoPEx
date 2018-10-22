""" TEST FILE
Some tests are run to check the behaviour of the class
    - 'Problem'
in 'popex.popex_objects'. A toy problem is set up and each function is run
multiple times. The tests are all formulated in terms of the 'assert' method.
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
from popex.popex_objects import PoPEx, Problem, CatParam, ContParam

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
    return tuple([m1, m2, m3])


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
class TestProblem(unittest.TestCase):

    def setUp(self):
        self.prbl = Problem(generate_m=generate_m,
                            compute_log_p_lik=compute_log_p_lik,
                            compute_log_p_pri=compute_log_p_pri,
                            compute_log_p_gen=compute_log_p_gen,
                            get_hd_pri=get_hd_pri)

    def test_Problem_class(self):
        popex = PoPEx(ncmax=(10, 20, 0),
                      nmtype=3,
                      path_res=PATH_TEST)
        mod = self.prbl.generate_m()
        self.assertTrue(all([isinstance(mod[0], CatParam),
                             isinstance(mod[1], CatParam),
                             isinstance(mod[2], ContParam)]))
        p_lik = self.prbl.compute_log_p_lik(mod)
        log_p_pri = self.prbl.compute_log_p_pri(mod)
        log_p_gen = self.prbl.compute_log_p_gen(mod)

        popex.add_model(0, mod, p_lik, True, log_p_pri, log_p_gen, (2, 5, 0))

        mod = self.prbl.generate_m()
        p_lik = self.prbl.compute_log_p_lik(mod)
        log_p_pri = self.prbl.compute_log_p_pri(mod)
        log_p_gen = self.prbl.compute_log_p_gen(mod)

        popex.add_model(1, mod, p_lik, True, log_p_pri, log_p_gen, (1, 6, 0))

        mod = self.prbl.generate_m()
        p_lik = 0
        log_p_pri = self.prbl.compute_log_p_pri(mod)
        log_p_gen = self.prbl.compute_log_p_gen(mod)

        popex.add_model(2, mod, p_lik, False, log_p_pri, log_p_gen, (1, 6, 0))

        self.assertTrue(isinstance(popex.model, list))
        self.assertTrue(isinstance(popex.log_p_lik, np.ndarray))
        self.assertTrue(isinstance(popex.cmp_log_p_lik, np.ndarray))
        self.assertTrue(isinstance(popex.log_p_pri, np.ndarray))
        self.assertTrue(isinstance(popex.log_p_gen, np.ndarray))
        self.assertTrue(isinstance(popex.nc, list))

        self.assertEqual(popex.cmp_log_p_lik.dtype, 'bool')
        self.assertEqual(popex.nmod, 3)

        self.assertEqual(len(popex.model), 3)
        self.assertEqual(len(popex.log_p_lik), 3)
        self.assertEqual(len(popex.cmp_log_p_lik), 3)
        self.assertEqual(len(popex.log_p_pri), 3)
        self.assertEqual(len(popex.log_p_gen), 3)
        self.assertEqual(len(popex.nc), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
