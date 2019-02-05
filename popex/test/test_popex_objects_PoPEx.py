""" TEST FILE
Some tests are run to check the behaviour of the class
    - 'PoPEx'
in 'popex.popex_objects'. Some toy models are created and added to the PoPEx
structure. The tests are all formulated in terms of the 'assert' method.
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

# Package imports
from popex.popex_objects import PoPEx, CatParam, ContParam

PATH_TEST = os.path.dirname(os.path.dirname(__file__)) + '/'\
               + 'test/test_dir/'


# Unit test class
class TestPoPEx(unittest.TestCase):

    def setUp(self):
        param_val1 = np.array([0.5, 1.5, 2, 0.75, 1, 2.5])
        categories1 = [[(0.25, 0.75), (2.25, 2.75)],
                       [(0.75, 1.0)],
                       [(1.0, 2.25)]]
        mod11 = CatParam(param_val=param_val1, categories=categories1)

        param_val2 = np.array([1.75, 2.25, 2.249, 0.25])
        categories2 = [[(0.25, 0.75), (2.25, 2.75)],
                       [(0.75, 2.25)]]
        mod12 = CatParam(param_val=param_val2, categories=categories2)

        param_val3 = np.array([1., 2., 3.4, 5, 6])
        mod13 = ContParam(param_val=param_val3)

        self.mod1 = (mod11, mod12, mod13)
        self.p_lik1 = 2.5
        self.cmp_plik1 = True
        self.log_p_pri1 = -2.5
        self.log_p_gen1 = -1.5
        self.nc1 = (1, 2, 0)

        self.ncmax = (10, 20, 0)
        self.nmtype = 3
        self.nmc = 1

    def test_PoPEx(self):
        popex = PoPEx(ncmax=self.ncmax,
                      nmtype=self.nmtype,
                      path_res=PATH_TEST)
        self.assertTrue(isinstance(popex, PoPEx))
        self.assertTrue(hasattr(popex, 'model'))
        self.assertTrue(hasattr(popex, 'log_p_lik'))
        self.assertTrue(hasattr(popex, 'cmp_log_p_lik'))
        self.assertTrue(hasattr(popex, 'log_p_pri'))
        self.assertTrue(hasattr(popex, 'log_p_gen'))
        self.assertTrue(hasattr(popex, 'nc'))
        self.assertTrue(hasattr(popex, 'path_res'))

        popex.add_model(0, self.mod1, self.p_lik1, self.cmp_plik1,
                        self.log_p_pri1, self.log_p_gen1, self.nc1)

        self.assertTrue(isinstance(popex.model, list))
        self.assertTrue(isinstance(popex.log_p_lik, np.ndarray))
        self.assertTrue(isinstance(popex.cmp_log_p_lik, np.ndarray))
        self.assertTrue(isinstance(popex.log_p_pri, np.ndarray))
        self.assertTrue(isinstance(popex.log_p_gen, np.ndarray))
        self.assertTrue(isinstance(popex.nc, list))

        self.assertEqual(popex.cmp_log_p_lik.dtype, 'bool')
        self.assertEqual(popex.nmod, 1)

        popex.add_model(1, self.mod1, 0, False,
                        -12., -14.5, (15, 12, 0))

        self.assertTrue(isinstance(popex.model, list))
        self.assertTrue(isinstance(popex.log_p_lik, np.ndarray))
        self.assertTrue(isinstance(popex.cmp_log_p_lik, np.ndarray))
        self.assertTrue(isinstance(popex.log_p_pri, np.ndarray))
        self.assertTrue(isinstance(popex.log_p_gen, np.ndarray))
        self.assertTrue(isinstance(popex.nc, list))

        self.assertEqual(popex.cmp_log_p_lik.dtype, 'bool')
        self.assertEqual(popex.nmod, 2)

        self.assertEqual(len(popex.model), 2)
        self.assertEqual(len(popex.log_p_lik), 2)
        self.assertEqual(len(popex.cmp_log_p_lik), 2)
        self.assertEqual(len(popex.log_p_pri), 2)
        self.assertEqual(len(popex.log_p_gen), 2)
        self.assertEqual(len(popex.nc), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)