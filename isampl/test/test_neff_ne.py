""" This file tests the module 'neff.py'
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

import unittest
import numpy as np
import popex.isampl as isampl


class TestNeff(unittest.TestCase):

    def setUp(self):
        self.nz = 50
        self.no = 100
        self.nrnd = 200
        self.w1 = np.append(np.zeros(self.nz), np.ones(self.no))
        np.random.shuffle(self.w1)
        self.w2 = np.append(np.zeros(self.nz), np.ones(1))
        np.random.shuffle(self.w2)
        self.w3 = np.random.rand(self.nrnd)

    def test_ne(self):
        """ TEST METHOD
        This test considers the methods
            - ne
            - ne_gamma
            - ne_var
        from popex.isampl.neff.
        """
        # Test isampl.ne
        self.assertEqual(isampl.ne(self.w1), self.no)
        self.assertEqual(isampl.ne(self.w2), 1)

        # Test isampl.ne_gamma
        self.assertEqual(isampl.ne_gamma(self.w1), self.no)
        self.assertEqual(isampl.ne_gamma(self.w2), 1)

        # Test isampl.ne_var
        self.assertEqual(isampl.ne_var(self.w1), self.no)
        self.assertEqual(isampl.ne_var(self.w2), 1)

        # Test comparison
        ne = isampl.ne(self.w3)
        ne_gamma = isampl.ne_gamma(self.w3)
        ne_var = isampl.ne_var(self.w3)
        self.assertTrue(ne >= ne_gamma)
        self.assertTrue(ne_gamma >= ne_var)

    def test_alpha(self):
        """ TEST METHOD
        This test consider the alpha computation in popex.isampl.alpha
        """
        ne1 = self.nrnd / 4
        ne2 = self.nrnd
        ne3 = 0
        a1 = isampl.alpha(self.w3, ne1, a_init=1)
        a2 = isampl.alpha(self.w3, ne2, a_init=1)
        a3 = isampl.alpha(self.w3, ne3, a_init=1)

        print(a1, a2, a3)
        print(isampl.ne(self.w3 ** a1),
              isampl.ne(self.w3 ** a2),
              isampl.ne(self.w3 ** a3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
