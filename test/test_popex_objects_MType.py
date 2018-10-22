""" TEST FILE
Some tests are run to check the behaviour of all model type classes, namely
    - 'MType'
    - 'CatMType'
    - 'ContParam'
    - 'CatProb'
    - 'CatParam'
in 'popex.popex_objects'. The main focus lies in instantiating and changing
class instances and check their behaviour. The tests are all formulated in
terms of the 'assert' method.
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

# Package imports
from popex.popex_objects import MType, ContParam, CatMType, CatProb, CatParam


# Unit test class
class TestMTypes(unittest.TestCase):

    def setUp(self):
        self.categories1 = [[(0.25, 0.75), (2.25, 2.75)],
                            [(0.75, 1.0)],
                            [(1.0, 2.25)]]
        self.ncategories1 = 3
        self.categories2 = [[(0.25, 0.75), (2.25, 2.75)],
                            [(0.75, 2.25)]]
        self.ncategories2 = 2
        self.arr1 = np.array([0.5, 1.5, 2, 0.75, 1, 2.5])
        self.nparam1 = 6
        self.cat11 = np.array([0, 2, 2, 1, 2, 0], dtype='int')
        self.cat12 = np.array([0, 1, 1, 1, 1, 0])
        self.arr2 = np.array([1.75, 2.25, 2.249, 0.25])
        self.nparam2 = 4
        self.cat21 = np.array([2, 0, 2, 0], dtype='int')
        self.cat22 = np.array([1, 0, 1, 0])
        self.arr_2d = np.array([[1., 2., 3.], [4, 5, 6]])
        self.not_ndarray = [1., 2., 3., 4., 5.]
        self.nparam_not_ndarray = 5

        self. arr_prob1 = np.array([[0.50, 0.45, 0.05],
                                    [0.00, 0.01, 0.99],
                                    [0.35, 0.30, 0.35],
                                    [0.15, 0.45, 0.40],
                                    [0.10, 0.10, 0.80],
                                    [0.90, 0.05, 0.05]])

        self.arr_prob1 = np.array([[0.50, 0.45, 0.05],
                                   [0.00, 0.01, 0.99],
                                   [0.35, 0.30, 0.35],
                                   [0.15, 0.45, 0.40],
                                   [0.10, 0.10, 0.80],
                                   [0.90, 0.05, 0.05]])

    def test_MType(self):
        with self.assertRaises(TypeError) as _:
            MType()

    def test_CatMType(self):
        with self.assertRaises(TypeError) as _:
            CatMType()

    def test_ContParam(self):
        # Check instantiation
        dtype = 'float32'
        obj = ContParam(dtype_val=dtype, param_val=self.arr1)
        # print(obj)
        self.assertTrue(isinstance(obj, MType))
        self.assertFalse(isinstance(obj, CatMType))
        self.assertTrue(isinstance(obj, ContParam))
        self.assertTrue(hasattr(obj, 'param_val'))
        self.assertTrue(isinstance(obj.param_val, np.ndarray))
        self.assertEqual(obj.param_val.dtype, dtype)
        self.assertTrue(np.all(obj.param_val == self.arr1))
        self.assertEqual(obj.nparam, self.nparam1)

        # Check value change
        obj.param_val = self.arr2
        # print(obj)
        self.assertTrue(hasattr(obj, 'param_val'))
        self.assertTrue(isinstance(obj.param_val, np.ndarray))
        self.assertEqual(obj.param_val.dtype, self.arr2.dtype)
        self.assertTrue(np.all(obj.param_val == self.arr2))
        self.assertEqual(obj.nparam, self.nparam2)

        # Check transformation from non-ndarray
        obj.param_val = self.not_ndarray
        arr = np.array(self.not_ndarray)
        self.assertTrue(hasattr(obj, 'param_val'))
        self.assertTrue(isinstance(obj.param_val, np.ndarray))
        self.assertEqual(obj.param_val.dtype, arr.dtype)
        self.assertTrue(np.all(obj.param_val == arr))
        self.assertEqual(obj.nparam, self.nparam_not_ndarray)

        # Check error raising for 2-dimensional array
        with self.assertRaises(AttributeError) as _:
            obj.param_val = self.arr_2d

        # Check default definition
        obj = ContParam()
        self.assertTrue(isinstance(obj, MType))
        self.assertTrue(isinstance(obj, ContParam))
        self.assertTrue(hasattr(obj, 'param_val'))
        self.assertTrue(isinstance(obj.param_val, np.ndarray))
        self.assertEqual(obj.param_val.dtype, 'float64')
        self.assertTrue(np.all(obj.param_val == np.empty((0,))))
        self.assertEqual(obj.nparam, 0)

    def test_CatProb(self):
        # Check instantiation
        dtype = 'float64'
        obj = CatProb(dtype_val=dtype,
                      param_val=self.arr_prob1,
                      categories=self.categories1)
        # print(obj)
        self.assertTrue(isinstance(obj, MType))
        self.assertTrue(isinstance(obj, CatMType))
        self.assertTrue(isinstance(obj, CatProb))
        self.assertFalse(isinstance(obj, CatParam))
        self.assertTrue(hasattr(obj, 'param_val'))
        self.assertTrue(hasattr(obj, 'categories'))
        self.assertTrue(isinstance(obj.param_val, np.ndarray))
        self.assertTrue(isinstance(obj.categories, list))
        self.assertEqual(obj.param_val.dtype, dtype)
        self.assertTrue(np.all(obj.param_val == self.arr_prob1))
        self.assertEqual(obj.nparam, self.nparam1)
        self.assertEqual(obj.ncat, self.ncategories1)

        # Check error raising
        with self.assertRaises(AttributeError) as _:
            CatProb(dtype_val=dtype,
                    param_val=self.arr1,
                    categories=self.categories1)

    def test_CatParam(self):
        # Check instantiation
        dtype = 'float16'
        obj = CatParam(dtype_val=dtype,
                       param_val=self.arr1,
                       categories=self.categories1)
        # print(obj)
        self.assertTrue(isinstance(obj, MType))
        self.assertTrue(isinstance(obj, CatMType))
        self.assertFalse(isinstance(obj, CatProb))
        self.assertTrue(isinstance(obj, CatParam))
        self.assertTrue(hasattr(obj, 'param_val'))
        self.assertTrue(hasattr(obj, 'categories'))
        self.assertTrue(isinstance(obj.param_val, np.ndarray))
        self.assertTrue(isinstance(obj.categories, list))
        self.assertEqual(obj.param_val.dtype, dtype)
        self.assertTrue(np.all(obj.param_val == self.arr1))
        self.assertEqual(obj.nparam, self.nparam1)
        self.assertTrue(np.all(obj.param_cat == self.cat11))
        self.assertEqual(obj.ncat, self.ncategories1)

        # Check value change
        obj.param_val = self.arr2
        # print(obj)
        self.assertTrue(hasattr(obj, 'param_val'))
        self.assertTrue(hasattr(obj, 'categories'))
        self.assertTrue(isinstance(obj.param_val, np.ndarray))
        self.assertTrue(isinstance(obj.categories, list))
        self.assertEqual(obj.param_val.dtype, self.arr2.dtype)
        self.assertTrue(np.all(obj.param_val == self.arr2))
        self.assertEqual(obj.nparam, self.nparam2)
        self.assertTrue(np.all(obj.param_cat == self.cat21))
        self.assertEqual(obj.ncat, self.ncategories1)

        # Check category change
        obj.categories = self.categories2
        # print(obj)
        self.assertTrue(hasattr(obj, 'param_val'))
        self.assertTrue(hasattr(obj, 'categories'))
        self.assertTrue(isinstance(obj.param_val, np.ndarray))
        self.assertTrue(isinstance(obj.categories, list))
        self.assertEqual(obj.param_val.dtype, self.arr2.dtype)
        self.assertTrue(np.all(obj.param_val == self.arr2))
        self.assertEqual(obj.nparam, self.nparam2)
        self.assertTrue(np.all(obj.param_cat == self.cat22))
        self.assertEqual(obj.ncat, self.ncategories2)

        # Check error raising (wrong array dimension)
        with self.assertRaises(AttributeError) as _:
            CatParam(dtype_val=dtype,
                     param_val=self.arr_prob1,
                     categories=self.categories1)

        # Check error raising (not enough categories)
        with self.assertRaises(ValueError) as _:
            CatParam(dtype_val=dtype,
                     param_val=self.arr1,
                     categories=[[(0.1, 0.5)]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
