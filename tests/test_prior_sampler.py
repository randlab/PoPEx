import numpy as np
import popex.utils as utils

def test_check_interval():
    result = utils.check_interval(np.array([1, 2, 3, 4]), (1.5, 2.5))
    assert np.array_equal(result, np.array([False,  True, False, False]))

def test_check_category():
    list_values = np.array([1.2, 3.5, 6, 9])
    category = [(1,2),(3,4),(8,10)]
    result = utils.check_category(list_values, category)
    assert np.array_equal(result, [True, True, False, True])
