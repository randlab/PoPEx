import numpy as np
import popex.utils as utils
from popex.popex_objects import CatParam, CatProb

def test_check_interval():
    result = utils.check_interval(np.array([1, 2, 3, 4]), (1.5, 2.5))
    assert np.array_equal(result, np.array([False,  True, False, False]))

def test_check_category():
    list_values = np.array([1.2, 3.5, 6, 9])
    category = [(1,2),(3,4),(8,10)]
    result = utils.check_category(list_values, category)
    assert np.array_equal(result, [True, True, False, True])
    

def test_list_mCatParam_to_mCatProb():
    
    c1 = [[(0.5,1.5)], [(1.5,2.5)]]
    c2 = [[(0.5,1.5)], [(1.5,2.5)], [(2.5, 3.5)]]
    CatProb1 = CatProb(param_val=np.array([[0, 1], [1, 0], [0.5, 0.5]]),
            categories=c1)
    CatProb2 = CatProb(param_val=np.array([[0, 1,0], [0.5, 0, 0.5]]),
            categories=c2)
    mCatProb = (CatProb1, CatProb2)

    list_mCatParam = [(CatParam(param_val=[2,1,1],categories=c1),
        CatParam(param_val=[2,1],categories=c2)),
            (CatParam(param_val=[2,1,2],categories=c1),
                CatParam(param_val=[2,3],categories=c2))]

    res_mCatProb = utils.list_mCatParam_to_mCatProb(list_mCatParam)

    print(mCatProb[0], mCatProb[1])
    print(res_mCatProb[0], res_mCatProb[1])

    assert mCatProb == res_mCatProb
