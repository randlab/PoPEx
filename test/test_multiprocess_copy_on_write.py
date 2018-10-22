""" TEST FILE
Some test are run to consider the behaviour of the input structures in a
multiprocessing pool. In fact this file is used to consider the exact behaviour
of the arguments that are passed into an process (copy vs. deepcopy).

Conclusion
When a process is created, python issues a 'fork()'. This creates a child
process whose memory space is an exact copy of its parent. On linux this is
made efficient through "copy-on-write".
    From Wikipedia:
    'Copy-on-write (CoW or COW) is a resource-management technique used in
    computer programming to efficiently implement a "duplicate" or "copy"
    operation on modifiable resources.[3] If a resource is duplicated but not
    modified, it is not necessary to create a new resource; the resource can be
    shared between the copy and the original. Modifications must still create
    a copy, hence the technique: the copy operation is deferred to the first
    write.'

!!! CAUTION !!!
If an object is not changed but returned from a subprocess, a deepcopy is still
performed!!!!!!
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
from multiprocessing import Pool
from copy import deepcopy
from collections import deque
import time


def prc_change(foo):
    # Change foo
    foo.lst[2] = 888
    foo.lst.append(0.5)
    return None


def prc_return(foo):
    # No change is made but object is returned
    return foo


def prc_nothing(foo):
    return None


class Foo:
    def __init__(self, lst, arr):
        self.lst = lst
        self.arr = arr


def prc_eq(foo):
    slp = np.random.rand(1)[0] * 2
    time.sleep(slp)
    if len(foo.lst) == foo.arr.size:
        return True
    else:
        print(foo.lst)
        print(foo.arr)
        return False


# Unit test class
class TestCopyOnWrite(unittest.TestCase):

    def setUp(self):
        self.nproc = 5

        lst = [0.0, 0.1, 0.2, 0.3, 0.4]
        arr = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        self.bar_cp = Foo(lst=lst, arr=arr)
        self.bar_dcp = Foo(lst=deepcopy(lst), arr=deepcopy(arr))

    def test_change(self):
        """ 'copy-on-write' makes a deepcopy when the object is changed within
        a subprocess.
        """
        with Pool(processes=self.nproc) as pool:
            # Check with processes
            # --------------------
            p_cp = pool.apply_async(prc_change, (self.bar_cp,))
            p_dcp = pool.apply_async(prc_change, (deepcopy(self.bar_dcp),))
            p_cp.wait()
            p_dcp.wait()

            # Original objects must be unchanged
            self.assertFalse(self.bar_cp.lst[2] == 888)
            self.assertFalse(self.bar_dcp.lst[2] == 888)
            self.assertFalse(len(self.bar_cp.lst) == 6)
            self.assertFalse(len(self.bar_dcp.lst) == 6)

            # Check without processes
            # -----------------------
            prc_change(self.bar_cp)
            prc_change(deepcopy(self.bar_dcp))

            # Only 'deepcopy' objects must be unchanged
            self.assertTrue(self.bar_cp.lst[2] == 888)
            self.assertFalse(self.bar_dcp.lst[2] == 888)
            self.assertTrue(len(self.bar_cp.lst) == 6)
            self.assertFalse(len(self.bar_dcp.lst) == 6)

    def test_return(self):
        """ 'copy-on-write' makes a deepcopy when the object is returned from
        a subprocess.
        """
        with Pool(processes=self.nproc) as pool:
            # Check with processes
            # --------------------
            p_cp = pool.apply_async(prc_return, (self.bar_cp,))
            p_dcp = pool.apply_async(prc_return, (deepcopy(self.bar_dcp),))
            # Change returned objects
            ans_cp = p_cp.get()
            ans_dcp = p_dcp.get()
            prc_change(ans_cp)
            prc_change(ans_dcp)

            # Original objects must be unchanged
            self.assertFalse(self.bar_cp.lst[2] == 888)
            self.assertFalse(self.bar_dcp.lst[2] == 888)
            self.assertFalse(len(self.bar_cp.lst) == 6)
            self.assertFalse(len(self.bar_dcp.lst) == 6)

            # Returned objects must be changed
            self.assertTrue(ans_cp.lst[2] == 888)
            self.assertTrue(ans_dcp.lst[2] == 888)
            self.assertTrue(len(ans_cp.lst) == 6)
            self.assertTrue(len(ans_dcp.lst) == 6)

    def test_nothing(self):
        """ 'copy-on-write' makes no copy when the object is unchanged (and
        not returend) within a subprocess.
        """
        with Pool(processes=self.nproc) as pool:
            # Check with processes
            # --------------------
            p_cp = pool.apply_async(prc_nothing, (self.bar_cp,))
            p_dcp = pool.apply_async(prc_nothing, (deepcopy(self.bar_dcp),))
            # Change returned objects
            prc_change(self.bar_cp)
            prc_change(deepcopy(self.bar_dcp))
            p_cp.wait()
            p_dcp.wait()

            # Only deepcopy objects must be unchanged
            self.assertTrue(self.bar_cp.lst[2] == 888)
            self.assertFalse(self.bar_dcp.lst[2] == 888)
            self.assertTrue(len(self.bar_cp.lst) == 6)
            self.assertFalse(len(self.bar_dcp.lst) == 6)

    def test_pool(self):
        """ This test generates several child process that test whether two
        arrays have the same length (so no copy will be made). At the same
        time, the father process changes the two arrays such that an error is
        likely to be produces, when we do not pass a deepcopy of the argument.
        """
        nproc = 50
        nmax = 500

        proc = deque([], maxlen=nproc)
        obj = Foo([], np.empty((0,)))
        stop = False
        nsim = 0
        with Pool(processes=nproc) as pool:
            while not stop:
                # Create new process
                if len(proc) < nproc and nsim < nmax:
                    # The first option (copy) produces an error...
                    # proc.append(pool.apply_async(prc_eq, (obj,)))

                    # ...while the second (deepcopy) does not
                    proc.append(pool.apply_async(prc_eq, (deepcopy(obj),)))

                    nsim += 1
                # Check if a process is ready
                if proc[0].ready():
                    p = proc.popleft()
                    self.assertTrue(p.get())

                    # Change the object to cause an error
                    nmbr = np.random.randint(100, size=1)[0]
                    obj.lst.append(nmbr)
                    obj.arr = np.append(obj.arr, nmbr)

                    if len(proc) == 0 and nsim >= nmax:
                        stop = True
                else:
                    proc.rotate(-1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
