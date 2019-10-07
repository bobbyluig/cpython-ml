#!/usr/bin/env python3
import os
import lab
import json
import unittest
import copy

import sys
# import gc

# Ran 6 tests in 2.110s
# gc.set_threshold(700, 10, 10)

# Ran 6 tests in 2.026s
# gc.set_threshold(10000, 10, 10)

# Ran 6 tests in 1.907s
# gc.disable()

sys.setrecursionlimit(10000)

TEST_DIRECTORY = os.path.join(os.path.dirname(__file__), 'test_inputs')


class TestSat(unittest.TestCase):
    def opencase(self, casename):
        with open(os.path.join(TEST_DIRECTORY, casename + ".json")) as f:
            cnf = json.load(f)
            return [[(variable, polarity)
                     for variable, polarity in clause]
                    for clause in cnf]

    def satisfiable(self, casename):
        cnf = self.opencase(casename)
        asgn = lab.satisfying_assignment(copy.deepcopy(cnf))
        self.assertIsNotNone(asgn)
        # Check that every clause has some literal appearing in the assignment.
        self.assertTrue(all(any(variable in asgn and asgn[variable] == polarity
                                for variable, polarity in clause)
                            for clause in cnf))

    def unsatisfiable(self, casename):
        cnf = self.opencase(casename)
        asgn = lab.satisfying_assignment(copy.deepcopy(cnf))
        self.assertIsNone(asgn)

    def test_A_10_3_100(self):
        self.unsatisfiable('10_3_100')

    def test_B_20_3_1000(self):
        self.unsatisfiable('20_3_1000')

    def test_C_100_10_100(self):
        self.satisfiable('100_10_100')

    def test_D_1000_5_10000(self):
        self.unsatisfiable('1000_5_10000')

    def test_E_1000_10_1000(self):
        self.satisfiable('1000_10_1000')

    def test_F_1000_11_1000(self):
        self.satisfiable('1000_11_1000')


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
