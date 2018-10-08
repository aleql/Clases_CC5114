import unittest

import numpy as np

from clase1 import perceptron, p_sum


class Test_p_and_test(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        self.p_and = perceptron([1, 1], -1.5)

    def test_testablaVerdad(self):
        assert self.p_and.feed(np.array([1, 1])) == 1, "p_and not calculating values correctly"
        assert self.p_and.feed(np.array([0, 0])) == 0, "p_and not calculating values correctly"
        assert self.p_and.feed(np.array([0, 1])) == 0, "p_and not calculating values correctly"
        assert self.p_and.feed(np.array([0, 0])) == 0, "p_and not calculating values correctly"


class Test_p_or_test(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        self.p_or = perceptron([1, 1], -0.5)

    def test_tabla_verdad(self):
        assert self.p_or.feed(np.array([1, 1])) == 1, "p_or not calculating values correctly"
        assert self.p_or.feed(np.array([1, 0])) == 1, "p_or not calculating values correctly"
        assert self.p_or.feed(np.array([0, 1])) == 1, "p_or not calculating values correctly"
        assert self.p_or.feed(np.array([0, 0])) == 0, "p_or not calculating values correctly"

class Test_p_nand_test(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        self.p_nand = perceptron([-2, -2], 3)

    def test_tabla_verdad(self):
        assert self.p_nand.feed(np.array([1, 1])) == 0, "p_nand not calculating values correctly"
        assert self.p_nand.feed(np.array([0, 0])) == 1, "p_nand not calculating values correctly"
        assert self.p_nand.feed(np.array([0, 1])) == 1, "p_nand not calculating values correctly"
        assert self.p_nand.feed(np.array([0, 0])) == 1, "p_nand not calculating values correctly"

class Test_p_not_test(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        self.p_not = perceptron(-1, 1)

    def test_tabla_verdad(self):
        assert self.p_not.feed(1) == 0, "p_not not calculating values correctly"
        assert self.p_not.feed(0) == 1, "p_not not calculating values correctly"

class Test_p_sum(unittest.TestCase):

    def test_tabla_verdad(self):
        assert p_sum(1, 1)[0] == 0, "p_sum not calculating values correctly"
        assert p_sum(1, 0)[0] == 1, "p_sum not calculating values correctly"
        assert p_sum(0, 1)[0] == 1, "p_sum not calculating values correctly"
        assert p_sum(0, 0)[0] == 0, "p_sum not calculating values correctly"

        assert p_sum(1, 1)[1] == 1, "p_sum not calculating values correctly"
        assert p_sum(1, 0)[1] == 0, "p_sum not calculating values correctly"
        assert p_sum(0, 1)[1] == 0, "p_sum not calculating values correctly"
        assert p_sum(0, 0)[1] == 0, "p_sum not calculating values correctly"

