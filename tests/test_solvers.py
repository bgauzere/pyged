"""
Test for LSAP Solvers
"""

import pytest
np = pytest.importorskip("numpy")

from test_utils import *
from pyged.solvers import SolverLSAP

class TestSolver:
    """Tests for solver output"""

    def setup_method(self):
        self.solver = SolverLSAP()

    def test_constant_cost_function_solution(self):
        cm = constant_cost_matrix()
        rows_res, cols_res = self.solver.solve(cm)
        assert np.sum(cm[rows_res, cols_res]) == 2

    def test_riesen_cost_function_solution(self):
        cm = riesen_cost_matrix()
        rows_res, cols_res = self.solver.solve(cm)
        assert np.sum(cm[rows_res, cols_res]) == 6

    def test_neighborhood_cost_function_solution(self):
        cm = neighborhood_cost_matrix()
        rows_res, cols_res = self.solver.solve(cm)
        assert np.sum(cm[rows_res, cols_res]) == 9
