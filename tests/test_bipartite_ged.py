"""
Tests for the `bipartite_ged` module
"""

import sys
from unittest.mock import MagicMock


for lib in ("torch", "librariesImport", "gedlibpy", "sinkdiff.sinkdiff", "sinkdiff.sink_utils"):
    sys.modules[lib] = MagicMock()

import pytest
import networkx as nx
import pyged.bipartiteGED as bpged
import pyged.costfunctions as cf
from tests.test_utils import *

np = pytest.importorskip("numpy")

class TestComputeBipartiteCostMatrix:
    """Tests cost matrix"""

    def setup_method(self):
        self.g1, self.g2 = load_test_graphs()
        self.ccf = cf.ConstantCostFunction(
            1,
            1,
            1,
            1,
            comp_nodes,
            comp_edges
        )

    def test_constant_cost_function_cost_matrix(self):
        true_cm = constant_cost_matrix()
        cm = bpged.compute_bipartite_cost_matrix(self.g1, self.g2, self.ccf)
        assert np.array_equal(cm, true_cm, equal_nan=True)

    def test_riesen_cost_function_cost_matrix(self):
        rcf = cf.RiesenCostFunction(self.ccf)
        true_cm = riesen_cost_matrix()
        cm = bpged.compute_bipartite_cost_matrix(self.g1, self.g2, rcf)
        assert np.array_equal(cm, true_cm, equal_nan=True)

    def test_neighborhood_cost_function_cost_matrix(self):
        ncf = cf.NeighborhoodCostFunction(self.ccf)
        true_cm = neighborhood_cost_matrix()
        cm = bpged.compute_bipartite_cost_matrix(self.g1, self.g2, ncf)
        assert np.array_equal(cm, true_cm, equal_nan=True)

class TestGetOptimalMapping:
    """Tests for solver output"""

    def test_constant_cost_function_solution(self):
        cm = constant_cost_matrix()
        cols_res, rows_res = bpged.get_optimal_mapping(cm)
        assert np.sum(cm[rows_res, cols_res]) == 2

    def test_riesen_cost_function_solution(self):
        cm = riesen_cost_matrix()
        cols_res, rows_res = bpged.get_optimal_mapping(cm)
        assert np.sum(cm[rows_res, cols_res]) == 6
