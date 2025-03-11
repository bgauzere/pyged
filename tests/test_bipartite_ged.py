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
        rows_res, cols_res = bpged.get_optimal_mapping(cm)
        assert np.sum(cm[rows_res, cols_res]) == 2

    def test_riesen_cost_function_solution(self):
        cm = riesen_cost_matrix()
        rows_res, cols_res = bpged.get_optimal_mapping(cm)
        assert np.sum(cm[rows_res, cols_res]) == 6

    def test_neighborhood_cost_function_solution(self):
        cm = neighborhood_cost_matrix()
        rows_res, cols_res = bpged.get_optimal_mapping(cm)
        assert np.sum(cm[rows_res, cols_res]) == 9

def test_convert_mapping():
    g1, g2 = load_test_graphs()
    cols = np.array([0, 1, 2, 3, 4, 5, 6])
    rows = np.array([0, 1, 5, 2, 3, 4, 6])
    true_g12 = {"u1": "v1", "u2": "v2", "u3": None, "u4": "v3"}
    true_g21 = {"v1": "u1", "v2": "u2", "v3": "u4"}
    g12, g21 = bpged.convert_mapping(cols, rows, g1, g2)
    assert g12 == true_g12 and g21 == true_g21
