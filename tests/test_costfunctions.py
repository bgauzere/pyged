"""
Tests for module `costfunctions`
"""

import networkx as nx
import pyged.costfunctions as cf
from tests.test_utils import *

class TestConstantCostFunction:
    """Tests for `pyged.costfunctions.ConstantCostFunction`"""

    def setup_method(self):
        self.g1, self.g2 = load_test_graphs()
        self.cf = cf.ConstantCostFunction(
            1,
            1,
            1,
            1,
            comp_nodes,
            comp_edges
        )

    def test_cni(self):
        assert self.cf.cni("v1", self.g2) == 1

    def test_cnd(self):
        assert self.cf.cnd("u1", self.g1) == 1

    def test_cns_diff(self):
        assert self.cf.cns("u1", "v2", self.g1, self.g2) == 1

    def test_cns_same(self):
        assert self.cf.cns("u4", "v1", self.g1, self.g2) == 0

    def test_cei(self):
        assert self.cf.cei(("v1", "v2"), self.g2) == 1

    def test_ced(self):
        assert self.cf.ced(("u1", "u2"), self.g1) == 1

    def test_ces_diff(self):
        assert self.cf.ces(("u2", "u3"), ("v2", "v3"), self.g1, self.g2) == 1

    def test_ces_same(self):
        assert self.cf.ces(("u3", "u4"), ("v1", "v2"), self.g1, self.g2) == 0

class TestRiesenCostFunction:
    """Tests for `pyged.costfunctions.RiesenCostFunction`"""

    def setup_method(self):
        self.g1, self.g2 = load_test_graphs()
        self.cf = cf.RiesenCostFunction(cf.ConstantCostFunction(
            1,
            1,
            1,
            1,
            comp_nodes,
            comp_edges
        ))

    def test_cni(self):
        assert self.cf.cni("v1", self.g2) == 2

    def test_cnd(self):
        assert self.cf.cnd("u1", self.g1) == 2

    def test_cns_diff(self):
        assert self.cf.cns("u1", "v2", self.g1, self.g2) == 2

    def test_cns_same(self):
        assert self.cf.cns("u4", "v1", self.g1, self.g2) == 1

    def test_cei(self):
        assert self.cf.cei(("v1", "v2"), self.g2) == 1

    def test_ced(self):
        assert self.cf.ced(("u1", "u2"), self.g1) == 1

    def test_ces_diff(self):
        assert self.cf.ces(("u2", "u3"), ("v2", "v3"), self.g1, self.g2) == 1

    def test_ces_same(self):
        assert self.cf.ces(("u3", "u4"), ("v1", "v2"), self.g1, self.g2) == 0
