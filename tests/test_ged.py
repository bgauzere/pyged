"""
Test for the GED approximation
"""

from test_utils import *
import pyged.costfunctions as cf
from pyged.ged import GED

def test_ged():
    g1, g2 = load_test_graphs()
    ged = GED(cf.ConstantCostFunction(1, 1, 1, 1, comp_nodes, comp_edges))
    g12 = {"u1": "v1", "u2": "v2", "u3": None, "u4": "v3"}
    g21 = {"v1": "u1", "v2": "u2", "v3": "u4"}
    assert ged.ged(g1, g2, g12, g21)[0] == 4
