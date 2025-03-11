"""
Test for the GED approximation
"""

from test_utils import *
import pyged.costfunctions as cf
from pyged.ged import GED

class TestGED:
    """Tests the GED class"""

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

    def test_constant_cost_function_ged(self):
        ged = GED(self.ccf)
        assert ged.ged(self.g1, self.g2)[0] == 4
