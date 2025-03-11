"""
Test for LSAP Solvers
"""

import sys
from unittest.mock import MagicMock


for lib in ("torch", "librariesImport", "gedlibpy", "sinkdiff.sinkdiff", "sinkdiff.sink_utils"):
    sys.modules[lib] = MagicMock()

import pytest
np = pytest.importorskip("numpy")

from pyged.solvers import SolverLSAP

def test_solver_lsap():
    C = np.array([
        [2, 1, 1, 1, np.inf, np.inf, np.inf],
        [1, 0, 0, np.inf, 1, np.inf, np.inf],
        [2, 1, 1, np.inf, np.inf, 1, np.inf],
        [0, 1, 1, np.inf, np.inf, np.inf, 1],
        [1, np.inf, np.inf, 0, 0, 0, 0],
        [np.inf, 1, np.inf, 0, 0, 0, 0],
        [np.inf, np.inf, 1, 0, 0, 0, 0]
    ])
    solver = SolverLSAP()
    rows, cols = solver.solve(C)
    assert np.sum(C[rows, cols]) == 2
