"""
Functions for solving the LSAP
"""

from typing import Tuple
import numpy as np
import networkx as nx
from pyged.costfunctions import CostFunction, ConstantCostFunction
from pyged.solvers import Solver, SolverLSAP


def computeBipartiteCostMatrix(
        G1: nx.Graph,
        G2: nx.Graph,
        cf: CostFunction = ConstantCostFunction(1, 3, 1, 3)
    ) -> np.ndarray:
    """Compute a Cost Matrix according to cost function cf
    
    Parameters
    ----------
    G1, G2 : nx.Graph
        Graphs between which the GED is calculated
    cf : CostFunction
        Cost function to build the cost matrix
    
    Returns
    -------
    The cost matrix from the given function as a `numpy` array
    """
    n = G1.number_of_nodes()
    m = G2.number_of_nodes()
    nm = n + m
    C = np.ones([nm, nm])*np.inf
    C[n:, m:] = 0

    for i, u in enumerate(G1.nodes()):
        for j, v in enumerate(G2.nodes()):
            cost = cf.cns(u, v, G1, G2)
            C[i, j] = cost

    for i, v in enumerate(G1.nodes()):
        C[i, m + i] = cf.cnd(v, G1)

    for i, v in enumerate(G2.nodes()):
        C[n + i, i] = cf.cni(v, G2)
    return C


def getOptimalMapping(
        C: np.ndarray,
        lsap_solver: Solver = SolverLSAP
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an optimal linear mapping according to cost Matrix C

    Parameters
    ----------
    C : np.ndarray
        The cost matrix for the LSAP
    lsap_solver : Solver
        Solves the LSAP given a cost matrix

    Returns
    -------
    rho, varrho, `numpy` arrays for columns and lines mapping indices

    TODO inclure les progs C de Seb
    """
    rho, varrho = lsap_solver.solve(C)
    return rho, varrho
