import numpy as np
from scipy.optimize import linear_sum_assignment
from pyged.costfunctions import ConstantCostFunction
from pyged.solvers import Solver, SolverLSAP


def computeBipartiteCostMatrix(G1, G2, cf=ConstantCostFunction(1, 3, 1, 3)):
    """Compute a Cost Matrix according to cost function cf"""
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


def getOptimalMapping(C, lsap_solver: Solver = SolverLSAP):
    """Compute an optimal linear mapping according to cost Matrix C
    inclure les progs C de Seb

    """
    rho, varrho = lsap_solver.solve(C)
    return rho, varrho
