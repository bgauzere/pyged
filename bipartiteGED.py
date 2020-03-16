import numpy as np
from scipy.optimize import linear_sum_assignment
from .costfunctions import ConstantCostFunction


def computeBipartiteCostMatrix(G1, G2, cf=ConstantCostFunction(1, 3, 1, 3)):
    """Compute a Cost Matrix according to cost function cf"""
    n = G1.number_of_nodes()
    m = G2.number_of_nodes()
    nm = n + m
    C = np.ones([nm, nm])*np.inf
    C[n:, m:] = 0

    for u in G1.nodes():
        for v in G2.nodes():
            cost = cf.cns(u, v, G1, G2)
            C[u, v] = cost

    for v in G1.nodes():
        C[v, m + v] = cf.cnd(v, G1)

    for v in G2.nodes():
        C[n + v, v] = cf.cni(v, G2)
    return C


def getOptimalMapping(C, lsap_solver=linear_sum_assignment):
    """Compute an optimal linear mapping according to cost Matrix C
    inclure les progs C de Seb

    """
    row_ind, col_ind = lsap_solver(C)
    return col_ind, row_ind[np.argsort(col_ind)]
