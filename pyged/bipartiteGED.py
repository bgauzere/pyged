"""
Functions for solving the LSAP and creating an optimal node matching
"""

from typing import Tuple, Dict, Any, Iterable
import numpy as np
import networkx as nx
from pyged.costfunctions import CostFunction, ConstantCostFunction
from pyged.solvers import Solver, SolverLSAP


def compute_bipartite_cost_matrix(
        G1: nx.Graph,
        G2: nx.Graph,
        cf: CostFunction = ConstantCostFunction(1, 3, 1, 3)
    ) -> np.ndarray:
    """Compute a Cost Matrix according to cost function `cf`
    
    Parameters
    ----------
    G1, G2 : nx.Graph
        Graphs between which the GED is calculated
    cf : CostFunction
        Cost function to build the cost matrix
    
    Returns
    -------
    C: np.ndarray
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


#  _____ ___  ____   ___  
# |_   _/ _ \|  _ \ / _ \ 
#   | || | | | | | | | | |
#   | || |_| | |_| | |_| |
#   |_| \___/|____/ \___/ 
# TODO Retirer cette fonction
def get_optimal_mapping(
        C: np.ndarray,
        lsap_solver: Solver = SolverLSAP()
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
    rho, varrho: np.ndarray
        `numpy` arrays for columns and lines mapping indices

    TODO inclure les progs C de Seb
    """
    rho, varrho = lsap_solver.solve(C)
    return rho, varrho


def convert_mapping(
        rho: Iterable[int],
        varrho: Iterable[int],
        g1: nx.Graph,
        g2: nx.Graph
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """Convert a mapping from nodes index (int) to a mapping
    between nodes id (real node identifier in networkx)

    Parameters
    ----------
    rho, varrho : Iterable of ints
        Lists of indices, results of nodes mapping
        for each node of index i in g1, rho[i] if
        the index of matched node in g2
        varrho is the reversed list
    g1_to_g2, g2_to_g1 : networkx.Graph
        Graphs between which we map the nodes

    Returns
    -------
    rho, varrho : dictionnaries of nodes (Any) to nodes (Any)
        converted result of the mapping into dicts
    """
    assert len(rho) == len(varrho)
    nodes1, nodes2 = list(g1.nodes()), list(g2.nodes())
    g1_to_g2, g2_to_g1 = {}, {}
    for g1_index, g2_index in zip(rho, varrho):
        if g1_index < len(nodes1):
            g1_to_g2[nodes1[g1_index]] = nodes2[g2_index] if g2_index < len(nodes2) else None
        if g2_index < len(nodes2):
            g2_to_g1[nodes2[g2_index]] = nodes1[g1_index] if g1_index < len(nodes1) else None
    return g1_to_g2, g2_to_g1

