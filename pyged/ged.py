"""
Graph Edit Distance module

Defines a class computing GED between 2 graphs
"""

from typing import Optional, Tuple, Dict, Any, Iterable
import numpy as np
import networkx as nx
from pyged.costfunctions import CostFunction, ConstantCostFunction
from pyged.bipartiteGED import computeBipartiteCostMatrix, getOptimalMapping
from pyged.solvers import Solver, SolverLSAP


class GED():
    """Graph Edit Distance class
    
    Computes the GED of 2 grahs given a cost fucntion ans a LSAP solver
    """

    def __init__(
            self,
            cf: CostFunction = ConstantCostFunction(1, 3, 1, 3),
            solver: Solver = SolverLSAP()
        ):
        """Creates a Graph Edit Ditance computer

        Parameters
        ----------
        cf: CostFunction
            Functions defining the cost of edit operations
            Uses a constant cost function by default of costs
            * 1 for any substitution between nodes or edges
            * 3 for any deletion/insertion of nodes or edges
        solver: Solver
            Solver for the LSAP
            By default, a solver using Hungarian Algorithm will be used
        """
        self.cf = cf
        self.solver = solver


    def ged(
            self,
            G1: nx.Graph,
            G2: nx.Graph,
            rho: Optional[Dict[Any, Any|None]] = None,
            varrho: Optional[Dict[Any, Any|None]] = None
        ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute Graph Edit Distance between `G1` and `G2`
        according to mapping encoded within rho and varrho.

        Graph's node must be indexed by a index starting
        at 0 which is used in rho and varrho

        Parameters
        ----------
        G1, G2 : networkx graphs
            Graphs between which the GED is computed
        rho, varrho : dictionnaries of nodes (Any) to nodes (Any) (Optional)
            result of the matching between nodes
            if `None`, they will be computed

        Returns
        -------
        ged : float
            the Graph Edit Distance Upper Bound
        rho, barrho : dictionnaries of nodes (Any) to nodes (Any)
            result of the matching between nodes
        """
        # TODO : à sortir
        if ((rho is None) or (varrho is None)):
            C = computeBipartiteCostMatrix(G1, G2, self.cf)
            r, v = getOptimalMapping(C, lsap_solver=self.solver)
            rho, varrho = convert_mapping(r, v, G1, G2)

        # rho : V1 -> V2
        # varrho : V2 -> V1
        # print(f"{rho =}")
        ged = 0
        for v in G1.nodes():
            phi_i = rho[v]
            if (phi_i is None):
                ged += self.cf.cnd(v, G1)
            else:
                ged += self.cf.cns(v, phi_i, G1, G2)
        for u in G2.nodes():
            phi_j = varrho[u]
            if (phi_j is None):
                ged += self.cf.cni(u, G2)

        for e in G1.edges():
            i = e[0]
            j = e[1]
            phi_i = rho[i]
            phi_j = rho[j]
            if (phi_i is not None) and (phi_j is not None):
                # il est possible que l'arete existe dans G2
                mappedEdge = len(list(filter(lambda x: True if
                                             x == phi_j else False, G2[phi_i])))
                if (mappedEdge):
                    e2 = [phi_i, phi_j]
                    min_cost = min(self.cf.ces(e, e2, G1, G2),
                                   self.cf.ced(e, G1) + self.cf.cei(e2, G2))
                    ged += min_cost
                else:
                    ged += self.cf.ced(e, G1)
            else:
                ged += self.cf.ced(e, G1)
        for e in G2.edges():
            i = e[0]
            j = e[1]
            phi_i = varrho[i]
            phi_j = varrho[j]
            if (phi_i is not None) and (phi_j is not None):
                mappedEdge = len(list(filter(lambda x: True if x == phi_j
                                             else False, G1[phi_i])))
                if (not mappedEdge):
                    ged += self.cf.cei(e, G2)
            else:
                ged += self.cf.ced(e, G2)
        return ged, rho, varrho


def convert_mapping(
        rho: Iterable[int],
        varrho: Iterable[int],
        G1: nx.Graph,
        G2: nx.Graph
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """Convert a mapping from nodes index (int) to a mapping
    between nodes id (real node identifier in networkx)

    Parameters
    ----------
    rho, varrho : Iterable of ints
        Lists of indices, results of nodes mapping
        for each node of index i in G1, rho[i] if the matched node in G2
        varrho is the reverse list
    G1, G2 : networkx.Graph
        Graphs between which we map the nodes

    Returns
    -------
    rho, barrho : dictionnaries of nodes (Any) to nodes (Any)
        converted result of the mapping into dicts
    """
    rho_dict = {}
    varrho_dict = {}
    nodes_list_G1 = list(G1.nodes())
    nodes_list_G2 = list(G2.nodes())

    n = G1.number_of_nodes()
    m = G2.number_of_nodes()

    for i, rho_i in enumerate(rho[:n]):
        if (rho_i >= m):
            rho_dict[nodes_list_G1[i]] = None
        else:
            rho_dict[nodes_list_G1[i]] = nodes_list_G2[rho_i]

    for j, varrho_j in enumerate(varrho[:m]):
        if (varrho_j >= n):
            varrho_dict[nodes_list_G2[j]] = None
        else:
            varrho_dict[nodes_list_G2[j]] = nodes_list_G1[varrho_j]

    return rho_dict, varrho_dict
