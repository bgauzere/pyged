"""
Graph Edit Distance module

Defines a class computing GED between 2 graphs
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import networkx as nx
from pyged.costfunctions import CostFunction, ConstantCostFunction
from pyged.bipartiteGED import compute_bipartite_cost_matrix, convert_mapping
from pyged.solvers import Solver, SolverLSAP


class GED():
    """Graph Edit Distance class
    
    Computes an upper bound of the GED between 2 graphs given a cost function.

    This algorithm is based on 
    *Structural Pattern Recognition with Graph Edit Distance* [1]_.

    References
    ----------
    .. [1] K. Riesen, Structural Pattern Recognition with
       Graph Edit Distance, Switzerland, Springer, 2015

    See Also
    --------
    pyged.costfunctions
    pyged.solvers
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

        Examples
        --------
        >>> import networkx as nx
        >>> import pyged
        >>> g1 = nx.complete_graph(5)
        >>> g2 = nx.complete_graph(6)
        >>> ged = pyged.ged.GED(pyged.costfunctions.ConstantCostFunction(1, 1, 1, 1))
        >>> ged.ged(g1, g2)[0]
        6
        """
        # TODO : à sortir
        if ((rho is None) or (varrho is None)):
            C = compute_bipartite_cost_matrix(G1, G2, self.cf)
            r, v = self.solver.solve(C)
            rho, varrho = convert_mapping(r, v, G1, G2)

        ccf = self.cf if isinstance(self.cf, ConstantCostFunction) else self.cf.ccf
        ged = 0
        for v in G1.nodes():
            phi_i = rho[v]
            if phi_i is None:
                ged += ccf.cnd(v, G1)
            else:
                ged += ccf.cns(v, phi_i, G1, G2)
        for u in G2.nodes():
            phi_j = varrho[u]
            if phi_j is None:
                ged += ccf.cni(u, G2)

        for e in G1.edges():
            i = e[0]
            j = e[1]
            phi_i = rho[i]
            phi_j = rho[j]
            if (phi_i is not None) and (phi_j is not None):
                # il est possible que l'arete existe dans G2
                mappedEdge = len(list(filter(lambda x: True if
                                             x == phi_j else False, G2[phi_i])))
                if mappedEdge:
                    e2 = [phi_i, phi_j]
                    min_cost = min(ccf.ces(e, e2, G1, G2),
                                   ccf.ced(e, G1) + ccf.cei(e2, G2))
                    ged += min_cost
                else:
                    ged += ccf.ced(e, G1)
            else:
                ged += ccf.ced(e, G1)
        for e in G2.edges():
            i = e[0]
            j = e[1]
            phi_i = varrho[i]
            phi_j = varrho[j]
            if (phi_i is not None) and (phi_j is not None):
                mappedEdge = len(list(filter(lambda x: True if x == phi_j
                                             else False, G1[phi_i])))
                if not mappedEdge:
                    ged += ccf.cei(e, G2)
            else:
                ged += ccf.ced(e, G2)
        return ged, rho, varrho
