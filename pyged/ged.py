from scipy.optimize import linear_sum_assignment
import networkx as nx
from pyged.costfunctions import CostFunction
from pyged.bipartiteGED import computeBipartiteCostMatrix, getOptimalMapping
from pyged.solvers import Solver, SolverLSAP
#
# Notes
# -TODO
# par défaut :             cf=ConstantCostFunction(1, 3, 1, 3),


class GED():
    def __init__(self, cf: CostFunction, solver: Solver = None):
        """
        Parameters
        ------------
        cf:CostFunction

        solver:


        """

        self.cf = cf
        if solver is None:
            solver = SolverLSAP()
        self.solver = solver

# if (method == 'Riesen'):
#     cf_bp = RiesenCostFunction(cf, lsap_solver=solver)
# elif (method == 'Neighboorhood'):
#     cf_bp = NeighboorhoodCostFunction(cf, lsap_solver=solver)
# elif (method == 'Basic'):
#     cf_bp = cf

    def ged(self, G1: nx.Graph, G2: nx.Graph, rho=None, varrho=None):
        """Compute Graph Edit Distance between G1 and G2 according to mapping
        encoded within rho and varrho.

        Graph's node must be indexed by a index starting at 0 which is
        used in rho and varrho

        Parameters
        ----------------

        G1, G2 : networkx graphs


        """
        # TODO : à sortir
        if ((rho is None) or (varrho is None)):
            C = computeBipartiteCostMatrix(G1, G2, self.cf)
            r, v = getOptimalMapping(C, lsap_solver=self.solver)
            rho, varrho = convert_mapping(r, v, G1, G2)

        # rho : V1 -> V2
        # varrho : V2 -> V1

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


def convert_mapping(rho, varrho, G1, G2):
    """
    Convert a mapping from nodes index (int) to a mapping between
    nodes id (real node identifier in networkx)
    returns: two dicts

    Parameters
    --------------
    rho : rho[i] = phi(i), i \in G1
    varrho : varrho[j] = phi^-1(j), j \in G2
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
