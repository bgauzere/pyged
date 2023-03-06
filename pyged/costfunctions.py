import sys
from typing import Protocol, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx

"""
   Classes encoding cost functions

A cost function class must provide elementary costs for elementary edit operations. Namely:
- cns : node substitution cost
- cnd : node deletion cost
- cni : node insertion cost

- ces : edge substitution cost
- ced : edge deletion cost
- cei : edge insertion cost
"""


class CostFunction(Protocol):
    def cns(self, node_u, node_v, g1: nx.Graph, g2: nx.Graph) -> float:
        """Returns the substitution cost between node_u and node_v in g1 and g2 resp.

        Parameters
        ----------
        node_u : index of node u in g1
            index of node u in g1
        node_v : 
            index of node v in g2
        g1 : networkx.Graph
            Graph containing u
        g2 : networkx.Graph
            Graph containing v

        Returns
        ---------
        a positive float value
        """
        ...

    def cnd(self, node_u: Any, g1: nx.Graph) -> float:
        """Returns the deletion cost of node_u in g1.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        ---------
        a positive float value
        """
        ...

    def cni(self, node_u: Any, g1: nx.Graph) -> float:
        """Returns the insertion cost of node_u in g1.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        ---------
        a positive float value
        """
        ...

    def ces(self, e1: tuple[Any, Any], e2: tuple[Any, Any],
            g1: nx.Graph, g2: nx.Graph) -> float:
        """Returns the substitution cost between edge e1 and edge e2 in g1 and g2 resp.

        Parameters
        ----------
        e1 : tuple[Any,Any]
            edge in g1
        e2 : tuple[Any,Any] : 
            edge in g2
        g1 : networkx.Graph
            Graph containing u
        g2 : networkx.Graph
            Graph containing v

        Returns
        ---------
        a positive float value
        """
        ...

    def ced(self, e1: tuple[Any, Any], g1: nx.Graph) -> float:
        """Returns the deletion cost of edge e1 in g1.

        Parameters
        ----------
        e1 : tuple[Any,Any]
            edge to delete in G1
        g1 : networkx.Graph
            Graph containing e1

        Returns
        ---------
        a positive float value
        """
        ...

    def cei(self, e1: tuple[Any, Any], g1: nx.Graph) -> float:
        """Returns the insertion cost of edge e1 in g1.

        Parameters
        ----------
        e1 : tuple[Any,Any]
            edge to insert in G1
        g1 : networkx.Graph
            Graph containing e1

        Returns
        ---------
        a positive float value
        """
        ...


class ConstantCostFunction:
    """Define a symmetric constant cost fonction for edit operations

    TODO : transformer label_to_compare en fonction de test d'égalité des labels
    de noeuds et d'aretes. Avec une valeur par défaut sur un label particulier

    """

    def __init__(self, cns, cni, ces, cei, label_to_compare="atom"):
        self.cns_ = cns
        self.cni_ = self.cnd_ = cni
        self.ces_ = ces
        self.cei_ = self.ced_ = cei
        self.label_to_compare = label_to_compare

    def cns(self, node_u, node_v, g1, g2):
        """ return substitution edit operation cost between
        node_u of G1 and node_v of G2"""
        label_u = g1.nodes[node_u].get(self.label_to_compare, None)
        label_v = g2.nodes[node_v].get(self.label_to_compare, None)
        if (label_u == label_v):
            return 0
        else:
            return self.cns_

    def cnd(self, u, G1):
        return self.cnd_

    def cni(self, v, G2):
        return self.cni_

    def ces(self, e1, e2, G1, G2):
        """
        An edge is a 2-tuple : [ firstnode, secondnode]
        """
        have_same_label = True
        for label in G1[e1[0]][e1[1]].keys():   # should have same labels
            have_same_label &= (G1[e1[0]][e1[1]][label] !=
                                G2[e2[0]][e2[1]][label])

        return have_same_label * self.ces_

    def ced(self, e1, G1):
        return self.ced_

    def cei(self, e2, G2):
        return self.cei_


class RiesenCostFunction():
    """ Cost function associated to the computation of a cost matrix between nodes for LSAP"""

    def __init__(self, cf: CostFunction,
                 lsap_solver=linear_sum_assignment):
        self.cf_ = cf
        self.lsap_solver_ = lsap_solver

    def cns(self, u, v, G1, G2):
        """ u et v sont des id de noeuds """
        n = len(G1[u])
        m = len(G2[v])
        sub_C = np.ones([n+m, n+m]) * sys.maxsize
        sub_C[n:, m:] = 0
        i = 0
        l_nbr_u = G1[u]
        l_nbr_v = G2[v]
        for nbr_u in l_nbr_u:
            j = 0
            e1 = [u, nbr_u, G1[u][nbr_u]]
            for nbr_v in G2[v]:
                e2 = [v, nbr_v, G2[v][nbr_v]]
                sub_C[i, j] = self.cf_.ces(e1, e2, G1, G2)
                j += 1
            i += 1

        i = 0
        for nbr_u in l_nbr_u:
            sub_C[i, m+i] = self.cf_.ced([u, nbr_u], G1)
            i += 1

        j = 0
        for nbr_v in l_nbr_v:
            sub_C[n+j, j] = self.cf_.cei([v, nbr_v], G2)
            j += 1
        row_ind, col_ind = self.lsap_solver_(sub_C)
        cost = np.sum(sub_C[row_ind, col_ind])
        return self.cf_.cns(u, v, G1, G2) + cost

    def cnd(self, u, G1):
        cost = 0
        for nbr in G1[u]:
            cost += self.cf_.ced([u, nbr], G1)

        return self.cf_.cnd(u, G1) + cost

    def cni(self, v, G2):
        cost = 0
        for nbr in G2[v]:
            cost += self.cf_.cei([v, nbr], G2)
        return self.cf_.cni(v, G2) + cost


class NeighboorhoodCostFunction():
    """ Cost function associated to the computation of a cost matrix between nodes for LSAP"""

    def __init__(self, cf: CostFunction, lsap_solver=linear_sum_assignment):
        self.cf_ = cf
        self.lsap_solver_ = lsap_solver

    def cns(self, u, v, G1, G2):
        """ u et v sont des id de noeuds """
        n = len(G1[u])
        m = len(G2[v])
        sub_C = np.ones([n+m, n+m]) * sys.maxsize
        sub_C[n:, m:] = 0
        i = 0
        l_nbr_u = G1[u]
        l_nbr_v = G2[v]
        for nbr_u in l_nbr_u:
            j = 0
            e1 = [u, nbr_u]
            for nbr_v in G2[v]:
                e2 = [v, nbr_v]
                sub_C[i, j] = self.cf_.ces(e1, e2, G1, G2)
                sub_C[i, j] += self.cf_.cns(nbr_u, nbr_v, G1, G2)
                j += 1
            i += 1

        i = 0
        for nbr_u in l_nbr_u:
            sub_C[i, m+i] = self.cf_.ced([u, nbr_u], G1)
            sub_C[i, m+i] += self.cf_.cnd(nbr_u, G1)
            i += 1

        j = 0
        for nbr_v in l_nbr_v:
            sub_C[n+j, j] = self.cf_.cei([v, nbr_v], G2)
            sub_C[n+j, j] += self.cf_.cni(nbr_v, G2)
            j += 1

        row_ind, col_ind = self.lsap_solver_(sub_C)
        cost = np.sum(sub_C[row_ind, col_ind])
        return self.cf_.cns(u, v, G1, G2) + cost

    def cnd(self, u, G1):
        cost = 0
        for nbr in G1[u]:
            cost += self.cf_.ced([u, nbr], G1)
        return self.cf_.cnd(u, G1) + cost

    def cni(self, v, G2):
        cost = 0
        for nbr in G2[v]:
            cost += self.cf_.cei([v, nbr, G2], G2)
        return self.cf_.cni(v, G2) + cost
