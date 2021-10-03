import numpy as np
from scipy.optimize import linear_sum_assignment
import sys

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


class ConstantCostFunction:
    """ Define a symmetric constant cost fonction for edit operations """

    def __init__(self, cns, cni, ces, cei):
        self.cns_ = cns
        self.cni_ = self.cnd_ = cni
        self.ces_ = ces
        self.cei_ = self.ced_ = cei

    def cns(self, node_u, node_v, g1, g2, label_to_compare="atom"):
        """ return substitution edit operation cost between
        node_u of G1 and node_v of G2"""
        label_u = g1.nodes[node_u].get(label_to_compare, None)
        label_v = g2.nodes[node_v].get(label_to_compare, None)
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

    def __init__(self, cf, lsap_solver=linear_sum_assignment):
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

    def __init__(self, cf, lsap_solver=linear_sum_assignment):
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
