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

import sys
from typing import Protocol, Any, Callable, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx


class CostFunction(Protocol):
    """`CostFuncion` protocol
    
    Designs the methods for classes defining cost functions
    """

    def cns(self, node_u: Any, node_v: Any, g1: nx.Graph, g2: nx.Graph) -> float:
        """Returns the substitution cost between `node_u` and `node_v` in `g1` and `g2` resp.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        node_v : Any
            index of node v in g2
        g1 : networkx.Graph
            Graph containing u
        g2 : networkx.Graph
            Graph containing v

        Returns
        -------
        A positive float value
        """
        ...

    def cnd(self, node_u: Any, g1: nx.Graph) -> float:
        """Returns the deletion cost of `node_u` in `g1`.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        -------
        A positive float value
        """
        ...

    def cni(self, node_u: Any, g1: nx.Graph) -> float:
        """Returns the insertion cost of `node_u` in `g1`.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        -------
        A positive float value
        """
        ...

    def ces(self, e1: Tuple[Any, Any], e2: Tuple[Any, Any],
            g1: nx.Graph, g2: nx.Graph) -> float:
        """Returns the substitution cost between edge `e1` and edge `e2` in `g1` and `g2` resp.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge in g1
        e2 : Tuple[Any, Any]
            edge in g2
        g1 : networkx.Graph
            Graph containing e1
        g2 : networkx.Graph
            Graph containing e2

        Returns
        -------
        A positive float value
        """
        ...

    def ced(self, e1: Tuple[Any, Any], g1: nx.Graph) -> float:
        """Returns the deletion cost of edge `e1` in `g1`.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge to delete in g1
        g1 : networkx.Graph
            Graph containing e1

        Returns
        -------
        A positive float value
        """
        ...

    def cei(self, e2: Tuple[Any, Any], g2: nx.Graph) -> float:
        """Returns the insertion cost of edge `e2` in `g2`.

        Parameters
        ----------
        e2 : Tuple[Any, Any]
            edge to insert in g2
        g2 : networkx.Graph
            Graph containing e2

        Returns
        -------
        A positive float value
        """
        ...


class ConstantCostFunction:
    """Define a symmetric constant cost fonction for edit operations"""

    def __init__(
            self,
            cns: int|float,
            cni: int|float,
            ces: int|float,
            cei: int|float,
            node_comp: Optional[
                Callable[[Any, Any, nx.Graph, nx.Graph], bool]
            ] = None,
            edge_comp: Optional[
                Callable[[Tuple[Any, Any], Tuple[Any, Any], nx.Graph, nx.Graph], bool]
            ] = None
        ):
        """Creates a constant cost for edit operations

        Parameters
        ----------
        cns : int|float
            Node substitution cost
        cni : int|float
            Node insertion & deletion cost
        ces : int|float
            Edge substitution cost
        cei : int|float
            Edge insertion & deletion cost
        node_comp : Callable(Any, Any, nx.Graph, nx.Graph) -> bool
            Boolean function for node comparison. The function will be called this way :

                node_comp(u, v, g1, g2)

            with `u` and `v`, nodes from resp. `g1` and `g2`

            Shoud return `True` if nodes `u` and `v` are the same

        edge_comp : Callable(Tuple[Any, Any], Tuple[Any, Any], nx.Graph, nx.Graph) -> bool
            Boolean function for edge comparison. The function will be called this way :

                edge_comp(e1, e2, g1, g2)

            with `e1` and `e2`, edges from resp. `g1` and `g2`

            Should return `True` if edges `e1` and `e2` are the same

        Notes
        -----
        * An edge is a `Tuple` of 2 nodes (`Any`)
        * `node_comp` and `edge_comp` should return `True` whether the nodes/edges are the same
        * These function will make substitution cost at 0 if it's the case
        """
        self.cns_ = cns
        self.cni_ = self.cnd_ = cni
        self.ces_ = ces
        self.cei_ = self.ced_ = cei
        self.compare_nodes = node_comp if node_comp is not None\
            else (lambda u, v, g1, g2: self.cns_)
        self.compare_edges = edge_comp if edge_comp is not None\
            else (lambda e1, e2, g1, g2: self.ces_)

    def cns(self, node_u: Any, node_v: Any, g1: nx.Graph, g2: nx.Graph) -> float:
        """Returns the substitution cost between `node_u` and `node_v` in `g1` and `g2` resp.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        node_v : Any
            index of node v in g2
        g1 : networkx.Graph
            Graph containing u
        g2 : networkx.Graph
            Graph containing v

        Returns
        -------
        A positive float value
        """
        return 0 if self.compare_nodes(node_u, node_v, g1, g2) else self.cns_

    def cnd(self, u: Any, g1: nx.Graph) -> float:
        """Returns the deletion cost of `node_u` in `g1`.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        -------
        A positive float value
        """
        return self.cnd_

    def cni(self, v: Any, g2: nx.Graph):
        """Returns the insertion cost of `node_u` in `g1`.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        -------
        A positive float value
        """
        return self.cni_

    def ces(self, e1: Tuple[Any, Any], e2: Tuple[Any, Any], g1: nx.graph, g2: nx.Graph) -> float:
        """Returns the substitution cost between edge `e1` and edge `e2` in `g1` and `g2` resp.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge in g1
        e2 : Tuple[Any, Any]
            edge in g2
        g1 : networkx.Graph
            Graph containing e1
        g2 : networkx.Graph
            Graph containing e2

        Returns
        -------
        A positive float value
        """
        return 0 if self.compare_edges(e1, e2, g1, g2) else self.ces_

    def ced(self, e1: Tuple[Any, Any], g1: nx.Graph) -> float:
        """Returns the deletion cost of edge `e1` in `g1`.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge to delete in g1
        g1 : networkx.Graph
            Graph containing e1

        Returns
        -------
        A positive float value
        """
        return self.ced_

    def cei(self, e2: Tuple[Any, Any], g2: nx.Graph) -> float:
        """Returns the insertion cost of edge `e2` in `g2`.

        Parameters
        ----------
        e2 : Tuple[Any, Any]
            edge to insert in g2
        g2 : networkx.Graph
            Graph containing e2

        Returns
        -------
        A positive float value
        """
        return self.cei_


class RiesenCostFunction():
    """Cost function associated to the computation of a cost matrix between nodes for LSAP
    
    Features the very local graph structure by including possible cost on adjacents edges
    """

    def __init__(
            self,
            cf: CostFunction,
            lsap_solver: Callable[
                [np.ndarray],
                Tuple[np.ndarray, np.ndarray]
            ] = linear_sum_assignment
        ):
        """Creates a cost depending on the current node and its edges

        Parameters
        ----------
        cf : CostFunction
            the costs for operations
        lsap_solver : Callable(np.ndarray) -> Tuple[np.ndarray, np.ndarray]
            function to solve the LSAP problem. It will be given a cost matrix and must
            return the optimum nodes matching as 2 `numpy` arrays of rows and cols indices
        """
        self.cf_ = cf
        self.lsap_solver_ = lsap_solver

    def cns(self, u: Any, v: Any, g1: nx.Graph, g2: nx.Graph) -> float:
        """Returns the substitution cost between `node_u` and `node_v` in `g1` and `g2` resp.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        node_v : Any
            index of node v in g2
        g1 : networkx.Graph
            Graph containing u
        g2 : networkx.Graph
            Graph containing v

        Returns
        -------
        A positive float value
        """
        n = len(g1[u])
        m = len(g2[v])
        sub_C = np.ones([n+m, n+m]) * sys.maxsize
        sub_C[n:, m:] = 0
        i = 0
        l_nbr_u = g1[u]
        l_nbr_v = g2[v]
        for nbr_u in l_nbr_u:
            j = 0
            e1 = [u, nbr_u, g1[u][nbr_u]]
            for nbr_v in g2[v]:
                e2 = [v, nbr_v, g2[v][nbr_v]]
                sub_C[i, j] = self.cf_.ces(e1, e2, g1, g2)
                j += 1
            i += 1

        i = 0
        for nbr_u in l_nbr_u:
            sub_C[i, m+i] = self.cf_.ced([u, nbr_u], g1)
            i += 1

        j = 0
        for nbr_v in l_nbr_v:
            sub_C[n+j, j] = self.cf_.cei([v, nbr_v], g2)
            j += 1
        row_ind, col_ind = self.lsap_solver_(sub_C)
        cost = np.sum(sub_C[row_ind, col_ind])
        return self.cf_.cns(u, v, g1, g2) + cost

    def cnd(self, u: Any, g1: nx.Graph) -> float:
        """Returns the deletion cost of `node_u` in `g1`.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        -------
        A positive float value
        """
        cost = 0
        for nbr in g1[u]:
            cost += self.cf_.ced([u, nbr], g1)

        return self.cf_.cnd(u, g1) + cost

    def cni(self, v: Any, g2: nx.Graph):
        """Returns the insertion cost of `node_u` in `g1`.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        -------
        A positive float value
        """
        cost = 0
        for nbr in g2[v]:
            cost += self.cf_.cei([v, nbr], g2)
        return self.cf_.cni(v, g2) + cost

    def ces(
            self,
            e1: Tuple[Any, Any],
            e2: Tuple[Any, Any],
            g1: nx.Graph,
            g2: nx.Graph
        ) -> float:
        """Returns the substitution cost between edge `e1` and edge `e2` in `g1` and `g2` resp.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge in `g1`
        e2 : Tuple[Any, Any]
            edge in `g2`
        g1 : networkx.Graph
            Graph containing `e1`
        g2 : networkx.Graph
            Graph containing `e2`

        Returns
        -------
        a positive float value
        """
        return self.cf_.ces(e1, e2, g1, g2)

    def ced(self, e1: Tuple[Any, Any], g1: nx.graph) -> float:
        """Returns the deletion cost of edge `e1` in `g1`.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge to delete in `g1`
        g1 : networkx.Graph
            Graph containing `e1`

        Returns
        -------
        a positive float value
        """
        return self.cf_.ced(e1, g1)

    def cei(self, e2: Tuple[Any, Any], g2: nx.graph) -> float:
        """Returns the insertion cost of edge `e2 in `g2`.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge to insert in `g1`
        g1 : networkx.Graph
            Graph containing `e1`

        Returns
        -------
        a positive float value
        """
        return self.cf_.cei(e2, g2)

class NeighborhoodCostFunction():
    """Cost function associated to the computation of a cost matrix between nodes for LSAP
    
    Includes local structure with adjacent edges and neighbors nodes possibles edition costs
    """

    def __init__(
            self,
            cf: CostFunction,
            lsap_solver: Callable[
                [np.ndarray],
                Tuple[np.ndarray, np.ndarray]
            ] = linear_sum_assignment
        ):
        """Creates a cost depending on the current node, its edges and neighbors

        Parameters
        ----------
        cf : CostFunction
            the costs for operations
        lsap_solver : Callable(np.ndarray) -> Tuple[np.ndarray, np.ndarray]
            function to solve the LSAP problem. It will be given a cost matrix and must
            return the optimum nodes matching as 2 `numpy` arrays of rows and cols indices
        """
        self.cf_ = cf
        self.lsap_solver_ = lsap_solver

    def cns(self, u: Any, v: Any, g1: nx.Graph, g2: nx.Graph) -> float:
        """Returns the substitution cost between `node_u` and `node_v` in `g1` and `g2` resp.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        node_v : Any
            index of node v in g2
        g1 : networkx.Graph
            Graph containing u
        g2 : networkx.Graph
            Graph containing v

        Returns
        -------
        A positive float value
        """
        n = len(g1[u])
        m = len(g2[v])
        sub_C = np.ones([n+m, n+m]) * sys.maxsize
        sub_C[n:, m:] = 0
        i = 0
        l_nbr_u = g1[u]
        l_nbr_v = g2[v]
        for nbr_u in l_nbr_u:
            j = 0
            e1 = [u, nbr_u]
            for nbr_v in g2[v]:
                e2 = [v, nbr_v]
                sub_C[i, j] = self.cf_.ces(e1, e2, g1, g2)
                sub_C[i, j] += self.cf_.cns(nbr_u, nbr_v, g1, g2)
                j += 1
            i += 1

        i = 0
        for nbr_u in l_nbr_u:
            sub_C[i, m+i] = self.cf_.ced([u, nbr_u], g1)
            sub_C[i, m+i] += self.cf_.cnd(nbr_u, g1)
            i += 1

        j = 0
        for nbr_v in l_nbr_v:
            sub_C[n+j, j] = self.cf_.cei([v, nbr_v], g2)
            sub_C[n+j, j] += self.cf_.cni(nbr_v, g2)
            j += 1

        row_ind, col_ind = self.lsap_solver_(sub_C)
        cost = np.sum(sub_C[row_ind, col_ind])
        return self.cf_.cns(u, v, g1, g2) + cost

    def cnd(self, u: Any, g1: nx.Graph) -> float:
        """Returns the deletion cost of `node_u` in `g1`.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        -------
        A positive float value
        """
        cost = 0
        for nbr in g1[u]:
            cost += self.cf_.ced([u, nbr], g1)
        return self.cf_.cnd(u, g1) + cost

    def cni(self, v: Any, g2: nx.Graph) -> float:
        """Returns the insertion cost of `node_u` in `g1`.

        Parameters
        ----------
        node_u : Any
            index of node u in g1
        g1 : networkx.Graph
            Graph containing u

        Returns
        -------
        A positive float value
        """
        cost = 0
        for nbr in g2[v]:
            cost += self.cf_.cei([v, nbr, g2], g2)
        return self.cf_.cni(v, g2) + cost

    def ces(
            self,
            e1: Tuple[Any, Any],
            e2: Tuple[Any, Any],
            g1: nx.Graph,
            g2: nx.Graph
        ) -> float:
        """Returns the substitution cost between edge `e1` and edge `e2` in `g1` and `g2` resp.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge in `g1`
        e2 : Tuple[Any, Any]
            edge in `g2`
        g1 : networkx.Graph
            Graph containing `e1`
        g2 : networkx.Graph
            Graph containing `e2`

        Returns
        -------
        a positive float value
        """
        return self.cf_.ces(e1, e2, g1, g2)

    def ced(self, e1: Tuple[Any, Any], g1: nx.graph) -> float:
        """Returns the deletion cost of edge `e1` in `g1`.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge to delete in `g1`
        g1 : networkx.Graph
            Graph containing `e1`

        Returns
        -------
        a positive float value
        """
        return self.cf_.ced(e1, g1)

    def cei(self, e2: Tuple[Any, Any], g2: nx.graph) -> float:
        """Returns the insertion cost of edge `e2 in `g2`.

        Parameters
        ----------
        e1 : Tuple[Any, Any]
            edge to insert in `g1`
        g1 : networkx.Graph
            Graph containing `e1`

        Returns
        -------
        a positive float value
        """
        return self.cf_.cei(e2, g2)
