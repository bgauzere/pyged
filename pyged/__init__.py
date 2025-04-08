"""
PyGED
=====

Python package allowing to compute an upper bound
of the **Graph Edit Distance** (*GED*).

This bipartite *GED* approximation algorithm is based on
*Structural Pattern Recognition with Graph Edit Distance* [1]_.

It uses a node matching between the graphs to
approximate the *GED*, by solving a Linear Sum
Assignment Problem (*LSAP*) instead of a
Quadratic Assignment Problem.

A *LSAP* can be solved by using a cost matrix
containing the assignment cost of each pair of
nodes, with which we want to minimize the
final assigment cost [2]_.

This implementation offers some cost functions to build
the cost matrix, as well as multiple *LSAP* solvers

Examples
--------
>>> import networkx as nx
>>> import pyged
>>> g1, g2 = nx.Graph(), nx.Graph()
>>> g1.add_nodes_from([
...     ("u1", {"Label": 1}),
...     ("u2", {"Label": 2}),
...     ("u3", {"Label": 1}),
...     ("u4", {"Label": 3})
... ])
>>> g1.add_edges_from([
...     ("u1", "u2", {"edge_attr": 1}),
...     ("u2", "u3", {"edge_attr": 2}),
...     ("u2", "u4", {"edge_attr": 1}),
...     ("u3", "u4", {"edge_attr": 1})
... ])
>>> g2.add_nodes_from([
...     ("v1", {"Label": 3}),
...     ("v2", {"Label": 2}),
...     ("v3", {"Label": 3})
... ])
>>> g2.add_edges_from([
...     ("v1", "v2", {"edge_attr": 1}),
...     ("v2", "v3", {"edge_attr": 1})
... ])
>>> ged = pyged.ged.GED(pyged.costfunctions.ConstantCostFunction(
...     1,
...     1,
...     1,
...     1,
...     lambda u, v, g1, g2: g1.nodes[u]["Label"] == g2.nodes[v]["Label"],
...     lambda e1, e2, g1, g2: g1[e1[0]][e1[1]]["edge_attr"] == g2[e2[0]][e2[1]]["edge_attr"]
... ))
>>> ged.ged(g1, g2)[0]
4

References
----------
.. [1] K. Riesen, Structural Pattern Recognition with
   Graph Edit Distance, Switzerland, Springer, 2015

.. [2] Linear Sum Assignment Problem, Wikipedia,
   https://en.wikipedia.org/wiki/Assignment_problem


See Also
--------
scipy.optimize.linear_sum_assignment
pyged.costfunctions
pyged.solvers
networkx.graph_edit_distance
"""

from pyged.costfunctions import *
from pyged.solvers import *
from pyged.bpged_utils import *
from pyged.bipartite_ged import *
