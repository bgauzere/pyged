# PyGED

![Static Badge](https://img.shields.io/badge/PyGED-2.0-green?style=plastic)
![Static Badge](https://img.shields.io/badge/Python-3.10|3.11|3.12|3.13+-yellow?style=plastic&logo=python&logoColor=3776AB&link=https%3A%2F%2Fwww.python.org%2F)
![Static Badge](https://img.shields.io/badge/NetworkX-3.0+-blue?style=plastic&logo=networkx&logoColor=blue&link=https://networkx.org/)
![Static Badge](https://img.shields.io/badge/Numpy-2.0+-blue?style=plastic&logo=numpy&logoColor=013243&link=https://numpy.org/)
![Static Badge](https://img.shields.io/badge/Scipy-1.4+-blue?style=plastic&logo=scipy&logoColor=8CAA6&link=https://scipy.org/)

`PyGED` is a python package allowing to compute an upper bound
of the **Graph Edit Distance** (*GED*).

This approximation algorithm is based on
*Structural Pattern Recognition with Graph Edit Distance* [1]_,
where we consider the graphs as a bipartite graph.

It uses a node matching between the graphs to
approximate the *GED*, by solving a Linear Sum
Assignment Problem instead (*LSAP*) of a
Quadratic Assignment Problem.

A *LSAP* can be solved by using a cost matrix
containing the assignment cost of each pair of
nodes, with which we want to minimize the
final assignment cost [2]_.

## Dependencies

`PyGED` uses the Graph class from `NetworkX` (that support python from 3.10+)

Unless another algorithm is used to compute the node matching, this algorithm will use `scipy.optimize.linear_sum_assignment`

`Numpy` arrays are used for the cost matrix

Eventually, `pytest` is required to run the unit tests (as well as `numpy`)

Dependencies can be installed using `pip`

> pip install networkx numpy scipy

or

> pip install -r requirements.txt

or by using the `pipenv` to setup an environment

> pipenv install

## Content

This implementation offers 3 differents ways to parametrize the approximation of the *GED*, which are 3 cost functions valuating edit operations :

* `ConstantCostFunction`
* `RiesenCostFunction`
* `NeighborhoodCostFunction`

These edit cost functions are used to build the cost matrix, in order to solve the *LSAP*, which will minimize the assignment cost (ie in our case, the edit cost)

But to create a better matching, more complex cost functions can be used, that will be able to retrieve local structural data, hence improving the node matching thanks to comparisons between neighborhoods informations.

## Small examples

The algorithm can be called directly on 2 graphs. It will then use the default edit costs for the edit operations (ie a cost of 1 for any substitution, and a cost of 3 for insertion and deletion).

```python
>>> import networkx as nx
>>> import pyged
>>> g1 = nx.complete_graph(5)
>>> g2 = nx.complete_graph(6)
>>> ged = pyged.ged.GED()
>>> ged.ged(g1, g2)[0]
18
```

The results will vary according to the given cost function

```python
>>> ged = pyged.ged.GED(pyged.costfunctions.ConstantCostFunction(
...     1,
...     1,
...     1,
...     1
... ))
>>> ged.ged(g1, g2)[0]
6
```

In this case, all costs of 1 means we count the number of edit operations performed.

When it comes to labeled graphs, we might want to use more suitable cost functions :

```python
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
```

In order to perform the matching, we want the algorithm to be able to compare the labels

```python
# u is a node of g1, and v, a node of g2
def compare_nodes(u, v, g1, g2):
    return g1.nodes[u]["Label"] == g2.nodes[v]["Label"]

# e1 is an edge (tuple of nodes) of g1, and e2, an edge of g2
def compare_edges(e1, e2, g1, g2):
    return g1[e1[0]][e1[1]]["edge_attr"] == g2[e2[0]][e2[1]]["edge_attr"]
```

We define functions to tell the algorithm whether two nodes/edges can be considered the same. Hence, improving the matching and reducing the cost of substitution from `u` to `u`.

```python
>>> cost_function = pyged.costfunctions.ConstantCostFunction(
...     1,
...     1,
...     1,
...     1,
...     compare_nodes,
...     compare_edges
... )
```

We can then use this cost function as a construction basis for other functions

```python
>>> riesen_cost_function = pyged.costfunctions.RiesenCostFunction(cost_function)
>>> ged = pyged.ged.GED(riesen_cost_function)
>>> ged.ged(g1, g2)[0]
5
```

## References

[1] K. Riesen, Structural Pattern Recognition with
    Graph Edit Distance, Switzerland, Springer, 2015

[2] Linear Sum Assignment Problem, [Wikipedia](https://en.wikipedia.org/wiki/Assignment_problem)

## See also

* [scipy.optimize.linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
* pyged.costfunctions
* pyged.solvers
* [networkx.graph_edit_distance](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.graph_edit_distance.html)
