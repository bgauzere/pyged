"""
Function making tests code simple
"""

from typing import Tuple, Any
import networkx as nx

def load_test_graphs() -> Tuple[nx.Graph, nx.Graph]:
    """Creates 2 graphs for the tests"""
    g1, g2 = nx.Graph(), nx.Graph()

    g1.add_nodes_from([
        ("u1", {"Label": 1}),
        ("u2", {"Label": 2}),
        ("u3", {"Label": 1}),
        ("u4", {"Label": 3})
    ])
    g1.add_edges_from([
        ("u1", "u2", {"edge_attr": 1}),
        ("u2", "u3", {"edge_attr": 2}),
        ("u2", "u4", {"edge_attr": 1}),
        ("u3", "u4", {"edge_attr": 1})
    ])

    g2.add_nodes_from([
        ("v1", {"Label": 3}),
        ("v2", {"Label": 2}),
        ("v3", {"Label": 3})
    ])
    g2.add_edges_from([
        ("v1", "v2", {"edge_attr": 1}),
        ("v2", "v3", {"edge_attr": 1})
    ])

    return g1, g2

def comp_nodes(u: Any, v: Any, g1: nx.Graph, g2: nx.Graph) -> bool:
    """`True` if nodes `u` from `g1` & `v` from `g2` are identical"""
    return g1.nodes[u]["Label"] == g2.nodes[v]["Label"]

def comp_edges(e1: Tuple[Any, Any], e2: Tuple[Any, Any], g1: nx.Graph, g2: nx.Graph) -> bool:
    """`True` if edges `e1` from `g1` & `e2` from `g2` are identical"""
    return g1[e1[0]][e1[1]]["edge_attr"] == g2[e2[0]][e2[1]]["edge_attr"]
