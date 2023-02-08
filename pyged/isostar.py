

def compute_star(G, node_index):
    return 1


if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt
    from GED import ged
    G1 = nx.read_gml(
        "/home/bgauzere/work/Recherche/Datasets/Acyclic/gml/dimethyl_ether.ct.gml")
    G2 = nx.read_gml(
        "/home/bgauzere/work/Recherche/Datasets/Acyclic/gml/dimethyl_sulfide.ct.gml")

    # nx.draw_networkx(G1, with_labels=True)
    # plt.show()
    # nx.draw_networkx(G2, with_labels=True)
    # plt.show()
    print(ged(G1, G2))
    # print(ged(G1, G1))
    print(G1.nodes['0'].keys())
