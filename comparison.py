import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import networkx as nx
from openmht.weighted_graph import WeightedGraph
from openmht.kalman_filter import KalmanFilter
from local_mwis import MWISLocal, genetic
from time import time

def exact_solution(g):
    gh_graph = WeightedGraph()
    for n in g.nodes():
        gh_graph.add_weighted_vertex(str(n), g.nodes[n]['weight'])
    gh_graph.set_edges(g.edges())
    mwis_ids, max_val = gh_graph.mwis(flag_return_max_val=True)
    return mwis_ids, max_val


def generate_random_graph(seed=0, num_max_nodes=20):
    np.random.seed(seed)
    coef_connection = 0.99 ** num_max_nodes
    N = np.random.randint(3, num_max_nodes)
    weights = np.random.uniform(0, 1, (N,))
    G = nx.Graph()
    # for i in range(N):
    #     G.add_node(i, attr_dict={'weight': weights[i]})
    for i in range(N):
        G.add_node(i, weight=weights[i])

    for i in range(N):
        for j in range(N):
            if j > i and np.random.uniform() < coef_connection:
                G.add_edge(i, j)

    return G


def check_set_independent(s, g):
    for a in s:
        for b in s:
            if (a, b) in g.edges():
                return False
    return True


def get_time(fun):
    s = time()
    out = fun()
    return out, time() - s


def simplify_graph(wg, flag_print_size=False):
    nodes = []
    for n in wg.nodes():
        nodes.append((n, g.nodes[n]['weight']))
    edges = wg.edges()
    if flag_print_size:
        print("len(nodes), len(edges)", len(nodes), len(edges))
    return nodes, edges


if __name__ == "__main__":

    g = generate_random_graph(num_max_nodes=100, seed=4)
    mwis_local = MWISLocal()

    # print(state)
    # print(mwis_local.value(state))
    # print(mwis_local.heuristic(state))
    # print(mwis_local.actions(state))
    # print(mwis_local.result(state, 4))
    # exit()
    # result = hill_climbing(mwis_local, iterations_limit=20)
    ts = []
    simplify_graph(g, flag_print_size=True)
    for _ in range(100):
        result, t = get_time(lambda: mwis_local.run(*simplify_graph(g)))
        print(_, result.state, mwis_local.problem.value(result.state))
        ts.append(t)
    t1 = np.average(ts)

    exit()

    print(result.state, mwis_local.problem.value(result.state))
    (exact_state, exact_value), t2 = get_time(lambda: exact_solution(g))

    print(t1, t2)
    print(result.state, mwis_local.problem.value(result.state))
    print(check_set_independent(result.state, g))
    # nx.draw(g)
    # plt.show()