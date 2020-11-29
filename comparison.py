import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import networkx as nx
from openmht.weighted_graph import WeightedGraph
from openmht.kalman_filter import KalmanFilter
from local_mwis import MWISLocal
from time import time

def exact_solution(g):
    gh_graph = WeightedGraph()
    for n in g.nodes():
        gh_graph.add_weighted_vertex(str(n), g.node[n]['weight'])
    gh_graph.set_edges(g.edges())
    mwis_ids, max_val = gh_graph.mwis(flag_return_max_val=True)
    return mwis_ids, max_val


def generate_random_graph(seed=0, num_max_nodes=20):
    np.random.seed(seed)
    coef_connection = 1 - 0.9 ** num_max_nodes
    N = np.random.randint(3, num_max_nodes)
    weights = np.random.uniform(0, 1, (N,))
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, attr_dict={'weight': weights[i]})

    for i in range(N):
        for j in range(N):
            if j > i and np.random.uniform() > coef_connection:
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


if __name__ == "__main__":
    from simpleai.search.local import hill_climbing, genetic

    g = generate_random_graph(num_max_nodes=45)
    mwis_local = MWISLocal()
    mwis_local.initialize_graph(g)

    initial_state = mwis_local.generate_random_state()
    mwis_local.initial_state = initial_state
    # print(state)
    # print(mwis_local.value(state))
    # print(mwis_local.heuristic(state))
    # print(mwis_local.actions(state))
    # print(mwis_local.result(state, 4))
    # exit()
    # result = hill_climbing(mwis_local, iterations_limit=20)
    result, t1 = get_time(lambda: genetic(mwis_local, iterations_limit=20, population_size=4, mutation_chance=0.1))

    print(result.state, mwis_local.value(result.state))
    (exact_state, exact_value), t2 = get_time(lambda: exact_solution(g))

    print(t1, t2)
    print(result.state, mwis_local.value(result.state))
    print(check_set_independent(result.state, g))
    # nx.draw(g)
    # plt.show()