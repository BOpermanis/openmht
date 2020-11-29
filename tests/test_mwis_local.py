from simpleai.search.local import hill_climbing
from comparison import generate_random_graph, check_set_independent
from local_mwis import MWISLocal


for seed in range(100):
    g = generate_random_graph()
    mwis_local = MWISLocal()
    mwis_local.initialize_graph(g)

    for _ in range(100):
        state = mwis_local.generate_random_state()
        assert check_set_independent(state, g)