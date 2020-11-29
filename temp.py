from simpleai.search.local import hill_climbing
from comparison import generate_random_graph, check_set_independent
from local_mwis import MWISLocal

g = generate_random_graph(num_max_nodes=30)
mwis_local = MWISLocal()
mwis_local.initialize_graph(g)
s1 = mwis_local.generate_random_state()
s2 = mwis_local.generate_random_state()
print(s1)
print(s2)
print(mwis_local.crossover(s1, s2))
exit()

for seed in range(100):
    print(111111, seed)
    g = generate_random_graph()
    mwis_local = MWISLocal()
    mwis_local.initialize_graph(g)

    print(mwis_local.initial_state)
    print(mwis_local.mutate(mwis_local.initial_state))

    for _ in range(100):
        print(1111, _)
        state = mwis_local.generate_random_state()
        print(state)
        exit()
        assert check_set_independent(state, g)