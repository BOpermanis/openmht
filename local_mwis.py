import numpy as np
import random
from simpleai.search import SearchProblem, astar
from simpleai.search.local import hill_climbing
from collections import defaultdict
from itertools import chain

def check_set_independent(s, g):
    for a in s:
        for b in s:
            if (a, b) in g.edges():
                return False
    return True

max_mutate = 2
max_crossover = 0.5

class MWISLocal(SearchProblem):

    def initialize_graph(self, wg):
        self.wg = wg
        self.wg_edges = {*wg.edges()} | {*[(b, a) for a, b in wg.edges()]}
        self.node2nodes = defaultdict(set)
        self.nodes = {*wg.nodes()}
        self.weights = {n: wg.node[n]['weight'] for n in self.nodes}
        for i, j in wg.edges():
            self.node2nodes[i].add(j)
            self.node2nodes[j].add(i)
        self.initial_state = self.generate_random_state()

    def actions(self, state):
        s_not = set()
        for n in state:
            s_not.update(self.node2nodes[n])
        return self.nodes - s_not

    def result(self, state, action):
        return state.union({action})

    def value(self, state):
        return sum([self.weights[n] for n in state])

    def generate_random_state(self):
        s_independent = set()
        s_compliment = {*self.nodes}

        while len(s_compliment) > 0:
            n0 = random.choice([*s_compliment])
            s_independent.add(n0)
            s_compliment.difference_update(self.node2nodes[n0])
            s_compliment.remove(n0)

        return s_independent

    def heuristic(self, state):
        return -sum([len(self.node2nodes[n]) for n in state])

    def _add_whats_left(self, state):
        # adds so that there is not compliment left (should make optmization more effective if all weights are positive)
        possible_actions = self.actions(state)
        while len(possible_actions) > 0:
            n0 = random.choice([*possible_actions])
            state.add(n0)
            possible_actions.difference_update(self.node2nodes[n0])
            possible_actions.remove(n0)
        return state

    def mutate(self, state):
        # cross both strings, at a random point
        mutated = state.copy()
        mutated_list = list(state)
        random.shuffle(mutated_list)

        num_mutated = 0
        for n0 in mutated_list:
            mutated.remove(n0)
            potential = list(self.node2nodes[n0])
            random.shuffle(potential)
            for n1 in potential:
                if mutated.isdisjoint(self.node2nodes[n1]):
                    mutated.add(n1)
                    num_mutated += 1
                    break

            if num_mutated == max_mutate:
                break

        # adds so that there is not compliment left (should make optmization more effective if all weights are positive)
        mutated = self._add_whats_left(mutated)

        return mutated

    def crossover(self, state1, state2):

        soup = list(state1.union(state2))
        random.shuffle(soup)

        s_independent = set()
        s_compliment = {*self.nodes}

        for n0 in soup:
            if not s_independent.isdisjoint(self.node2nodes[n0]):
                continue
            s_independent.add(n0)
            s_compliment.difference_update(self.node2nodes[n0])
            s_compliment.remove(n0)

        s_independent = self._add_whats_left(s_independent)

        return s_independent