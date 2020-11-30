import numpy as np
import random
from simpleai.search import SearchProblem, astar
from simpleai.search.local import hill_climbing
from collections import defaultdict
from itertools import chain
from pprint import pprint

from simpleai.search.utils import BoundedPriorityQueue, InverseTransformSampler
from simpleai.search.models import SearchNodeValueOrdered
from simpleai.search.local import _create_genetic_expander

"""
simpleai versions used:
simpleai                           0.8.2
simplegeneric                      0.8.1
"""

import math
import random

def check_set_independent(s, g):
    for a in s:
        for b in s:
            if (a, b) in g.edges():
                return False
    return True

max_mutate = 2
max_crossover = 0.5

class MWISLocalProblem(SearchProblem):

    def initialize_graph(self, nodes, edges):
        self.node2nodes = defaultdict(set)
        self.weights = {id:w for id, w in nodes}
        self.nodes = set(self.weights.keys())
        for i, j in edges:
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


def _local_search(problem, fringe_expander, iterations_limit=0, fringe_size=1,
                  random_initial_states=False, stop_when_no_better=True,
                  viewer=None):
    '''
    Basic algorithm for all local search algorithms.
    '''
    if viewer:
        viewer.event('started')

    fringe = BoundedPriorityQueue(fringe_size)

    if problem.initial_states is None:
        random_initial_states = True

    if random_initial_states:
        for _ in range(fringe_size):
            s = problem.generate_random_state()
            fringe.append(SearchNodeValueOrdered(state=s, problem=problem))
    else:
        for initial_state in problem.initial_states:
            if not isinstance(initial_state, SearchNodeValueOrdered):
                initial_state = SearchNodeValueOrdered(state=initial_state, problem=problem)
            fringe.append(initial_state)

    finish_reason = ''
    iteration = 0
    run = True
    best = None

    while run:
        if viewer:
            viewer.event('new_iteration', list(fringe))

        old_best = fringe[0]
        fringe_expander(fringe, iteration, viewer)
        best = fringe[0]

        iteration += 1

        if iterations_limit and iteration >= iterations_limit:
            run = False
            finish_reason = 'reaching iteration limit'
        elif old_best.value >= best.value and stop_when_no_better:
            run = False
            finish_reason = 'not being able to improve solution'

    if viewer:
        viewer.event('finished', fringe, best, 'returned after %s' % finish_reason)

    return best, fringe


def genetic(problem, population_size=100, mutation_chance=0.1,
            iterations_limit=0, viewer=None):
    '''
    Genetic search.

    population_size specifies the size of the population (ORLY).
    mutation_chance specifies the probability of a mutation on a child,
    varying from 0 to 1.
    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.generate_random_state, SearchProblem.crossover,
    SearchProblem.mutate and SearchProblem.value.
    '''
    return _local_search(problem,
                         _create_genetic_expander(problem, mutation_chance),
                         iterations_limit=iterations_limit,
                         fringe_size=population_size,
                         random_initial_states=False,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer)


class MWISLocal:
    def __init__(self, iterations_limit=20, population_size=4, mutation_chance=0.1):
        self.problem = MWISLocalProblem()
        self.iterations_limit = iterations_limit
        self.population_size = population_size
        self.mutation_chance = mutation_chance
        self.initial_states = None

    def run(self, nodes, edges):
        self.problem.initialize_graph(nodes, edges)
        self.problem.initial_states = self.initial_states
        result, population = genetic(self.problem,
                                     iterations_limit=self.iterations_limit,
                                     population_size=self.population_size,
                                     mutation_chance=self.mutation_chance)
        self.initial_states = population
        return result

