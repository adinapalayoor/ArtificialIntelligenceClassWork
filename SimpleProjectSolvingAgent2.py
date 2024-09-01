# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:47:10 2023

@author: adina_l1uzsjt
"""

import math
import random
from searchnew import *

class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, state, goal, graph):
        super().__init__(state, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf
class SimpleProblemSolvingAgent:
    
    def greedy_best_first_search(self, goal):
        problem = GraphProblem(self.state, goal, self.graph)
        result = greedy_best_first_graph_search(problem, lambda node: euc_distance(self.state, goal, self.graph))
        if result:
            solution = [self.state] + result.solution()
            path_costs = calculate_path_cost(solution, self.graph)
            return solution, path_costs
        else:
            return None
        
    def astar_search(self,goal):
        problem = GraphProblem(self.state, goal, self.graph)
        result = (problem, problem.h)
        if result:
            solution = [self.state] + result.solution()
            path_costs = calculate_path_cost(solution, self.graph)
            return solution, path_costs
        else:
            return None
        

    def hill_climbing(self, state):
            init, g = state[0], state[-1]
            current_path = state
            for i in range(10000):
                init = random.choice(current_path)
                index = current_path.index(init)
                first_half = current_path[:index]
                neighbor = first_half + random_path_between_cities(init, g, self.graph)
                if calculate_path_cost(neighbor, self.graph) <= calculate_path_cost(current_path, self.graph):
                    current_path = neighbor
            return current_path, calculate_path_cost(current_path, self.graph)

    def simulated_annealing(self, state, schedule=exp_schedule()):
        init, g = state[0], state[-1]
        current_path = state
        for t in range(sys.maxsize):
            T = schedule(t)
            if T == 0:
                return current_path
            init = random.choice(current_path)
            index = current_path.index(init)
            first_half = current_path[:index]
            neighbor = first_half + random_path_between_cities(init, g, self.graph)
            delta_e = calculate_path_cost(current_path, self.graph) - calculate_path_cost(neighbor, self.graph)
            if delta_e > 0 or probability(np.exp(delta_e / T)):
                current_path = neighbor
        return current_path
    
    def random_path_between_cities(initial_city, goal_city, graph):
        current_city = initial_city
        path = [current_city]
    
        while current_city != goal_city:
            neighbors = list(graph.get(current_city).keys())
            if not neighbors:
                break
    
            next_city = random.choice(neighbors)
            path.append(next_city)
            current_city = next_city
    
        return path

    def euc_distance(initial, goal, graph):
        initial_location = graph.locations[initial]
        goal_location = graph.locations[goal]
        euc_dist = ((initial_location[0] - goal_location[0])**2 + (initial_location[1] - goal_location[1])**2) ** 0.5
        return euc_dist
    
    def calculate_path_cost(node_sequence, graph):
        cost = 0
        for i in range(len(node_sequence) - 1):
            cost += graph.get(node_sequence[i], node_sequence[i + 1])
        return cost
    
# def print_solution(node):
#     if node is None:
#         print("No solution found.")
#     else:
#         path = []
#         while node:
#             path.insert(0, node.state)
#             node = node.parent
#         print("Path:", " -> ".join(path))
#         print("Cost:", path_cost(path))


# def path_cost(path):
#     cost = 0
#     for i in range(len(path) - 1):
#         cost += romania_map.graph_dict[path[i]][path[i + 1]]
#     return cost  
