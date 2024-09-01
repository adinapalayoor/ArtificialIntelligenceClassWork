# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:47:10 2023

@author: adina_l1uzsjt
"""
import math
import random


class Graph:
    #initialize graph object
    def __init__(self, graph_dict=None, locations=None):
        if graph_dict is None:
            graph_dict = {}
        if locations is None:
            locations = {}
        self.graph_dict = graph_dict
        self.locations = locations 


class Node:
    #initialize a state, patent, action, path cost and heuristic in Node object
    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.path_cost + self.heuristic) < (other.path_cost + other.heuristic)


class Problem:
    #Problem class defines the initial state, goal state and graph
    def __init__(self, initial, goal, graph):
        self.initial = initial
        self.goal = goal
        self.graph = graph

    def actions(self, state):
        return list(self.graph.graph_dict[state].keys())

    def result(self, state, action):
        return action

    def step_cost(self, state1, action, state2):
        return self.graph.graph_dict[state1][state2]

    def heuristic(self, state):
        x1, y1 = self.graph.locations[state]
        x2, y2 = self.graph.locations[self.goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

#the following are the searching algorithms to be used by the Problem definition
def greedy_best_first_search(problem):
    initial_node = Node(problem.initial)
    if initial_node.state == problem.goal:
        return initial_node

    frontier = [initial_node]
    explored = set()

    while frontier:
        frontier.sort()
        node = frontier.pop(0)
        explored.add(node.state)

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state not in explored and child_state not in [n.state for n in frontier]:
                child_node = Node(child_state, node, action, node.path_cost + problem.step_cost(node.state, action, child_state),
                                  problem.heuristic(child_state))
                if child_node.state == problem.goal:
                    return child_node
                frontier.append(child_node)

    return None


def a_star_search(problem):
    initial_node = Node(problem.initial)
    if initial_node.state == problem.goal:
        return initial_node

    frontier = [initial_node]
    explored = set()

    while frontier:
        frontier.sort()
        node = frontier.pop(0)
        explored.add(node.state)

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state not in explored and child_state not in [n.state for n in frontier]:
                child_node = Node(child_state, node, action, node.path_cost + problem.step_cost(node.state, action, child_state),
                                  problem.heuristic(child_state))
                if child_node.state == problem.goal:
                    return child_node
                frontier.append(child_node)

    return None


def hill_climbing(problem):
    current_node = Node(problem.initial)
    if current_node.state == problem.goal:
        return current_node

    while True:
        neighbors = [Node(s, current_node, a, current_node.path_cost + problem.step_cost(current_node.state, a, s), problem.heuristic(s))
                     for a, s in problem.graph.graph_dict[current_node.state].items()]

        if not neighbors:
            return None

        neighbors.sort()
        next_node = neighbors[0]

        if next_node.state == problem.goal:
            return next_node
        elif next_node.path_cost + next_node.heuristic >= current_node.path_cost + current_node.heuristic:
            return current_node

        current_node = next_node


def simulated_annealing(problem, T=1.0, T_min=0.01, alpha=0.9):
    current_node = Node(problem.initial)
    if current_node.state == problem.goal:
        return current_node

    while T > T_min:
        neighbors = [Node(s, current_node, a, current_node.path_cost + problem.step_cost(current_node.state, a, s), problem.heuristic(s))
                     for a, s in problem.graph.graph_dict[current_node.state].items()]

        if not neighbors:
            return None

        next_node = random.choice(neighbors)
        delta_E = next_node.path_cost - current_node.path_cost

        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            current_node = next_node

        T *= alpha

    return None


def print_solution(node):
    if node is None:
        print("No solution found.")
    else:
        path = []
        while node:
            path.insert(0, node.state)
            node = node.parent
        print("Path:", " -> ".join(path))
        print("Cost:", path_cost(path))


def path_cost(path):
    cost = 0
    for i in range(len(path) - 1):
        cost += romania_map.graph_dict[path[i]][path[i + 1]]
    return cost


if __name__ == "__main__":
    romania_map = Graph(dict(
        Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
        Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
        Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
        Drobeta=dict(Mehadia=75),
        Eforie=dict(Hirsova=86),
        Fagaras=dict(Sibiu=99),
        Hirsova=dict(Urziceni=98),
        Iasi=dict(Vaslui=92, Neamt=87),
        Lugoj=dict(Timisoara=111, Mehadia=70),
        Oradea=dict(Zerind=71, Sibiu=151),
        Pitesti=dict(Rimnicu=97),
        Rimnicu=dict(Sibiu=80),
        Urziceni=dict(Vaslui=142)))

    romania_map.locations = dict(
        Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
        Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
        Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
        Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
        Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
        Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
        Vaslui=(509, 444), Zerind=(108,531))

