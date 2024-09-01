import random
import math
from collections import deque

class Problem:
    def __init__(self, graph, initial, goal):
        self.graph = graph
        self.initial = initial
        self.goal = goal
        self.locations = {}

    def actions(self, state):
        return list(self.graph[state].keys())

    def result(self, state, action):
        return action

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, cost_so_far, state1, action, state2):
        return cost_so_far + self.graph[state1][state2]

    def straight_line_distance(self, city1, city2):
        x1, y1 = self.locations[city1]
        x2, y2 = self.locations[city2]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def best_first_search(self, display=False):
        node = self.Node(self.initial)
        frontier = deque([node])
        explored = set()
        while frontier:
            node = frontier.popleft()
            if self.goal_test(node.state):
                if display:
                    print(f"Best-First Search: Goal reached. Path cost: {node.path_cost}")
                return node
            explored.add(node.state)
            for child in node.expand(self):
                if child.state not in explored and child not in frontier:
                    frontier.append(child)
        if display:
            print("Best-First Search: Goal not reached.")
        return None

    def astar_search(self, display=False):
        node = self.Node(self.initial)
        frontier = deque([node])
        explored = set()
        while frontier:
            node = frontier.popleft()
            if self.goal_test(node.state):
                if display:
                    print(f"A* Search: Goal reached. Path cost: {node.path_cost}")
                return node
            explored.add(node.state)
            for child in node.expand(self):
                if child.state not in explored and child not in frontier:
                    frontier.append(child)
                elif child in frontier:
                    if child.path_cost < frontier[frontier.index(child)].path_cost:
                        frontier.remove(child)
                        frontier.append(child)
        if display:
            print("A* Search: Goal not reached.")
        return None

    def hill_climbing_search(self, display=False):
        current_node = self.Node(self.initial)
        while True:
            neighbors = current_node.expand(self)
            if not neighbors:
                if display:
                    print("Hill Climbing Search: Local maximum reached.")
                return current_node
            neighbor = min(neighbors, key=lambda node: self.heuristic(node.state))
            if self.heuristic(neighbor.state) >= self.heuristic(current_node.state):
                if display:
                    print("Hill Climbing Search: Local maximum reached.")
                return current_node
            current_node = neighbor

    def simulated_annealing(self, display=False):
        current_node = self.Node(self.initial)
        temperature = 1000
        cooling_rate = 0.95
        while temperature > 0.1:
            neighbor = random.choice(current_node.expand(self))
            delta_e = self.heuristic(neighbor.state) - self.heuristic(current_node.state)
            if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
                current_node = neighbor
            temperature *= cooling_rate
        if display:
            print(f"Simulated Annealing: Temperature dropped to {temperature:.2f}")
        return current_node

    class Node:
        def __init__(self, state, parent=None, action=None, path_cost=0):
            self.state = state
            self.parent = parent
            self.action = action
            self.path_cost = path_cost
            self.depth = 0
            if parent:
                self.depth = parent.depth + 1

        def expand(self, problem):
            return [self.child_node(problem, action)
                    for action in problem.actions(self.state)]

        def child_node(self, problem, action):
            next_state = problem.result(self.state, action)
            return self.__class__(next_state, self, action,
                                   problem.path_cost(self.path_cost, self.state, action, next_state))

# Define the undirected graph representing the Romania map
romania_map = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Bucharest': {'Urziceni': 85, 'Pitesti': 101, 'Giurgiu': 90, 'Fagaras': 211},
    'Craiova': {'Drobeta': 120, 'Rimnicu': 146, 'Pitesti': 138},
    'Drobeta': {'Mehadia': 75},
    'Eforie': {'Hirsova': 86},
    'Fagaras': {'Sibiu': 99},
    'Hirsova': {'Urziceni': 98},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Pitesti': {'Rimnicu': 97},
    'Rimnicu': {'Sibiu': 80},
    'Urziceni': {'Vaslui': 142}
}

# Define the locations of cities
romania_map_locations = {
    'Arad': (91, 492), 'Bucharest': (400, 327), 'Craiova': (253, 288),
    'Drobeta': (165, 299), 'Eforie': (562, 293), 'Fagaras': (305, 449),
    'Giurgiu': (375,270), 'Hirsova':(534, 350), 'Iasi':(473, 506),
    'Lugoj':(165, 379), 'Mehadia':(168, 339), 'Neamt':(406, 537),
    'Oradea':(131, 571), 'Pitesti':(320, 368), 'Rimnicu':(233, 410),
    'Sibiu':(207, 457), 'Timisoara':(94, 410), 'Urziceni':(456, 350),
    'Vaslui':(509, 444), 'Zerind':(108, 531)}
