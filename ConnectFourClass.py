# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:09:53 2023

@author: adina_l1uzsjt
"""


import random
import copy
import math

class ConnectFourGame:
    def __init__(self, player1, player2):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.players = [player1, player2]

    def print_board(self):
        for row in self.board:
            print("|".join(row))
        print("---------------")

    def make_move(self, player, column):
        for row in range(5, -1, -1):
            if self.board[row][column] == ' ':
                self.board[row][column] = player
                return True
        return False

    def is_winner(self, player):
    
        for row in range(6):
            for col in range(4):
                if all(self.board[row][col + i] == player for i in range(4)):
                    return True

        for row in range(3):
            for col in range(7):
                if all(self.board[row + i][col] == player for i in range(4)):
                    return True

        
        for row in range(3):
            for col in range(4):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    return True

       
        for row in range(3):
            for col in range(3, 7):
                if all(self.board[row + i][col - i] == player for i in range(4)):
                    return True

        return False

    def is_full(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def play(self):
        turn = 0
        while True:
            player = self.players[turn % 2]
            column = player.get_move(self.board)
            
            if not self.make_move(player.symbol, column):
                print("Invalid move. Column is full. Try again.")
                continue

            self.print_board()

            if self.is_winner(player.symbol):
                print(f"Player {player.symbol} wins!")
                break
            elif self.is_full():
                print("The game is a tie!")
                break

            turn += 1

class Player:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_move(self, board):
        pass

class MCT_Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.N = 0
        self.U = 0

class MonteCarloTreeSearch:
    def select_move(self, game):
        player = game.get_current_player()
        state = game

        def ucb(child_node):
            if child_node.N == 0:
                return float('inf')
            exploitation = child_node.U / child_node.N
            exploration = (2 * (2 * math.log(child_node.parent.N) / child_node.N) ** 0.5) if child_node.N > 0 else 0
            return exploitation + exploration

        def select(node):
            while node.children:
                node = max(node.children.keys(), key=ucb)
            return node

        def expand(node):
            if not node.children and not game.is_game_over():
                actions = game.get_available_moves()
                node.children = {
                    MCT_Node(state=game.result_from_move(action), parent=node): action
                    for action in actions
                }
            return select(node)

        def simulate(game, state):
            while not game.is_game_over():
                action = random.choice(game.get_available_moves())
                state = game.result_from_move(action)
            utility = game.calculate_utility(player)
            return -utility

        def backprop(node, utility):
            while node:
                if utility > 0:
                    node.U += utility
                node.N += 1
                node = node.parent

        root = MCT_Node(state=state)

        for _ in range(1000):
            leaf = select(root)
            child = expand(leaf)
            result = simulate(game, child.state)
            backprop(child, result)

        max_state = max(root.children, key=lambda p: p.N)
        
        return root.children.get(max_state)
    
class HeuristicAlphaBetaTreeSearch:
    def select_move(self, game):
        player = game.get_current_player()

        def max_value(game, alpha, beta, depth):
            if game.is_game_over() or depth == 0:
                return game.calculate_utility(player)

            v = -float('inf')
            for action in game.get_available_moves():
                result_state = game.result_from_move(action)
                v = max(v, min_value(result_state, alpha, beta, depth - 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(game, alpha, beta, depth):
            if game.is_game_over() or depth == 0:
                return game.calculate_utility(player)

            v = float('inf')
            for action in game.get_available_moves():
                result_state = game.result_from_move(action)
                v = min(v, max_value(result_state, alpha, beta, depth - 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        depth = 4  

        for action in game.get_available_moves():
            result_state = game.result_from_move(action)
            value = min_value(result_state, alpha, beta, depth)
            if value > best_value:
                best_value = value
                best_move = action
                alpha = max(alpha, best_value)

        return best_move
    
class MonteCarloTreeSearch(Player):
    def get_move(self, board):
        
        return random.choice([col for col in range(7) if board[0][col] == ' '])

class HeuristicAlphaBetaTreeSearch(Player):
    def get_move(self, board):
        
        return random.choice([col for col in range(7) if board[0][col] == ' '])