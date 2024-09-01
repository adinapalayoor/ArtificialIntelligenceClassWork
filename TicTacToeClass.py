# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:15:44 2023

@author: adina_l1uzsjt
"""
import numpy as np
import random
import math

class TicTacToe:
    def __init__(self):
        #initialize a tic tac toe game
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.rounds = 3
        self.player_X = None
        self.player_O = None
        self.utility_X = 0  # Initialize utility for player X
        self.utility_O = 0  # Initialize utility for player O
        
    def get_current_player(self):
        #returns the X or O player
        return self.current_player

    def select_player_X(self, player_type):
        #select the type of player X and the corresponding class
        player_mapping = {
            1: RandomPlayer(),
            2: MiniMaxPlayer(),
            3: AlphaBetaPlayer(),
            4: HeuristicPlayer(),
            5: MCTSPlayer(),
            6: QueryPlayer()
        }
        #ensure that the number entered is between 1-6
        while int(player_type) not in player_mapping:
            print("Invalid player type. Please select a number between 1 and 6.")
            player_type = input("Please enter the type of player for Player X: ")

        self.player_X = player_mapping[int(player_type)]

    def select_player_O(self, player_type):
        #select the type of player O and the corresponding class
        player_mapping = {
            1: RandomPlayer(),
            2: MiniMaxPlayer(),
            3: AlphaBetaPlayer(),
            4: HeuristicPlayer(),
            5: MCTSPlayer(),
            6: QueryPlayer()
        }
        #ensure player type is between 1-6
        while int(player_type) not in player_mapping:
            print("Invalid player type. Please select a number between 1 and 6.")
            player_type = input("Please enter the type of player for Player O: ")

        self.player_O = player_mapping[int(player_type)]

    def make_player_X_move(self):
        #move player X
        if self.player_X:
            move = self.player_X.select_move(self)
            self.make_move(move)

    def make_player_O_move(self):
        #move player O
        if self.player_O:
            move = self.player_O.select_move(self)
            self.make_move(move)

    def make_move(self, move):
        #make the move on the board as either an X or O
        row, col = move
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'X' if self.current_player == 'O' else 'O'
            
            # Update utility after each move
            winner = self.check_winner()
            if winner == 'X':
                self.utility_X += 1
                self.utility_O -= 1
            elif winner == 'O':
                self.utility_X -= 1
                self.utility_O += 1

    def get_available_moves(self):
        #returns all possible moves
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == ' ':
                    moves.append((row, col))
        return moves

    def check_winner(self):
        # Check rows, columns, and diagonals for a winner
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return self.board[0][i]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        return None

    def display_board(self):
        #display the board with 3x3 dots 
        for row in self.board:
            formatted_row = ['X' if cell == 'X' else 'O' if cell == 'O' else '.' for cell in row]
            print(' '.join(formatted_row))
        print()

    def play_game(self):
        #keep track of wins each round
        player_X_wins =0
        player_O_wins =0
        for round in range(self.rounds):
            print(f"Round {round + 1}:")
            self.board = [[' ' for _ in range(3)] for _ in range(3)]  # Reset the game board
    
            while True:
                #prints all available actions by current player
                print(f"Available Action by the Player {self.current_player}: {self.get_available_moves()}")
                if self.current_player == 'X':
                    move = self.player_X.select_move(self)
                else:
                    move = self.player_O.select_move(self)
    
                if move is None:
                    print("The game was a draw.")
                    break
    
                print("The Action by the Player", self.current_player, "Is", move)
                self.make_move(move)
                self.display_board()
                winner = self.check_winner()
                if winner:
                    #show winner of round
                    if winner == 'X':
                        print("Player X wins this round.")
                        player_X_wins += 1
        
                    elif winner == "O":
                        print("Player O wins this round.")
                        player_O_wins += 1
                    break
                else:
                   # print(f"Player X's Utility: {self.utility_X}")
                
                    print("current state:")
                    self.display_board()
        print("Overall Results:")
        #print winner of all three rounds
        if player_X_wins > player_O_wins:
            print("Player X can win two out of three rounds in the game.")
            print("Player X is the winner.")
        elif player_O_wins > player_X_wins:
            print("Player O can win two out of three rounds in the game.")
            print("Player O is the winner.")
        else:
            print("No Player can win two out of three rounds in the game.")
            print("The game was a draw.")
        
    def is_game_over(self):
        #check if game is over
        return self.check_winner() is not None or not self.get_available_moves()


    def result_from_move(self, move):
        #show result from move displayed as new TTT game
        new_game = TicTacToe()
        new_game.board = [[' ' for _ in range(3)] for _ in range(3)]
        new_game.current_player = 'X' if self.current_player == 'O' else 'O'

        for row in range(3):
            for col in range(3):
                new_game.board[row][col] = self.board[row][col]

        new_game.make_move(move)
        return new_game
    
    def calculate_utility(self, player):
        #calculate utility of game
        if player == 'X':
            opponent = 'O'
        else:
            opponent = 'X'
        
        winner = self.check_winner()
        if winner == player:
            return 1  # Player wins
        elif winner == opponent:
            return -1  # Opponent wins
        else:
            return 0  # It's a draw or the game is still ongoing




class RandomPlayer:
    def select_move(self, game):
        available_moves = game.get_available_moves()
        if not available_moves:
            return None  # No available moves; the game is a draw
        return random.choice(available_moves)

    
class MiniMaxPlayer:
    def select_move(self, game):
        player = game.get_current_player()

        def max_value(game):
            if game.is_game_over():
                return game.calculate_utility(player)

            v = -float('inf')
            for move in game.get_available_moves():
                new_game = game.result_from_move(move)
                v = max(v, min_value(new_game))
            return v

        def min_value(game):
            if game.is_game_over():
                return game.calculate_utility(player)

            v = float('inf')
            for move in game.get_available_moves():
                new_game = game.result_from_move(move)
                v = min(v, max_value(new_game))
            return v

        best_move = None
        best_value = -float('inf')
        for move in game.get_available_moves():
            new_game = game.result_from_move(move)
            value = min_value(new_game)
            if value > best_value:
                best_value = value
                best_move = move

        return best_move


class AlphaBetaPlayer:
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

class HeuristicPlayer:
    def select_move(self, game):
        player = game.get_current_player()
        best_score = -np.inf
        best_move = None

        def alpha_beta_cutoff_search(state, depth, alpha, beta):
            if game.is_game_over() or depth == 0:
                return game.calculate_utility(player)

            for action in game.get_available_moves():
                result_state = game.result_from_move(action)
                score = -alpha_beta_cutoff_search(result_state, depth - 1, -beta, -alpha)

                if score >= beta:
                    return score

                if score > alpha:
                    alpha = score
                    if depth == 4:
                        best_move = action

            return alpha

        for move in game.get_available_moves():
            result_state = game.result_from_move(move) 
            score = -alpha_beta_cutoff_search(result_state, 4, -np.inf, np.inf)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

class MCT_Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.N = 0
        self.U = 0

class MCTSPlayer:
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



class QueryPlayer:
    def select_move(self, game):
        print("Enter your move in the format (row, col):")
        while True:
            try:
                move = input("Your move: ")
                row, col = map(int, move.strip("()").split(","))

                if (row, col) in game.get_available_moves():
                    return (row, col)
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Please enter a move in the format (row, col).")
