# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:08:27 2023

@author: adina_l1uzsjt
"""

from ConnectFourClass import *
# Example usage
if __name__ == "__main__":
    player1 = MonteCarloTreeSearch('X')
    player2 = HeuristicAlphaBetaTreeSearch('O')

    game = ConnectFourGame(player1, player2)
    game.play()