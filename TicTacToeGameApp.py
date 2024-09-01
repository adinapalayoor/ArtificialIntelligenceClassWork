# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:16:08 2023

@author: adina_l1uzsjt
"""

from TicTacToeClass import TicTacToe

# Define a dictionary to map user input to player classes
player_classes = {
    1: "Random Player",
    2: "MiniMax Player",
    3: "Alpha Beta Player",
    4: "Heuristic Alpha Beta Player",
    5: "MCTS Player",
    6: "Query Player",
}

def main():
    while True:
        #create TTT object
        tic_tac_toe_game = TicTacToe()
        print("Player Selection:")
        for num, player in player_classes.items():
            print(f"{num}. {player}")
        #ask for user input for players
        player_X_type = input("Please enter the type of player for Player X: ")
        tic_tac_toe_game.select_player_X(player_X_type)
        player_O_type = input("Please enter the type of player for Player O: ")
        tic_tac_toe_game.select_player_O(player_O_type)

        while not tic_tac_toe_game.is_game_over():
            if tic_tac_toe_game.get_current_player() == 'X':
                tic_tac_toe_game.play_game()
            else:
                tic_tac_toe_game.play_game()
        #check if user wants to play again
        play_again = input("Do you want to play again? (yes/no): ")
        if play_again.lower() != 'yes':
            print("Thank You for Playing Our Game")
            break

if __name__ == "__main__":
    main()




