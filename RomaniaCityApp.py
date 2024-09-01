# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:35:07 2023

@author: adina_l1uzsjt
"""
from SimpleProjectSolvingAgent import *


def main():

    while True:
        #prints out the cities
        print("Here are all the possible Romania cities that can be traveled:")
        for city in romania_map.graph_dict.keys():
            print(city)

        
        print("Please enter the origin city: ")
        origin_city = input()
        #checks if origin city is valid
        while origin_city not in romania_map.locations:
                print(f"Could not find {origin_city} please try again:")
                origin_city = input()
    
        print("Please enter the destination city: ")
        destination_city = input()
        #checks if the cities are the same
        while destination_city not in romania_map.locations or destination_city == origin_city:
            #checks if destination city is valid
            if destination_city not in romania_map.locations:
                print(f"Could not find {destination_city} please try again:")
                print("Please enter the destination city: ")
                destination_city = input()
            else:
                print("The same city can't be both origin and destination. Please try again.")
                print("Please enter the origin city:")
                origin_city = input()
                print("Please enter the destination city: ")
                destination_city = input()
        
        #create a Problem instance
        problem = Problem(origin_city,destination_city,romania_map)
        #perform greedy best-first search
        print("Greedy Best-First Search:")
        #result_greedy = greedy_best_first_search(problem)
        #print_solution(result_greedy)
    
        #perform A* search
        print("\nA* Search:")
        #result_astar = a_star_search(problem)
        #print_solution(result_astar)
    
        #perform hill climbing
        print("\nHill Climbing:")
        #result_hill_climbing = hill_climbing(problem)
        #print_solution(result_hill_climbing)
    
        #perform simulated annealing
        print("\nSimulated Annealing:")
        #result_simulated_annealing = simulated_annealing(problem)
        #print_solution(result_simulated_annealing)
        
        #ask if user would like to find the path between any other two cities
        new_path = input("Would you like to find the best path between and other two cities?")
        if new_path.lower() != 'yes':
            print("Thank You for Using Our App")
            break

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
    
    main()
    
