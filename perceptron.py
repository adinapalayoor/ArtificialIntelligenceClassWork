# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:19:55 2023

@author: adina_l1uzsjt
"""

from sklearn.linear_model import Perceptron
import numpy as np

def main():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 1])  # OR gate outputs

    per = Perceptron(random_state=42, max_iter=20,tol=.001, verbose=2)
    per.fit(X_train, y_train)
    
    test_data = np.array([[0, 1]])
    prediction = per.predict(test_data)
    print(f"Input: {test_data[0]}, Prediction: {prediction[0]}")

if __name__ == "__main__":
    main()
    