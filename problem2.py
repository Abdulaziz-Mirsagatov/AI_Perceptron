from utills import *
from os import makedirs
import matplotlib.pyplot as plt

learning_rate = 0.01
epochs = 15

# make the necessary directories
makedirs("output", exist_ok=True)
[makedirs(f"output/perceptron_{i}", exist_ok=True) for i in range(10)]

generate_data_sets(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

# train the perceptron for recognizing 9s
simulate_perceptron("9", learning_rate, epochs)
