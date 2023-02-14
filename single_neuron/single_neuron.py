import numpy as np
import matplotlib.pyplot as plt
import math 
from ordered_set import OrderedSet
import itertools

def learn(w_before, data, labels, n):
    perdict_set = []
    training_data = data
    training_labels = labels
    weights = w_before
    b=1


    for _ in range(n):
        for i, data in enumerate(training_data):
            prediction = activation_function(n_out(weights, data, b))
            perdict_set.append(prediction)
            
            error = error_function(prediction, training_labels[i])
            
            if error != 0:
                weights = weights[0]+(error*data[0]), weights[1]+(error*data[1])
                b += error
                
        print(perdict_set)
        perdict_set.clear()
    return weights, b

def n_out(w, x, b):
    return w[0]*x[0] + w[1]*x[1]+b

def activation_function(x):
    if x <= 0:
        return 0
    return 1

def error_function(prediction, actual):
    return actual - prediction

def main():
    
    
    # Datasets
    set_1 = [(-9,-10), (-8, 20), (-6, 5), (2, 20), (4,25)]
    set_2 = [(-6, -25), (-2, -10), (1, -10), (4, -16), (5,9)]
    training_data = set_1 + set_2
    
    # Labels
    training_labels = [0,0,0,0,0,1,1,1,1,1]
    
    # Weight generator
    random_gen = np.random.RandomState(1)
    weight = random_gen.normal(loc = 0.0, scale = 0.01, size = 2 )
    weights = (weight[0], weight[1])
    
    # Training
    weights, b = learn(weights,training_data,training_labels,5)
     
    print("weight t :",weights)
    print("b t :",b)
 
    
    test_set_1 = [(-6, 15), (-5, 19), (-4, 12), (1, 19), (-8, -1)]
    test_set_2 = [(2, 1), (5, 7), (6, 20), (10, 2), (6,-5)]
    test_data = set_1 + set_2 + test_set_1 + test_set_2
    test_labels = training_labels + [0,0,0,0,0,1,1,1,1,1]

    weights2, b2 = learn(weights, test_data, test_labels, 8)

    print("weight2 :",weights2)
    print("b2 :",b2)

    
    # Draw a plot
    x1, y1 = zip(*set_1)
    x2, y2 = zip(*set_2)
    x1t, y1t = zip(*test_set_1)
    x2t, y2t = zip(*test_set_2)
    
    fig, ax = plt.subplots()
    xmin, xmax = -10, 10
    X = np.arange(xmin, xmax, 1)

    plt.scatter(x1, y1, color="blue")
    plt.scatter(x2, y2, color="red")
    plt.scatter(x1t, y1t, color="lightblue")
    plt.scatter(x2t, y2t, color="orange")

    # Tytuł i nazwy osi
    plt.title("Wykres punktowy")
    plt.xlabel("X")
    plt.ylabel("Y")
    a = -weights2[0]/weights2[1]
    c = -b2/weights[1]
    print(a,c)
    ax.plot(X, a * X + c )
    plt.show()

if __name__ == "__main__":
    main()