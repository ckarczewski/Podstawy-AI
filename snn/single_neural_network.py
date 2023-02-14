import numpy as np

def learning(weight, x_train, t_labels, n):
    """
    weight - weight matrix before learning
    x_train - training data
    t_labels - training label
    n - iteration
    """
    example_size = x_train.shape[1]
    learning_rate = 0.1
    
    for i in range(n):
        example_nr = round(np.random.uniform(low=1, high=example_size))-1
        x = x_train[:,example_nr]
        y = nn_output(weight, x)
        error = t_labels[:,example_nr] - y
        d_error = error_function(error, y)
        d_weight = np.dot(learning_rate * x, d_error.T)
        weight = weight + d_weight
    
    return weight

def weight_generator(network_input, neurons):
    random_gen = np.random.RandomState(1)
    weight = random_gen.normal(loc = 0.0, scale = 0.01, size = (network_input, neurons))
    return weight

def nn_output(weights, data):
    U = weights.T * data
    return activation_function(U)

def activation_function(U):
    beta = 5
    return 1 / (1 + np.exp(-beta * U))

def error_function(error,y):
    beta = 5
    derivative = np.multiply((beta*y), (1-y))
    return np.multiply(error, derivative)

def main():
    training_data = np.matrix([[4, 2, -1], 
                              [0.01, -1, 3.5], 
                              [0.01, 2, 0.01], 
                              [-1, 2.5, -2],
                              [-1.5, 2, 1.5]])
    
    training_labels = np.matrix([[1,0,0], 
                                [0,1,0],
                                [0,0,1]])
    
    # weights = weight_generator(5,3)
    
    weights = np.matrix([[0.0629, -0.0805, -0.0685], 
                        [0.0812, -0.0443, 0.0941], 
                        [-0.0746, 0.0094, 0.0914], 
                        [0.0827, 0.0915, -0.0029],
                        [0.0265, 0.0930, 0.0601]])
    
    print("weight before \n", weights)
    y_before = nn_output(weights,training_data)
    print("y before learning \n", y_before.round(4))
 
    weights_trained = learning(weights,training_data,training_labels,10)
    
    print("weight after learning \n", weights_trained.round(4))
    y_learned = nn_output(weights_trained, training_data)
    print("y after learning \n", y_learned.round(4))
    print("expected y\n", training_labels)


if __name__ == "__main__":
    main()