import numpy as np
import neurolab as nl
import pylab as pl
import matplotlib.pyplot as plt
import math 
from ordered_set import OrderedSet
import itertools
from statistics import mean 

def objective_function(population):
    # change binary to decimal and use adaptation function to get score
    return [adaptation_function(int(''.join(map(lambda x: str(int(x)), x)),2)) for x in population]

def adaptation_function(x):
    # adaptation function
    return 0.2 * np.sqrt(x) + 2*np.sin(2*np.pi*0.02*x) + 5

def adaptation_index(scor):
    # get adaptation 
    s_sum = sum(scor)
    return [(s/s_sum) for s in scor]

def roulette_wheel_selection(adaptation, population, n):
    # roulette wheel selection, selects chromosomes for the parent pool
    indexes = np.random.choice(len(population), n, p=adaptation)
    return [population[ix] for ix in indexes]

def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1)-1)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

def mutation(bit_string, r_mut):
    for i in range(len(bit_string)):
        # check for a mutation
        if np.random.rand() < r_mut:
            # flip the bit
            bit_string[i] = 1 - bit_string[i]
def main():
    n_pop = 50
    pk = [0.5, 0.6, 0.7, 0.8, 1]
    pm = 0
    children = []
    generations = 200
    fp = []
    fp_set = []

    for pki in pk:
        # create new population
        random_gen = np.random.RandomState(1)
        population = [random_gen.randint(0, 2, 8).tolist() for _ in range(n_pop)]
        for j in range(generations):
            # get y = f(x)
            scores = objective_function(population)
            # get adaptation
            adaptation = adaptation_index(scores)
            # get parent pool
            parent_pool = roulette_wheel_selection(adaptation, population, n_pop)

            # crossing / mutation - new population     
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                # print("i=",i,"i+1=",i+1)
                p1, p2 = parent_pool[i], parent_pool[i+1]
                # crossover and mutation
                for c in crossover(p1, p2, pki):
                    # mutation
                    mutation(c, pm)
                    # store for next generation
                    children.append(c)
            population = children.copy()
            children.clear()
            # calculate mean value of population
            m_fun = mean(objective_function(population))
            fp.append(m_fun)
            
            # return mean value in last generation 
            if j == 199:
                m200 = mean(objective_function(population))
                print(pki,round(m200,4))
        temp = {
            pki : fp.copy()
        }
        fp_set.append(temp)
        
        fp.clear()


    fig, ax = plt.subplots()
    X = [i for i in range(generations)]
    plt.title("fp / generations plot")
    plt.xlabel("Generations")
    plt.ylabel("fp")
    ax.plot(X, fp_set[0].get(0.5), label = "pk = 0.5")
    ax.plot(X, fp_set[1].get(0.6), label = "pk = 0.6")
    ax.plot(X, fp_set[2].get(0.7), label = "pk = 0.7")
    ax.plot(X, fp_set[3].get(0.8), label = "pk = 0.8")
    ax.plot(X, fp_set[4].get(1), label = "pk = 1")
    ax.legend()
    plt.show()
if __name__ == "__main__":
    main()