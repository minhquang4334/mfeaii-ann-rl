import os
import logging
import numpy as np

'''
@ Multifactorial Evolutionary Algorithm
'''
def sbx_crossover(p1, p2, sbxdi):
    D = p1.shape[0]
    cf = np.empty([D])
    u = np.random.rand(D)        

    cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (sbxdi + 1)))
    cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (sbxdi + 1)))

    c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
    c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)

    c1 = np.clip(c1, 0, 1)
    c2 = np.clip(c2, 0, 1)

    return c1, c2

def mutate(p, pmdi):
    mp = float(1. / p.shape[0])
    u = np.random.uniform(size=[p.shape[0]])
    r = np.random.uniform(size=[p.shape[0]])
    tmp = np.copy(p)
    for i in range(p.shape[0]):
        if r[i] < mp:
            if u[i] < 0.5:
                delta = (2*u[i]) ** (1/(1+pmdi)) - 1
                tmp[i] = p[i] + delta * p[i]
            else:
                delta = 1 - (2 * (1 - u[i])) ** (1/(1+pmdi))
                tmp[i] = p[i] + delta * (1 - p[i])
    return tmp

'''
@ other ultilities
'''
def find_scalar_fitness(factorial_cost):
    return 1 / np.min(np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1, axis=1)

def get_best_individual(population, factorial_cost):
    K = factorial_cost.shape[1]
    p_bests = []
    y_bests = []
    for k in range(K):
        best_index = np.argmax(factorial_cost[:, k])
        p_bests.append(population[best_index, :])
        y_bests.append(factorial_cost[best_index, k])
    return p_bests, y_bests

def get_population_by_skill_factor(population, skill_factor_list, skill_factor):
    return population[np.where(skill_factor_list == skill_factor)]

def get_logger(env_name, exp_id):
    if not os.path.exists('data/%s' % env_name):
        os.mkdir('data/%s' % env_name)
    filename = 'data/%s/%s.csv' % (env_name, exp_id)
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    if os.path.exists(filename):
        os.remove(filename)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger