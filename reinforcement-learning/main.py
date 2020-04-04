import time
import argparse
import numpy as np
from helpers import *
from tasks import Sphere, CartPole, Acrobot, FlappyBird

def optimizer(tasks, logger, time_logger, num_pop, num_gen, sbxdi, pmdi, rmp):
    start_at = time.time()
    elapsed_time = 0

    # Initialize
    num_dim = max([task.dim for task in tasks])
    num_task = len(tasks)
    num_pop = num_pop * num_task

    population = np.random.rand(2 * num_pop, num_dim)
    skill_factor = np.array([i % num_task for i in range(2 * num_pop)])
    factorial_cost = np.full([2 * num_pop, num_task], np.inf)
    scalar_fitness = np.empty([2 * num_pop])

    elapsed_time += time.time() - start_at
    # Evaluate
    for i in range(2 * num_pop):
        sf = skill_factor[i]
        factorial_cost[i, sf] = tasks[sf].fitness(population[i])
    start_at = time.time()
    scalar_fitness = find_scalar_fitness(factorial_cost)

    # Sort
    sort_index = np.argsort(scalar_fitness)[::-1]
    population = population[sort_index]
    skill_factor = skill_factor[sort_index]
    factorial_cost = factorial_cost[sort_index]

    # Reset offspring fitness
    factorial_cost[num_pop:, :] = np.inf

    # Evolve
    for gen in range(num_gen):
        # permute current population
        permutation_index = np.random.permutation(num_pop)
        population[:num_pop] = population[:num_pop][permutation_index]
        skill_factor[:num_pop] = skill_factor[:num_pop][permutation_index]
        factorial_cost[:num_pop] = factorial_cost[:num_pop][permutation_index]

        # single task
        if rmp == 0:
            single_task_index = []
            for k in range(num_task):
                single_task_index += list(np.where(skill_factor[:num_pop] == k)[0])
            population[:num_pop] = population[:num_pop][single_task_index]
            skill_factor[:num_pop] = skill_factor[:num_pop][single_task_index]
            factorial_cost[:num_pop] = factorial_cost[:num_pop][single_task_index]

        for i in range(0, num_pop, 2):
            # select pair to crossover
            p1, p2 = population[i], population[i + 1]
            sf1, sf2 = skill_factor[i], skill_factor[i + 1]

            # crossover
            if sf1 == sf2:
                c1, c2 = sbx_crossover(p1, p2, sbxdi)
                skill_factor[num_pop + i] = sf1
                skill_factor[num_pop + i + 1] = sf1
            elif sf1 != sf2 and np.random.rand() < rmp:
                c1, c2 = sbx_crossover(p1, p2, sbxdi)

                # assign skill factor
                if np.random.rand() < 0.5:
                    skill_factor[num_pop + i] = sf1
                else:
                    skill_factor[num_pop + i] = sf2
                if np.random.rand() < 0.5:
                    skill_factor[num_pop + i + 1] = sf1
                else:
                    skill_factor[num_pop + i + 1] = sf2
            else:
                c1 = np.copy(p1)
                c2 = np.copy(p2)
                skill_factor[num_pop + i] = sf1
                skill_factor[num_pop + i + 1] = sf2

            # mutate
            c1 = mutate(c1, pmdi)
            c2 = mutate(c2, pmdi)
            sf1 = skill_factor[num_pop + i]
            sf1 = skill_factor[num_pop + i + 1]

            # assign
            population[num_pop + i, :], population[num_pop + i + 1, :] = c1[:], c2[:]

        # Evaluate
        elapsed_time += time.time() - start_at
        for i in range(num_pop, 2 * num_pop):
            sf = skill_factor[i]
            factorial_cost[i, sf] = tasks[sf].fitness(population[i])
        start_at = time.time()
        scalar_fitness = find_scalar_fitness(factorial_cost)

        # Sort
        sort_index = np.argsort(scalar_fitness)[::-1]
        population = population[sort_index]
        skill_factor = skill_factor[sort_index]
        factorial_cost = factorial_cost[sort_index]

        # Reset offspring fitness
        factorial_cost[num_pop:, :] = np.inf

        best_fitness = np.min(factorial_cost, axis=0)
        mean_fitness = [np.mean(factorial_cost[:, i][np.isfinite(factorial_cost[:, i])]) for i in range(num_task)]
        info = ','.join([str(gen), 
                         ','.join(['%f' % _ for _ in best_fitness]),
                         ','.join(['%f' % _ for _ in mean_fitness]),
            ])
        logger.info(info)

    elapsed_time += time.time() - start_at
    time_logger.info('%f' % elapsed_time)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pop', type=int, default=20)
    parser.add_argument('--num_gen', type=int, default=100)
    parser.add_argument('--sbxdi', type=int, default=2)
    parser.add_argument('--pmdi', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=20)

    args = parser.parse_args()
    tasks = [CartPole(0.8 + i * 10) for i in range(10)]
    # tasks = [Acrobot(1.0 + 0.1 * i) for i in range(5)]
    # tasks = [FlappyBird(0.8 + 0.1 * i) for i in range(5)]
    TASK_NAME = 'cartpole'
    for exp_id in range(6, args.repeat):
        logger = get_logger(TASK_NAME, 'st_%d' % exp_id)
        time_logger = get_logger(TASK_NAME, 'st_time_%d' % exp_id)
        optimizer(tasks, logger, time_logger, args.num_pop, args.num_gen, args.sbxdi, args.pmdi, 0)

        logger = get_logger(TASK_NAME, 'mt_%d' % exp_id)
        time_logger = get_logger(TASK_NAME, 'mt_time_%d' % exp_id)
        optimizer(tasks, logger, time_logger, args.num_pop, args.num_gen, args.sbxdi, args.pmdi, 0.3)

if __name__ == '__main__':
    main()
