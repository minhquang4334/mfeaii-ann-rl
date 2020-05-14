from mfea_ii_lib import *

def mfeaii(taskset, config, callback=None, problem="mfea-ann"):
    # unpacking hyper-parameters
    if(problem == "mfea-ann"):
        K = len(taskset.config['hiddens'])
        N = config['pop_size'] * K
        D = taskset.D_multitask
        dims = taskset.dims
    if(problem == "mfea-rl"):
        K = len(taskset)
        N = config['pop_size'] * K
        D = max([task.dim for task in taskset])

    T = config['num_iter']
    sbxdi = config['sbxdi']
    pmdi  = config['pmdi']
    pswap = config['pswap']
    # rmp_matrix = np.zeros([K, K])
    rmp_matrix = np.eye(K)

    # initialize
    population = np.random.rand(2 * N, D)
    skill_factor = np.array([i % K for i in range(2 * N)])
    factorial_cost = np.full([2 * N, K], np.inf)
    scalar_fitness = np.empty([2 * N])

    # evaluate
    for i in range(2 * N):
        sf = skill_factor[i]
        if(problem == "mfea-ann"):
            factorial_cost[i, sf] = taskset.evaluate(population[i], sf)
        if(problem == "mfea-rl"):
            factorial_cost[i, sf] = taskset[sf].evaluate(population[i])
        # factorial_cost[i, sf] = functions[sf](population[i])
    scalar_fitness = calculate_scalar_fitness(factorial_cost)

    # sort 
    sort_index = np.argsort(scalar_fitness)[::-1]
    population = population[sort_index]
    skill_factor = skill_factor[sort_index]
    factorial_cost = factorial_cost[sort_index]

    # evolve
    iterator = trange(T)
    for t in iterator:
        # permute current population
        permutation_index = np.random.permutation(N)
        population[:N] = population[:N][permutation_index]
        skill_factor[:N] = skill_factor[:N][permutation_index]
        factorial_cost[:N] = factorial_cost[:N][permutation_index]
        factorial_cost[N:] = np.inf

        # learn rmp
        subpops    = get_subpops(population, skill_factor, N)
        if(problem == "mfea-ann"):
            # subpops = taskset.decode_pop_to_task_size(subpops)
            rmp_matrix = learn_rmp(subpops, [D]*3)
        if(problem == "mfea-rl"):
            rmp_matrix = learn_rmp(subpops, [D]*3)

        # select pair to crossover
        for i in range(0, N, 2):
            p1, p2 = population[i], population[i + 1]
            sf1, sf2 = skill_factor[i], skill_factor[i + 1]

            # crossover
            if sf1 == sf2:
                c1, c2 = sbx_crossover(p1, p2, sbxdi)
                c1 = mutate(c1, pmdi)
                c2 = mutate(c2, pmdi)
                c1, c2 = variable_swap(c1, c2, pswap)
                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf1
            elif sf1 != sf2 and np.random.rand() < rmp_matrix[sf1, sf2]:
                c1, c2 = sbx_crossover(p1, p2, sbxdi)
                c1 = mutate(c1, pmdi)
                c2 = mutate(c2, pmdi)
                # c1, c2 = variable_swap(c1, c2, pswap)
                if np.random.rand() < 0.5: 
                    skill_factor[N + i] = sf1
                    skill_factor[N + i + 1] = sf2
                else: 
                    skill_factor[N + i] = sf2
                    skill_factor[N + i + 1] = sf1
            else:
                # p2  = find_relative(population, skill_factor, sf1, N)
                # c1, c2 = sbx_crossover(p1, p2, sbxdi)
                # c1 = mutate(c1, pmdi)
                # c2 = mutate(c2, pmdi)
                # c1, c2 = variable_swap(c1, c2, pswap)
                # skill_factor[N + i] = sf1
                # skill_factor[N + i + 1] = sf1
                p1_  = find_relative(population, skill_factor, sf1, N)
                c, c_ = sbx_crossover(p1, p1_, sbxdi)
                c = mutate(c, pmdi)
                c_ = mutate(c_, pmdi)
                c, c_ = variable_swap(c, c_, pswap)
                c1 = c

                p2_  = find_relative(population, skill_factor, sf2, N)
                c, c_ = sbx_crossover(p2, p2_, sbxdi)
                c = mutate(c, pmdi)
                c_ = mutate(c_, pmdi)
                c, c_ = variable_swap(c, c_, pswap)
                c2 = c

                skill_factor[N + i] = sf1
                skill_factor[N + i + 1] = sf2

            population[N + i, :], population[N + i + 1, :] = c1[:], c2[:]

        # evaluate
        for i in range(N, 2 * N):
            sf = skill_factor[i]
            if(problem == "mfea-ann"):
                factorial_cost[i, sf] = taskset.evaluate(population[i], sf)
            if(problem == "mfea-rl"):
                factorial_cost[i, sf] = taskset[sf].evaluate(population[i])
            # factorial_cost[i, sf] = functions[sf](population[i])
        scalar_fitness = calculate_scalar_fitness(factorial_cost)
        
        # begin quang test limit parents, offspring amounts
        #follow by mfea2 precise: select parents and offsprings with ratio parents/offsprings = 1/2 on each task

        tmp_population = np.random.rand(2 * N, D)
        tmp_scalar_fitness = np.empty([2 * N])
        tmp_factorial_cost = np.full([2 * N, K], np.inf)
        tmp_skill_factor = np.array([i % K for i in range(2 * N)])
        idx_start = 0
        n_size = config['pop_size']
        p_rate = config['parents_rate']
        idx_end = n_size
        for k in range(K):
            # get parents
            # index = np.where(skill_factor[:N] == k)
            # scalar_fitness_parents = scalar_fitness[:N][index]
            # population_parents = population[:N][index]
            # factorial_cost_parents = factorial_cost[:N][index]

            # sort_index = np.argsort(scalar_fitness_parents)[::-1][:int(np.floor(n_size * p_rate))]
            # scalar_fitness_parents = scalar_fitness_parents[sort_index]
            # population_parents = population_parents[sort_index]
            # factorial_cost_parents = factorial_cost_parents[sort_index]
            
            # # get offsprings
            # index = np.where(skill_factor[N:] == k)
            # scalar_fitness_offsprings = scalar_fitness[N:][index]
            # population_offsprings = population[N:][index]
            # factorial_cost_offsprings = factorial_cost[N:][index]

            # sort_index = np.argsort(scalar_fitness_offsprings)[::-1][:int(np.round(n_size * (1-p_rate)))]
            # scalar_fitness_offsprings = scalar_fitness_offsprings[sort_index]
            # population_offsprings = population_offsprings[sort_index]
            # factorial_cost_offsprings = factorial_cost_offsprings[sort_index]

            index = np.where(skill_factor == k)
            scalar_fitness_parents = scalar_fitness[index]
            population_parents = population[index]
            factorial_cost_parents = factorial_cost[index]

            sort_index = np.argsort(scalar_fitness_parents)[::-1][:int(n_size)]
            scalar_fitness_parents = scalar_fitness_parents[sort_index]
            population_parents = population_parents[sort_index]
            factorial_cost_parents = factorial_cost_parents[sort_index]
            
            
            #save sort items
            tmp_population[idx_start:idx_end] = population_parents
            tmp_scalar_fitness[idx_start:idx_end] = scalar_fitness_parents
            tmp_factorial_cost[idx_start:idx_end] = factorial_cost_parents
            tmp_skill_factor[idx_start:idx_end] = np.array([k] * n_size)
            idx_start = idx_end
            idx_end += n_size

        sort_index = np.argsort(tmp_scalar_fitness[:N])[::-1]
        # print(sort_index, np.asarray(sort_index).shape)
        population[:N] = tmp_population[sort_index] 
        scalar_fitness[:N] = tmp_scalar_fitness[sort_index] 
        factorial_cost[:N] = tmp_factorial_cost[sort_index] 
        skill_factor[:N] = tmp_skill_factor[sort_index]


        # end quang test limit parents, offspring amounts
        # sort
        # sort_index = np.argsort(scalar_fitness)[::-1]
        # population = population[sort_index]
        # skill_factor = skill_factor[sort_index]
        # factorial_cost = factorial_cost[sort_index]
        # scalar_fitness = scalar_fitness[sort_index]

        best_fitness = np.min(factorial_cost, axis=0)
        c1 = population[np.where(skill_factor == 0)][0]
        c2 = population[np.where(skill_factor == 1)][0]

        # optimization info
        message = {'algorithm': 'mfeaii', 'rmp':'{} - {} - {}'.format(np.around(rmp_matrix[0, 1], 4), np.around(rmp_matrix[0, 2], 4), np.around(rmp_matrix[1, 2],4))}
        # message = {'algorithm': 'mfeaii', 'rmp':'{}'.format(np.around(rmp_matrix[0, 1], 4))}
        result = get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message)
        if callback:
            callback(result)
        desc = 'gen:{} fitness:{} message:{}'.format(t, ' '.join('{:0.4f}'.format(res.fun) for res in result), message)
        iterator.set_description(desc)
            

