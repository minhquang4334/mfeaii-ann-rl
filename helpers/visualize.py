import numpy as np
import matplotlib.pyplot as plt
from .instance import *

repeat = 1
NUMBER_OF_ALG1_TASK = 3
NUMBER_OF_ALG2_TASK = 1
ALG1 = "CEA"
ALG2 = "MFEA"
ALG3 = "MFEAII"
problem="ionosphere"
def load(algorithm):
    results = []
    for seed in range(1, repeat + 1):
        stats = np.load('results/%s/%s/%d.npy' % (problem, algorithm, seed))
        results.append(stats)
    return np.array(results)

def convergence_train(instance):
    # config = get_config('config.yaml')
    # conn = create_connection(config)
    # cur = conn.cursor()
    # cur.execute('SELECT TABLE iteration ADD COLUMN rmp VARCHAR(128);')
    
    fig, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    for k in range(3):
        results1 = instance[k][0]
        results2 = instance[k][1]
        results3 = instance[k][2]
        print(results1.shape, results2.shape, results3.shape)
        ax = axes[k]

        # CEA
        result = results1[:, :]

        mu = np.mean(result, axis = 0)
        sigma = np.std(result, axis = 0)
        x = np.arange(result.shape[1])

        line1, = ax.plot(x, mu, color = "blue")
        ax.fill_between(x, mu + 0, mu - 0, color = "blue", alpha = 0.3)

        # MFEA
        # result = results2[:, :, k]
        result = results2[:, :]

        mu = np.mean(result, axis = 0)
        sigma = np.std(result, axis = 0)
        x = np.arange(result.shape[1])

        line2, = ax.plot(x, mu, color = "red")
        ax.fill_between(x, mu + 0, mu - 0, color = "red", alpha = 0.3)

        # MFEA2
        # result = results2[:, :, k]
        result = results3[:, :]

        mu = np.mean(result, axis = 0)
        sigma = np.std(result, axis = 0)
        x = np.arange(result.shape[1])

        line3, = ax.plot(x, mu, color = "green")
        ax.fill_between(x, mu + 0, mu - 0, color = "green", alpha = 0.3)

        # Legend
        ax.grid()
        ax.legend((line1, line2, line3), (ALG1, ALG2, ALG3))
    plt.show()
    plt.savefig("mean_and_std.eps", format = "eps")

color = ['red', 'blue', 'green']
def convergence(instance, instances_name, X_Range):
    # config = get_config('config.yaml')
    # conn = create_connection(config)
    # cur = conn.cursor()
    # cur.execute('SELECT TABLE iteration ADD COLUMN rmp VARCHAR(128);')
    
    fig, axes = plt.subplots(1, 2)
    index = 0
    line = []
    p1 = (instance[0], instance[1], instance[2])
    p2 = (instance[3], instance[4], instance[5])
    instances1 = (instances_name[0], instances_name[1], instances_name[2])
    instances2 = (instances_name[3], instances_name[4], instances_name[5])
    for i in range(2):
        ax = axes[i]
        if(i == 0): 
            p = p1
            i_name = instances1
        else: 
            p = p2
            i_name = instances2

        for result in p:
            print (result.shape, index)
            result = np.asarray(result).astype(np.float)
            mu = np.mean(result, axis = 0)
            print (mu.shape)
            mu = mu.flatten()
            sigma = np.std(result, axis = 0)
            sigma = sigma.flatten()
            x = X_Range[index]
            x = np.squeeze(x, axis=0)
            print (mu.shape, x.shape)
            line1, = ax.plot(x, mu, color = color[index])
            ax.fill_between(x, mu + 0, mu - 0, color = color[index], alpha = 0.3)
            line.append(line1)
            if(index == 2):
                index = 0
            else:
                index = index + 1
            ax.grid()
            ax.legend(tuple(line), tuple(i_name))
        # Legend

   
    plt.show()
    plt.savefig("mean_and_std_sgd.eps", format = "eps")    

if __name__ == "__main__":
    list_instances = get_list_instance_name()
    # for name in list_instances:
    #     instance = Instance(config, name)
    #     print (name, len(instance.results_by_tasks), len(instance.results_by_tasks[0]))
    instance = Instance(config, "nbit_4_1")
    print (len(instance.results_by_tasks), len(instance.results_by_tasks[0]))
