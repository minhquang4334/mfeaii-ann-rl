import numpy as np
import matplotlib.pyplot as plt
from .instance import *

repeat = 1
NUMBER_OF_ALG1_TASK = 3
NUMBER_OF_ALG2_TASK = 1
ALG1 = "CEA"
ALG2 = "MFEA"
ALG3 = "MFEAII"
ALG4 = "MFEAIIQ"
problem="ionosphere"
def load(algorithm):
    results = []
    for seed in range(1, repeat + 1):
        stats = np.load('results/%s/%s/%d.npy' % (problem, algorithm, seed))
        results.append(stats)
    return np.array(results)

def convergence_train(instance):
    K = len(instance)
    print(K)
    fig, axes = plt.subplots(1, K)
    print(instance.shape)
    # instance = (instance + -5) * -1 #if using to evaluate flappybird problem
    instance = -instance #pixcel copter problem
    # instance = instance
    for k in range(K):
        results1 = instance[k][0]
        results2 = instance[k][1]
        results3 = instance[k][2]
        # results4 = instance[k][3]
        print(results1.shape, results2.shape, results3.shape)
        ax = axes[k]
        # CEA
        result = results1[:, :]
        mu = np.mean(result, axis = 0)
        sigma = np.std(result, axis = 0)
        # sigma = 0
        x = np.arange(result.shape[1])

        line1, = ax.plot(x, mu, color = "blue")
        ax.fill_between(x, mu + sigma, mu - sigma, color = "blue", alpha = 0.3)
        
        # MFEA
        # result = results2[:, :, k]
        result = results2[:, :]

        mu = np.mean(result, axis = 0)
        sigma = np.std(result, axis = 0) 
        # sigma = 0
        x = np.arange(result.shape[1])
        line2, = ax.plot(x, mu, color = "red")
        ax.fill_between(x, mu + sigma, mu - sigma, color = "red", alpha = 0.3)

        # MFEA2
        # result = results2[:, :, k]
        result = results3[:, :]

        mu = np.mean(result, axis = 0)
        sigma = np.std(result, axis = 0)
        # sigma = 0
        x = np.arange(result.shape[1])
        line3, = ax.plot(x, mu, color = "green")
        ax.fill_between(x, mu + sigma, mu - sigma, color = "green", alpha = 0.3)

        # Legend
        ax.grid()
        # ax.legend((line1, line2, line3), (ALG1, ALG2, ALG3))
        ax.set_title("Tác vụ {}".format(k+1))
        ax.set_ylabel('Reward')
        ax.set_xlabel('Số thế hệ')
    # handles, labels = ax.get_legend_handles_labels()
    fig.legend((line1, line2, line3), (ALG1, ALG2, ALG3))
    # fig.suptitle('Biểu đồ hội tụ Acrobot', fontsize=14)
    plt.show()
    plt.savefig("mean_and_std.eps", format = "eps")

color = ['red', 'green', 'blue']
def convergence(instance, instances_name, X_Range):
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
            if i==0:
                ax.set_title("Biểu đồ hội tụ 3 tác vụ tối ưu với MFEA2")
                ax.set_ylabel('MSE')
                ax.set_xlabel('Số thế hệ')
            else:
                ax.set_title("Giá trị rmp của từng cặp tác vụ")
                ax.set_ylabel('RMP')
                ax.set_xlabel('Số thế hệ')
            ax.grid()
            ax.legend(tuple(line), tuple(i_name))
            
            
        # Legend

   
    plt.show()
    plt.savefig("mean_and_std_sgd.eps", format = "eps")    

# Plot of final score
import os
def compare_final_score(instance, algs = ["CEA", "MFEA-I", "MFEA-II"]):

    #instance = (instance[:, :, :, -1] + -5) * -1 #flappy bird
    instance = instance[:, :, :, -1] * -1
    # instance = (instance[:, :, :, -1]) # if using to compare EA problem
    print(instance.shape)
    number_seed = instance.shape[2]
    number_tasks = instance.shape[0]
    # y = np.empty((len(algs), len(_files)))
    # Draw bar chart of average final score
    x = np.arange(1, 1 + len(algs))
    fig, axes = plt.subplots(1, number_tasks)
    w = 0.6 # Bar width
    for idx in range(number_tasks):
        ax = axes[idx]
        y = instance[idx]
        for i in range(len(algs)):
            ax.bar(x[i],
                    height = np.mean(y[i]),
                    yerr = np.std(y[i]),
                    width = w,
                    color = color[i],
                    alpha = 0.3,
                    capsize = 12,
                    label = algs[i]
                    )
            ax.set_title("Tác vụ {}".format(idx+1))
            ax.set_ylabel("Reward Value")
            ax.grid()
    # Draw scatter point of final scores
    for idx in range(number_tasks):
        y = instance[idx]
        ax = axes[idx]
        for i in range(len(algs)):
            for j in range(number_seed):
                ax.scatter(x[i] - w/2 + j / number_seed * w, y[i][j], color = color[i])

    # Final touch
    
    # ax.set_title("Final scores - {}".format("FlappyBird"))
    # fig.suptitle('Phân bố kết quả cuối cùng Acrobot', fontsize=14)
    ax.set_xticks(x)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    list_instances = get_list_instance_name()
    instance = Instance(config, "nbit_4_1")
    print (len(instance.results_by_tasks), len(instance.results_by_tasks[0]))

