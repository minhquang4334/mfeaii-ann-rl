import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

plt.style.use('ggplot')
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['lines.linewidth'] = 2

exp = 'flappybird'
names = ['t'] + ['best_%d' % _ for _ in range(5)] + ['mean_%d' % _ for _ in range(5)]

def get_average(type, algo, task_id):
    result = []
    for i in range(3):
        print('data/%s/%s_%d.csv' % (exp, algo, i))
        df = pd.read_csv('data/%s/%s_%d.csv' % (exp, algo, i), names=names)
        for i in [task_id]:        
            result.append(df['%s_%d' % (type, i)])
    return - np.mean(result, axis=0)

def get_std(type, algo, task_id):
    result = []
    for i in range(3):
        df = pd.read_csv('data/%s/%s_%d.csv' % (exp, algo, i), names=names)
        for i in [task_id]:
            result.append(df['%s_%d' % (type, i)])
    return np.std(result, axis=0)

def plot(type, algo, label, task_id):
    y = get_average(type, algo, task_id)
    x = np.arange(0, y.shape[0], 1)
    e = get_std(type, algo, task_id)
    plt.plot(x, y, label=label)
    plt.fill_between(x, y - e, y + e, alpha=0.5)

for task_id in range(5):
    plt.subplot(3, 2, task_id + 1)
    plot('mean', 'st', 'ea', task_id)
    plot('mean', 'mt', 'mfea', task_id)
    plt.ylabel('Fitness (Total Rewards)')
    plt.xlabel('Generations\n' + 'Task %d, Gravity %0.2f' % (task_id, (0.8 + 0.1 * task_id) * 9.8))
plt.legend()
plt.show()
