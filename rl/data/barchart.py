import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

plt.style.use('ggplot')
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'

# dataset = 'flappybird'
# num_run = 13
# num_task = 5

dataset = 'cartpole'
num_run = 14
num_task = 10

type = 'mean'

names = ['t'] + ['best_%d' % _ for _ in range(num_task)] + ['mean_%d' % _ for _ in range(num_task)]

meta = {
    'cartpole': [(0.8 + i * 10) for i in range(10)],
    'flappybird': [((0.8 + 0.1 * i) * 9.8) for i in range(5)],
}
reward_range = {
    'cartpole': [0, 200],
    'flappybird': [-5, 200]
}


def gather(algo, task_id):
    result = []
    for name in os.listdir(dataset):
        if re.search(r'%s_\d+\.csv' % algo, name):
            path = os.path.join(dataset, name)
            df = pd.read_csv(path, names=names)
            u = np.array(df['%s_%s' % (type, task_id)])[-1]
            result.append(-u)
    return np.array(result)

def main():
    x = []
    y = []
    for i in range(num_task):
        st = gather('st', i)
        mt = gather('mt', i)
        reward_min, reward_max = reward_range[dataset]
        m = np.mean(mt) - reward_min
        s = np.mean(st) - reward_min
        print(m, s)
        print(i, (m - s) / (reward_max - reward_min) * 100)
        x.append(i)
        y.append((m - s) / (reward_max - reward_min) * 100)
    width = 0.3
    plt.barh(x, y, width)
    plt.yticks([i for i in range(len(meta[dataset]))], ['%.2f m/s' % g for g in meta[dataset]])
    plt.xlim((-5, 15))

    for y_, x_ in enumerate(y):
        plt.text(x_, y_, '%0.2f%%' % y[y_])

    plt.xlabel('Percentage of improvement (%)')
    plt.ylabel('Gravity (m/s)')

    plt.show()

if __name__ == '__main__':
    main()
