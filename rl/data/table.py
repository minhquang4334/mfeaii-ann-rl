import pandas as pd
import numpy as np
import os
import re

dataset = 'flappybird' # 
num_run = 6
num_task = 5

# dataset = 'cartpole'
# num_run = 14
# num_task = 10

type = 'mean'

names = ['t'] + ['best_%d' % _ for _ in range(num_task)] + ['mean_%d' % _ for _ in range(num_task)]

meta = {
    'cartpole': [(0.8 + i * 10) for i in range(10)],
    'flappybird': [((0.8 + 0.1 * i) * 9.8) for i in range(5)],
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

def gather_time(algo):
    total_time = 0.
    count = 0.
    for name in os.listdir(dataset):
        for i in range(num_run):
            name = '%s_time_%d.csv' % (algo, i)
            path = os.path.join(dataset, name)
            count += 1.
            total_time += float(open(path).read())
    return total_time / count

def main():
    for i in range(num_task):
        st = gather('st', i)
        mt = gather('mt', i)

        # st_time = gather_time('st')
        # mt_time = gather_time('mt')
        st_time = 2.01
        mt_time = 2.09

        print('gravity=%0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f \\\\' % (
            meta[dataset][i],
            np.max(st), np.mean(st), np.std(st), st_time,
            np.max(mt), np.mean(mt), np.std(mt), mt_time))

if __name__ == '__main__':
    main()