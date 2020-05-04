import numpy as np
from .input_handler import mapping
import yaml

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def mse(y, y_pred):
    return 0.5 * np.mean(np.power(y - y_pred, 2))

class mtlTaskset:
    '''Including:
        - a dataset
        - three ANN configuration
    '''
    def __init__(self, config):
        self.config = config
        self.X, self.y = mapping[self.config['name']](self.config['path'])
        self.n_tasks = len(self.config['hiddens'])
        self.h_in = self.config['input']
        self.h_out = self.config['output']
        self.hiddens = self.config['hiddens']

    @property
    def H_task(self):
        h = []
        for idx in range(self.n_tasks):
            tmp = np.concatenate([[self.h_in], self.hiddens[idx], [self.h_out]])
            h.append(tmp)
        return h

    @property
    def L_task(self):
        return  [len(h) - 1 for h in self.H_task]

    @property
    def L_max(self):
        return max(self.L_task)

    @property
    def H_multitask(self):
        h_theta_multitask = max(self.H_task[idx][l-1] for idx, l in enumerate(self.L_task))
        h = np.zeros(self.L_max + 1, dtype=int)
        h[0] = self.h_in
        idx = 1
        while idx < self.L_max - 1:
            h_temp = max(self.H_task[id][idx] for id, l in enumerate(self.L_task))
            h[idx] = h_temp
            idx = idx + 1
        h[self.L_max - 1] =  h_theta_multitask
        h[self.L_max] =  self.h_out
        return h


    @property
    def D_multitask(self):
        sum = 0
        for i in range(self.L_max):
            sum = sum + self.H_multitask[i] * self.H_multitask[i + 1] + self.H_multitask[i + 1]
        return sum


    def W_multitask(self, genotype):
        w = np.zeros(self.L_max, dtype=list)
        b = np.zeros(self.L_max, dtype=list)
        base = 0
        idx = 1
        while idx < self.L_max + 1:
            n_in = self.H_multitask[idx -  1]
            n_out = self.H_multitask[idx]
            w_idx = genotype[base:(base + n_in * n_out)].reshape(n_in, n_out, order='F')
            base = base + n_in * n_out
            b_idx = genotype[base:(base + n_out)]
            base = base + n_out
            w[idx - 1] = w_idx
            b[idx - 1] = b_idx
            idx = idx + 1

        return w, b

    def W_per_task(self, idv, sf):
        n_layer = self.L_task[sf]
        w = np.zeros(self.L_task[sf], dtype=list)
        b = np.zeros(self.L_task[sf], dtype=list)
        w_t, b_t = self.W_multitask(idv)
        idx = 0
        while idx < n_layer - 1:
            n_in = self.H_task[sf][idx]
            n_out = self.H_task[sf][idx + 1]
            w_idx = w_t[idx][:n_in, :n_out]
            b_idx = b_t[idx][:n_out]
            w_idx = w_idx * 10 - 5
            b_idx = b_idx * 10 - 5
            w[idx] = w_idx
            b[idx] = b_idx
            idx = idx + 1
        n_in = self.H_task[sf][idx]
        n_out = self.H_task[sf][idx + 1]
        # print(w_t[self.L_max-1],w_t[self.L_max-1].shape, n_in, n_out)
        w[idx] = w_t[self.L_max-1][:n_in, :n_out] * 10 - 5
        b[idx] = b_t[self.L_max-1][:n_out] * 10 - 5
        return w, b


    @property
    def dims(self):
        d = []
        for idx, l in enumerate(self.L_task):
            sum = 0
            for i in range(l):
                sum = sum + self.H_task[idx][i] * self.H_task[idx][i + 1] + self.H_task[idx][i + 1]
            d.append(sum)
        return d
        
    def evaluate(self, solution, sf):
        '''
        Params
        ------
        - solution (vector): vector of weights of ANN
        - sf (int): skill factor
        '''
        w, b = self.W_per_task(idv=solution, sf=sf)
        out = self.X

        for idx, val in enumerate(w):
            out = sigmoid(out @ val + b[idx])
        return mse(self.y, out)

if __name__ == '__main__':
    config = yaml.load(open('data/instances.yaml').read())
    instance = 'nbit_8_3_test'
    taskset = mtlTaskset(config[instance])
    # print (taskset.H_task, taskset.L_task, taskset.H_multitask, taskset.D_multitask, taskset.dims)
    # print(taskset.W_per_task())
    pass
    # config = yaml.load(open('data/instances.yaml').read())
    # instance = 'ionosphere'
    # taskset = Taskset(config[instance])

    # solution = np.random.rand(taskset.D_multitask)
    # print(taskset.evaluate(solution, 2))
