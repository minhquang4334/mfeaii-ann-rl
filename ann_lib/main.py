import yaml
import numpy as np
from input_handler import mapping

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def mse(y, y_pred):
    return 0.5 * np.mean(np.power(y - y_pred, 2))

class Taskset:
    '''Including:
        - a dataset
        - three ANN configuration
    '''
    def __init__(self, config):
        self.config = config
        self.X, self.y = mapping[self.config['name']](self.config['path'])

    @property
    def D_multitask(self):
        num_input = self.config['input']
        num_hidden_max = max(self.config['hiddens'])
        return (num_input + 1) * num_hidden_max + num_hidden_max + 1

    def decode(self, solution, sf):
        num_input = self.config['input']
        num_hidden = self.config['hiddens'][sf]
        num_hidden_max = max(self.config['hiddens'])
        assert len(solution) == self.D_multitask
        start = 0
        end = start + num_input * num_hidden_max
        w1 = solution[start:end].reshape(num_input, num_hidden_max)[:, :num_hidden]

        start = end
        end = start + num_hidden_max
        b1 = solution[start:end][:num_hidden]

        start = end
        end = start + num_hidden_max
        w2 = solution[start:end].reshape(num_hidden_max, 1)[:num_hidden, :]

        start = end
        end = start + 1
        b2 = solution[start:end]
        return w1, b1, w2, b2

    def evaluate(self, solution, sf):
        '''
        Params
        ------
        - solution (vector): vector of weights of ANN
        - sf (int): skill factor
        '''
        w1, b1, w2, b2 = self.decode(solution, sf)
        out = sigmoid(self.X @ w1 + b1)
        out = sigmoid(out @ w2 + b2)
        return mse(self.y, out)

if __name__ == '__main__':
    config = yaml.load(open('data/instances.yaml').read())
    instance = 'ionosphere'
    taskset = Taskset(config[instance])

    solution = np.random.rand(taskset.D_multitask)
    print(taskset.evaluate(solution, 2))
