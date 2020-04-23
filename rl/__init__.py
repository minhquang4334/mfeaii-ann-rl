from .tasks import *

task_lists = {'CartPole': CartPole, 'Acrobot': Acrobot, 'FlappyBird': FlappyBird}
# task: {name, init, alpha, n_task} eq: param = init + alpha * i
def param(init, alpha, i): 
    return init + alpha * i

def create_tasks(task):
    tasks = []
    for i in range(task['n_task']):
        t = task_lists[task['name']](param(task['init'], task['alpha'], i))
        tasks.append(t)
    return tasks
    