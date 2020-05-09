from ann_lib import *
from mfea_ii_lib import *
from cea import cea
from mfea import mfea
from mfeaii import mfeaii
from experiment import *
from fnn import *
from helpers import *

# instances = get_config('ann_lib/data/instances.yaml')
#instances = get_config('ann_lib/data/same_topo_instance.yaml')
instances = get_config('ann_lib/data/mtl_instances.yaml')

methods = {'cea': cea, 'mfea': mfea, 'mfeaii': mfeaii}
sgd_method = {'sgd': ''}
# methods = {'mfeaii': mfeaii}

config = get_config('config.yaml')
db = config['database']
conn = create_connection(db)

def mfea_ann():
    local_config = config['ea']
    for seed in range(local_config['repeat']):
        for instance in instances:
            print(instance)
            taskset = create_general_taskset(instance)
            print (taskset.H_task, taskset.L_task, taskset.H_multitask, taskset.D_multitask, taskset.dims)
            for method in methods:
                results = []
                methods[method](taskset, local_config, callback=results.append)
                # Logging the result to database
                method_id = get_method_id(conn, db, name=method)
                instance_data = instances[instance]
                for k, hidden in enumerate(instance_data['hiddens']):
                    instance_id = get_instance_id(conn, db, instance, '{}hidden'.format(' '.join('{}-'.format(h) for h in [hidden])))
                    for result in results:
                        kwargs = {'method_id': method_id,
                                'instance_id': instance_id,
                                'best': result[k].fun,
                                'rmp': result[k].message['rmp'],
                                'best_solution': serialize(result[k].x),
                                'num_iteration': result[k].nit,
                                'num_evaluation': result[k].nfev,
                                'seed': seed,
                                }
                        add_iteration(conn, db, **kwargs)

from rl import *
def mfea_rl():
    local_config = config['rl']
    tasks_config = local_config['tasks']
    # tasks = [CartPole(0.8 + i * 10) for i in range(10)]
    # tasks = [Acrobot(1.0 + 0.1 * i) for i in range(5)]
    # tasks = create_tasks(tasks_config['CartPole'])
    results = []
    for seed in range(local_config['repeat']):
        for task in tasks_config:
            print('----------{}------------'.format(tasks_config[task]['name']))
            tasks = create_tasks(tasks_config[task])
            for method in methods:
                methods[method](tasks, local_config, callback=results.append, problem="mfea-rl")
                method_id = get_method_id(conn, db, name=method)
                

    # print(results[4][0], np.asarray(results).shape)
    # observation = np.array(tasks[0].env.state)
    # for _ in range(1000):
    #     tasks[0].env.render()
    #     action = tasks[0].action(observation, results[4][0].x)
    #     tasks[0].env.step(action)
    # tasks[0].env.close()

if __name__ == "__main__":
    mfea_ann()
    