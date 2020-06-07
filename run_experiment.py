from ann_lib import *
from mfea_ii_lib import *
from cea import cea
from mfea import mfea
from mfeaii import mfeaii
from experiment import *
from fnn import *
from helpers import *

# instances = get_config('ann_lib/data/instances.yaml')
# instances = get_config('ann_lib/data/same_topo_instance.yaml')
instances = get_config('ann_lib/data/mtl_instances.yaml')

methods = {'cea': cea, 'mfea': mfea, 'mfeaii': mfeaii}
sgd_method = {'sgd': ''}

config = get_config('config.yaml')
db = config['database']
conn = create_connection(db)

def mfea_ann():
    local_config = config['ea']
    for seed in range(local_config['repeat']):
        for instance in instances:
            print(instance)
            taskset = create_general_taskset(instance)
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
    results = []
    for seed in range(local_config['repeat']):
        for t in tasks_config:
            task = tasks_config[t]
            print('----------{}------------'.format(task['name']))
            tasks = create_tasks(task)
            for method in methods:
                results = []
                methods[method](tasks, local_config, callback=results.append, problem="mfea-rl")
                method_id = get_method_id(conn, db, name=method)
                for k in range(task['n_task']):
                    instance_id = get_instance_id(conn, db, task['name'], '{}__{}'.format(param(task['init'], task['alpha'], k), task['unit']))
                    for result in results:
                        # print(method_id, instance_id, result[k].nit, result[k].fun)
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

if __name__ == "__main__":
    mfea_rl()
    