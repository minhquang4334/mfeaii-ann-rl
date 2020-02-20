from ann_lib import create_taskset
from cea import *
from mfea import *
from mfeaii import *
from experiment import *

instances = get_config('ann_lib/data/instances.yaml')
methods = {'cea': cea, 'mfea': mfea, 'mfeaii': mfeaii}
methods = {'mfeaii': mfeaii}

config = get_config('config.yaml')
conn = create_connection(config)


for seed in range(config['repeat']):
    for instance in instances:
        results = []
        taskset = create_taskset(instance)
        for method in methods:
            methods[method](taskset, config, callback=results.append)
            # Logging the result to database
            method_id = get_method_id(conn, config, name=method)
            instance_data = instances[instance]
            for k, hidden in enumerate(instance_data['hiddens']):
                instance_id = get_instance_id(conn, config, instance, '{}-hidden'.format(hidden))
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
                    add_iteration(conn, config, **kwargs)

