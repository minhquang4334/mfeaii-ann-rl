from ann_lib import create_taskset
from cea import *
from mfea import *
from mfeaii import *
from experiment import *

instances = get_config('ann_lib/data/instances.yaml')
methods = {'cea': cea, 'mfea': mfea, 'mfeaii': mfeaii, 'sgd': ''}

config = get_config('config.yaml')
database_config = config['database']
conn = create_connection(database_config)
drop_experiment(conn, database_config)
create_experiment(conn, database_config)

cur = conn.cursor()
cur.execute('ALTER TABLE iteration ADD COLUMN rmp VARCHAR(128);')
conn.commit()

alter_method(conn, database_config)

for instance in instances:
    instance_data = instances[instance]
    for hidden in instance_data['hiddens']:
        add_instance(conn, database_config, instance, '{}-hidden'.format(' '.join('{}-'.format(h) for h in hidden)))

from rl import *
rl_task_config = config['rl']['tasks']
for t in rl_task_config:
    task = rl_task_config[t]
    for i in range(task['n_task']):
        add_instance(conn, database_config, task['name'], '{}__{}'
        .format(param(task['init'], task['alpha'], i), task['unit']))

for method in methods:
    add_method(conn, database_config, name=method)



