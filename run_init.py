from ann_lib import create_taskset
from cea import *
from mfea import *
from mfeaii import *
from experiment import *

instances = get_config('ann_lib/data/instances.yaml')
methods = {'cea': cea, 'mfea': mfea, 'mfeaii': mfeaii}

config = get_config('config.yaml')
conn = create_connection(config)
drop_experiment(conn, config)
create_experiment(conn, config)

alter_method(conn, config)
for instance in instances:
    instance_data = instances[instance]
    for hidden in instance_data['hiddens']:
        add_instance(conn, config, instance, '{}-hidden'.format(hidden))

for method in methods:
    add_method(conn, config, name=method)



