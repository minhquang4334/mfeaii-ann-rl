from ann_lib import create_taskset
from cea import *
from mfea import *
from mfeaii import *

config = load_config()
task = 'ionosphere'
taskset = create_taskset(task)
for ith in range(config['repeat_for_eval']):
    # np.save('./results/%s/cea_result/%d.npy' %(task, ith), cea(taskset, config))
    np.save('./results/%s/mfea_result/%d.npy' %(task, ith), mfea(taskset, config))
    np.save('./results/%s/mfeaii_result/%d.npy' %(task, ith), mfeaii(taskset, config))

