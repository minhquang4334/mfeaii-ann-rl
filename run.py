from ann_lib import create_taskset
from cea import *
from mfea import *

config = load_config()
taskset = create_taskset('ionosphere')
cea(taskset, config)
mfea(taskset, config)
