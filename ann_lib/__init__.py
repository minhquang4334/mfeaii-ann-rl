import os
import yaml
from .taskset import Taskset
from .mtl_taskset import mtlTaskset
from .same_topo_taskset import SameTopoTaskset
from .input_handler import *
BASEDIR = os.path.dirname(__file__)

def create_taskset(instance):
    config = yaml.load(open(os.path.join(BASEDIR, 'data/instances.yaml')).read())
    taskset = Taskset(config[instance])
    return taskset

def create_general_taskset(instance):
    config = yaml.load(open(os.path.join(BASEDIR, 'data/mtl_instances.yaml')).read())
    taskset = mtlTaskset(config[instance])
    return taskset

def create_same_topo_taskset(instance):
    config = yaml.load(open(os.path.join(BASEDIR, 'data/same_topo_instance.yaml')).read())
    taskset = SameTopoTaskset(config[instance])
    return taskset