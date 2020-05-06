import numpy as np 
from mfea_ii_lib import *
from experiment import *
config = get_config('./config.yaml')
conn = create_connection(config['database'])
cur = conn.cursor()
# cur.execute('ALTER TABLE iteration ADD COLUMN rmp VARCHAR(128);')

def get_method_from_name(method_name):
    cur.execute('SELECT * from method where `name` = "{}"'.format(method_name))
    re = cur.fetchall()
    results = []
    for r in re:
        results.append(r[0])
    return results[0]

def get_list_instance_name():
    cur.execute('SELECT DISTINCT `name` from instance')
    re = cur.fetchall()
    results = []
    for r in re:
        results.append(r[0])
    return tuple(results)
    pass

def group_result(re, index_of_seed): #group result by seed
    n_seed = config['repeat']
    results = [[] for _ in range(n_seed)]
    for r in re:
        index = int(r[index_of_seed])
        results[index].append(r)
    return np.asarray(results)

def group_result_by_index(re, index, number_of_el, margin=1): #group result by seed
    N = number_of_el
    results = [[] for _ in range(N)]
    for r in re:
        i = int(r[index]) - margin
        results[i].append(r)
    return np.asarray(results)


if __name__ == '__main__':
    print(get_method_from_name('sgd'))