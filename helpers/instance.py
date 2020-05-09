import numpy as np 
from experiment import *
from .visualize import *
config = get_config('./config.yaml')
config_db = config['database']
config_ea = config['ea']
conn = create_connection(config['database'])
cur = conn.cursor()
# cur.execute('ALTER TABLE iteration ADD COLUMN rmp VARCHAR(128);')

class Instance:
    def __init__(self, config, instance_name):
        self.conn = create_connection(config_db)
        self.cur = conn.cursor()
        self.instance_name = instance_name 
        
    @property
    def instances_id(self):
        query = 'SELECT id from instance where `name` = "{}"'.format(self.instance_name)
        cur.execute(query)
        re = cur.fetchall()
        results = []
        for r in re:
            results.append(r[0])
        return tuple(results)

    @property 
    def methods_id(self):
        cur.execute('SELECT id from method')
        re = cur.fetchall()
        results = []
        for r in re:
            results.append(r[0])
        return tuple(results)

    def best_results(self):
        max_iter = config_ea['num_iter'] - 1
        query = 'SELECT instance_id, method_id, best, seed from iteration where instance_id in {} And method_id in {} and num_iteration={} order by instance_id'.format(self.instances_id, self.methods_id, max_iter)
        cur.execute(query)
        re = cur.fetchall()
        # print (re, self.instances_id, self.methods_id)
        results = []
        print(self.instances_id, self.methods_id)
        for idx in self.instances_id:
            for idy in self.methods_id:
                total = 0
                number_idx = 0
                el = []
                for tmp in re:
                    if (tmp[0] == idx) & (tmp[1] == idy):
                        total += tmp[2]
                        number_idx += 1
                        el.append(tmp[2])
                el = np.asarray(el)
                std = round(np.std(el), 6)           
                avg = round(total/number_idx, 4)
                results.append([idx, idy, avg, std])
                
        return results

    @property
    def results_by_tasks(self):
        results = []
        for idx in self.instances_id:
            re = []
            for idy in self.methods_id:
                re.append(self.results_subtask(idx, idy))
            results.append(re)
        results = np.asarray(results)
        return results

    
    def results_subtask(self, instance_id, method_id):
        query = 'SELECT instance_id, method_id, best, seed, num_iteration, num_evaluation from iteration where instance_id = {} And method_id = {}'.format(instance_id, method_id)
        cur.execute(query)
        re = cur.fetchall()
        # print (re, self.instances_id, self.methods_id)
        return re
    
    def results_rmp(self, method_id):
        query = 'SELECT instance_id, method_id, best, rmp, num_iteration, seed from iteration where instance_id in {} And method_id = {}'.format(self.instances_id, method_id)
        cur.execute(query)
        re = cur.fetchall()
        # print (re, self.instances_id, self.methods_id)
                
        return re

if __name__ == '__main__':
    pass