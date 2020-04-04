import numpy as np 
from experiment import *
from visualize import *
config = get_config('config.yaml')
conn = create_connection(config)
cur = conn.cursor()
# cur.execute('ALTER TABLE iteration ADD COLUMN rmp VARCHAR(128);')

class Instance:
    def __init__(self, config, instance_name):
        self.conn = create_connection(config)
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
        max_iter = config['num_iter'] - 1
        query = 'SELECT instance_id, method_id, best, seed from iteration where instance_id in {} And method_id in {} and num_iteration={}'.format(self.instances_id, self.methods_id, max_iter)
        cur.execute(query)
        re = cur.fetchall()
        # print (re, self.instances_id, self.methods_id)
        results = []
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
                std = round(np.std(el), 4)           
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

def evaluate_EA():
    list_instances = get_list_instance_name()
    instance = Instance(config, 'nbit_8_3') # task 8 bit
    arr = np.asarray(instance.results_by_tasks)
    best_result = arr[0,:,:,:] # task 8bit max
    # best_result = np.sort(best_result, axis = -1)

    print (best_result.shape)
    result = []
    for el in best_result:
        re = []
        for tmp in el:
            if(len(re) <= tmp[3]):
                re.append([])
            re[int(tmp[3])].append(tmp)
        result.append(re)
    result = np.asarray(result)
    result = result[:,:,:,2]
    print (result.shape)
    convergence_train(result)

def compare_mfea2_sgd(method_id):
    instance = Instance(config, 'nbit_8_3') # task 8 bit
    result_mfea2_8bit = np.asarray(instance.results_subtask(instance_id=24, method_id=3))
    result_sgd_8bit = np.asarray(instance.results_subtask(instance_id=1, method_id=method_id))
    result_mfea2_8bit = result_mfea2_8bit[result_mfea2_8bit[:, 5] <= 200000]
    result_mfea2_8bit = group_result(result_mfea2_8bit, 3)
    result_sgd_8bit = group_result(result_sgd_8bit, 3)
    result_mfea2_8bit = result_mfea2_8bit[:,:, [2,5]]
    result_sgd_8bit = result_sgd_8bit[:,:, [2,5]]
    result = (result_mfea2_8bit, result_sgd_8bit)
    XRange = [_[0][:, [1]].T for _ in result]

    convergence(result, ['MFEA2', 'SGD'], XRange)

# def find_best_index(result, index):
#     i = 0
#     max = 0
#     max_index = 0
#     for re in result:
#         if(re[index] > max):
#             max = re[index]
#             max_index = i
#         i += 1
#     return max_index

def group_result_by_index(re, index, number_of_el, margin=1): #group result by seed
    N = number_of_el
    results = [[] for _ in range(N)]
    for r in re:
        i = int(r[index]) - margin
        results[i].append(r)
    return np.asarray(results)

def export_result(instances):
    ''' print template
    CEA (4,5,6) & $0.0331 \pm 0.0091 $& $0.0166 \pm 0.0097$ & $\mathbf{0.0058 \pm 0.0012}$ \\ 
    MFEA (4,5,6) & $0.0268 \pm 0.0078$ & $\mathbf{0.0116 \pm 0.0034}$ & $0.0068 \pm 0.0016$ \\ 
    MFEAII (4,5,6)  & $\mathbf{0.0260 \pm 0.01}$ & $0.0163 \pm 0.0083$ & $0.0140 \pm 0.0106$ \\ \hline
    CEA (6,7,8) & $0.0115 \pm 0.0083$ & $0.0072 \pm 0.0064$ & $0.0026 \pm 0.0018$ \\ 
    MFEA (6,7,8) & $\mathbf{0.0082 \pm 0.0051}$ & $\mathbf{0.0029 \pm 0.0012}$ & $\mathbf{0.0012 \pm 0.0009}$ \\ 
    MFEAII (6,7,8)  & $0.0091 \pm 0.0066$ & $0.0038 \pm 0.0024$ & $0.0013 \pm 0.001$ \\ \hline
    '''
    K = 3
    results = []
    texts = ''
    for instance in instances:
        result = group_result_by_index(instance, 1, K)
        results.append(result)

    for re in results:
        item_template = '& ${} \\pm {}$ '
        item_bold_template = ' & $\\mathbf {} \\pm {}$ '
        for tmp in re:
            index = 0
            text = ''
            for item in tmp:
                text += item_template.format(item[2], item[3])
                index += 1
            text += '\\\\ \n'
            texts += text
            print (text)
    return text

def result_to_string():
    list_instances = get_list_instance_name()
    results = []
    for ins in list_instances:
        instance = Instance(config, ins)
        result = instance.best_results()
        results.append(result)
        print(ins, result)
    export_result(results)


def mfea2_rmp():
    instance = Instance(config, 'nbit_8_3') # task 8 bit
    result_mfea2_8bit = np.asarray(instance.results_rmp(method_id=3))
    print (result_mfea2_8bit.shape)
    result_mfea2_8bit = group_result_by_index(re=result_mfea2_8bit, index=0, number_of_el=3, margin=22)
    print (result_mfea2_8bit.shape)
    results = []
    for ins in result_mfea2_8bit:
        result = group_result_by_index(ins, index=5, number_of_el=config['repeat'], margin=0)
        results.append(result)
    results = np.asarray(results)[:,:,:, [2,3]]
    rmp_results = results[0, :, :, 1]
    rmp = []
    for r in rmp_results:
        tmp_results = [tmp.split('-') for tmp in r]        
        rmp.append(tmp_results)
    
    re = []
    for r in rmp: 
        tasks = [[] for _ in range(3)]
        for tmp in r:
            tasks[0].append(float(tmp[0]))
            tasks[1].append(float(tmp[1]))
            tasks[2].append(float(tmp[2]))
        re.append(tasks)
    re = np.asarray(re)
    print (results.shape, re.shape)
    final_results = (results[0, :, :, 0], results[1, :, :, 0], results[2, :, :, 0], re[:, 0, :], re[:, 1, :], re[:, 2, :])
    XRange = [[np.arange(1000)] for _ in final_results]
    convergence(final_results, ['TASK1', 'TASK2', 'TASK3', 'RMP1', 'RMP2', 'RMP3'], XRange)

    # result_mfea2_8bit = group_result(result_mfea2_8bit, 3)
    # result_mfea2_8bit = result_mfea2_8bit[:,:, [1, 3]]
    # mfea2_ = result_mfea2_8bit[:,:, [0]]
    # rmp_ = result_mfea2_8bit[:,:, [1]]
    # result = (mfea2_, rmp_)
    # XRange = [[np.arange(1000)] for _ in result]
    # print (result)
    # convergence(result, ['MFEA2', 'RMP'], XRange)


if __name__ == '__main__':
    compare_mfea2_sgd(5)