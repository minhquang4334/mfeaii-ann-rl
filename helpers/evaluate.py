import numpy as np 
from .helpers import *
from .instance import * 

def evaluate_EA():
    list_instances = get_list_instance_name()
    instance = Instance(config, 'nbit_4_3') # task 8 bit
    arr = np.asarray(instance.results_by_tasks)
    best_result = arr # task 8bit max
    # best_result = np.sort(best_result, axis = -1)

    print (best_result.shape)
    results = []
    for best in best_result:
        result = []
        for el in best:
            re = []
            for tmp in el:
                if(len(re) <= tmp[3]):
                    re.append([])
                re[int(tmp[3])].append(tmp)
            result.append(re)
        result = np.asarray(result)
        result = result[:,:,:,2]
        results.append(result)
    results = np.asarray(results)
    print (results.shape)
    convergence_train(results)

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

def mfea2_rmp():
    instance = Instance(config, 'nbit_4_3') # task 8 bit
    result_mfea2_8bit = np.asarray(instance.results_rmp(method_id=3))
    print (result_mfea2_8bit.shape)
    result_mfea2_8bit = group_result_by_index(re=result_mfea2_8bit, index=0, number_of_el=3, margin=1)
    print (result_mfea2_8bit.shape)
    results = []
    for ins in result_mfea2_8bit:
        result = group_result_by_index(ins, index=5, number_of_el=15, margin=0)
        results.append(result)
    results = np.asarray(results)[:,:,:, [2,3]]
    rmp_results = results[0, :, :, 1]
    rmp = []
    for r in rmp_results:
        tmp_results = [tmp.split(' - ') for tmp in r]        
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
    print (re[:10, :])
    final_results = (results[0, :, :, 0], results[1, :, :, 0], results[2, :, :, 0], re[:, 0, :], re[:, 1, :], re[:, 2, :])
    XRange = [[np.arange(1000)] for _ in final_results]
    convergence(final_results, ['TASK1', 'TASK2', 'TASK3', 'RMP1-2', 'RMP1-3', 'RMP2-3'], XRange)
