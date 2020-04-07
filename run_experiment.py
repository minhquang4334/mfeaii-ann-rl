from ann_lib import *
from cea import *
from mfea import *
from mfeaii import *
from experiment import *
from fnn import *
from data import *
instances = get_config('ann_lib/data/instances.yaml')
methods = {'cea': cea, 'mfea': mfea, 'mfeaii': mfeaii}
sgd_method: {'sgd': ''}
# methods = {'mfeaii': mfeaii}

config = get_config('config.yaml')
conn = create_connection(config)

def EA():
    for seed in range(config['repeat']):
        for instance in instances:
            taskset = create_taskset(instance)
            for method in methods:
                results = []
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

def SGD():
    sgd_method_id = get_method_from_name('sgd')
    local_config = config['sgd']
    N = 8
    TrainData = generateNbitDataSet(N)
    TestData = generateNbitDataSet(N)
    TrainData = np.concatenate((TrainData[0], TrainData[1]), axis=1)
    TestData = np.concatenate((TestData[0], TestData[1]), axis=1)
    # TrainData = TrainData[0:128, :]
    # TestData = TestData[0:32, :]
    Hidden = 10
    Input = N
    Output = 1
    TrSamples =  TrainData.size
    TestSize = TestData.size
    Topo = [Input, Hidden, Output] 

    #save result    
    MaxRun = local_config['repeat']
    trainPerf, testPerf, trainMSE, testMSE, Epochs, Time = tuple(np.repeat([np.zeros(MaxRun)], 6, axis=0))

    for run in range(0, MaxRun):
        results = {}
        print (run)
        fnnSGD = Network(Topo, TrainData, TestData, local_config['num_epoch'], TrSamples, local_config['max_eval']) # Stocastic GD
        start_time=time.time()
        (erEp,  trainMSE[run] , trainPerf[run] , Epochs[run], results) = fnnSGD.BP_GD(local_config['learning_rate'], local_config['mRate'], local_config['useNestmomen'],  local_config['useStocasticGD'], local_config['useVanilla'])   
        
        Time[run]  =time.time() - start_time
        (testMSE[run], testPerf[run]) = fnnSGD.TestNetwork(TestData, TestSize, local_config['testTolerance'])
        # Save result
        for result in results:
            kwargs = {  'method_id': sgd_method_id,
                        'instance_id': 1,
                        'best': result.fun,
                        'rmp': result.message,
                        'best_solution': serialize(result.x),
                        'num_iteration': result.nit,
                        'num_evaluation': result.nfev,
                        'seed': run,
                        }
            add_iteration(conn, config, **kwargs)
                
    print (trainPerf)  #[ 93.75 100.    75.   100.    93.75  93.75 100.   100.   100.   100.  ]
    print (testPerf) #[ 93.75 100.   100.   100.    93.75  93.75 100.   100.   100.   100.  ]
    print (trainMSE) # [0.01833272 0.02443385 0.01638689 0.01479283 0.0181849  0.03819253 0.01352228 0.02045447 0.03080476 0.03558831]
    print (testMSE) # [0.05162192 0.0020806  0.00169465 0.00403099 0.05158068 0.05243595 0.00091951 0.00348021 0.00418387 0.00441024]

    print (Epochs) # [3000. 2469. 3000. 2622. 3000. 3000. 2488. 1886. 1714. 1508.]
    print (Time) # [2.17364597 1.85449409 2.2716279  2.05616307 2.483567   2.94070888 2.18178201 1.59249997 1.40714502 1.19133902]
    print(np.mean(trainPerf), np.std(trainPerf)) # (95.625, 7.421463804398698)
    print(np.mean(testPerf), np.std(testPerf)) # (98.125, 2.8641098093474)
    print(np.mean(Time), np.std(Time)) # (2.015297293663025, 0.49489409397464507)
        
    # plt.figure()
    # plt.plot(erEp )
    # plt.ylabel('error')  
    # plt.savefig('out.png')

if __name__ == "__main__":
    SGD()