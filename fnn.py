# Rohitash Chandra, 2017 c.rohitash@gmail.conm

#!/usr/bin/python

# ref: http://iamtrask.github.io/2015/07/12/basic-python-network/  
 

#Sigmoid units used in hidden and output layer. gradient descent and stocastic gradient descent functions implemented with momentum. Note:
#  Classical momentum:

#vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) )
#W(t+1) = W(t) + vW(t+1)

#W Nesterov momentum is this: http://cs231n.github.io/neural-networks-3/

#vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) + momentum.*vW(t) )
#W(t+1) = W(t) + vW(t+1)

#http://jmlr.org/proceedings/papers/v28/sutskever13.pdf

#https://github.com/deepsemantic/Neural-Network-in-Python/blob/master/Neural_Net.py

# Numpy used: http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays
 

 

import matplotlib.pyplot as plt
import numpy as np 
import random
import time
from scipy.optimize import OptimizeResult

#An example of a class
class Network:

    def __init__(self, Topo, Train, Test, MaxTime, Samples, MinPer): 
        self.Top  = Topo  # NN topology [input, hidden, output]
        self.Max = MaxTime # max epocs
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Samples

        self.lrate  = 0 # will be updated later with BP call

        self.momenRate = 0
        self.useNesterovMomen = 0 #use nestmomentum 1, not use is 0

        self.minPerf = MinPer
        #initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
        np.random.seed()
        self.W1 = np.random.randn(self.Top[0]  , self.Top[1])  / np.sqrt(self.Top[0] )
        self.B1 = np.random.randn(1  , self.Top[1])  / np.sqrt(self.Top[1] ) # bias first layer
        self.BestB1 = self.B1
        self.BestW1 = self.W1
        self.W2 = np.random.randn(self.Top[1] , self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1  , self.Top[2])  / np.sqrt(self.Top[1] ) # bias second layer
        self.BestB2 = self.B2
        self.BestW2 = self.W2
        self.hidout = np.zeros((1, self.Top[1])) # output of first hidden layer
        self.out = np.zeros((1, self.Top[2])) #  output last layer

  
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self,actualout):
        error = np.subtract(self.out, actualout)
        sqerror= np.sum(np.square(error))/self.Top[2] 
        #print sqerror
        return sqerror
  
    def ForwardPass(self, X ):
        nesterov = self.W1 + self.momenRate*self.V1 if self.useNesterovMomen else 0 #Nesterov lookahead
        z1 = X.dot(self.W1) - self.B1  
        self.hidout = self.sigmoid(z1) # output of first hidden layer  
        nesterov = self.W2 + self.momenRate*self.V2 if self.useNesterovMomen else 0 #Nesterov lookahead
        z2 = self.hidout.dot(self.W2)  - self.B2 
        self.out = self.sigmoid(z2)  # output second hidden layer 
        
    def BackwardPassMomentum(self, Input, desired, vanilla):   
        
        old_w2 = self.W2
        old_w1 = self.W1
        out_delta = (desired - self.out)*(self.out*(1-self.out))  
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout))
              
        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

        if not vanilla: #momentum (both classic or Nesterov)
            self.W2 += self.momenRate * self.V2
            self.V2 = self.W2 - old_w2
            self.W1 += self.momenRate * self.V1
            self.V1 = self.W1 - old_w1

           

    def TestNetwork(self, Data, testSize, erTolerance):
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2])) 
        nOutput = np.zeros((1, self.Top[2]))
        clasPerf = 0
        sse = 0  
        self.W1 = self.BestW1
        self.W2 = self.BestW2 #load best knowledge
        self.B1 = self.BestB1
        self.B2 = self.BestB2 #load best knowledge
     
        for s in range(0, testSize):    
            Input[:]  =   Data[s,0:self.Top[0]] 
            Desired[:] =  Data[s,self.Top[0]:] 
            
            self.ForwardPass(Input ) 
            sse = sse+ self.sampleEr(Desired)
            if(np.isclose(self.out, Desired, atol=erTolerance).any()):
                clasPerf =  clasPerf + 1
   	    
        return (sse/testSize, float(clasPerf)/testSize * 100)

    
    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        self.BestB1 = self.B1
        self.BestB2 = self.B2  
        return [self.BestW1, self.BestW2, self.BestB1, self.BestB2]
    
    def saveResult(self, best_solution, mse, msg, num_iters, num_samples):
        result = OptimizeResult()
        result.x = best_solution
        result.fun = mse
        result.message = msg
        result.nit = num_iters
        result.nfev = (num_iters + 1) * num_samples
        return result

    def BP_GD(self, learnRate, mRate,  useNestmomen , stocastic, vanilla): # BP with SGD (Stocastic BP)
        self.lrate = learnRate
        self.momenRate = mRate
        self.useNesterovMomen =  useNestmomen  
        self.V1 = np.zeros((self.Top[0], self.Top[1]))
        self.V2 = np.zeros((self.Top[0], self.Top[2]))
     
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2])) 
        Er = []#np.zeros((1, self.Max)) 
        epoch = 0
        bestmse = 100
        bestTrain = 0
        results = []
        # while  epoch < self.Max and bestTrain < self.minPerf :
        num_eval = 0
        while  epoch < self.Max and num_eval < self.minPerf :
            sse = 0
            for s in range(0, self.NumSamples):
                 
                if(stocastic):
                   pat = random.randint(0, self.NumSamples-1) 
                else:
                   pat = s 

                Input[:]  =  self.TrainData[pat,0:self.Top[0]]  
                Desired[:] = self.TrainData[pat,self.Top[0]:]  

                self.ForwardPass(Input )
                self.BackwardPassMomentum(Input , Desired, vanilla)
                sse = sse+ self.sampleEr(Desired)
             
            # mse = np.sqrt(sse/self.NumSamples*self.Top[2]) # root mean square error
            # sse/self.NumSamples error tren 1 sample * 0.5 -> lay mse tren 1 sample
            mse = 0.5 * sse/self.NumSamples 
            if mse < bestmse:
               bestmse = mse
               self.saveKnowledge() 
               (x,bestTrain) = self.TestNetwork(self.TrainData, self.NumSamples, 0.2)
            
            num_eval = self.NumSamples * (epoch + 1)
            result = self.saveResult(best_solution=self.saveKnowledge(), mse=mse, msg=0, num_iters=epoch, num_samples=self.NumSamples)
            results.append(result)
            Er = np.append(Er, mse)
            epoch=epoch+1  

        return (Er,bestmse, bestTrain, epoch, results) 



def normalisedata(data, inputsize, outsize): # normalise the data between [0,1]
    traindt = data[:,np.array(range(0,inputsize))]	
    dt = np.amax(traindt, axis=0)
    tds = abs(traindt/dt) 
    return np.concatenate(( tds[:,range(0,inputsize)], data[:,range(inputsize,inputsize+outsize)]), axis=1)      

from ann_lib import *
from mfea_ii_lib import *
from experiment import *
from fnn import *
from helpers import *
from sklearn.model_selection import train_test_split

sgd_method = {'sgd': ''}
# methods = {'mfeaii': mfeaii}

config = get_config('config.yaml')
db = config['database']
conn = create_connection(db)

def run_sgd(instance, hidden=10, is_n_bit=False):
    # get config
    print(instance)
    sgd_method_id = get_method_from_name('sgd')
    local_config = config['sgd']

    # Create Task Set
    taskset = create_taskset(instance)    
    if(is_n_bit):
        taskset.X, taskset.y = taskset.X[0:64, :], taskset.y[0:64, :] # if n-bit problem
    print (taskset.X.shape, taskset.y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
    taskset.X, taskset.y, test_size=0.3, random_state=42)
    TrainData = np.concatenate((X_train, y_train), axis=1)
    TestData = np.concatenate((X_test, y_test), axis=1)
    
    # Set Network Config
    Hidden = hidden
    Input = np.asarray(X_train).shape[1]
    Output = np.asarray(y_train).shape[1]
    TrSamples =  TrainData.shape[0]
    TestSize = TestData.shape[0]
    Topo = [Input, Hidden, Output] 

    #save result    
    MaxRun = local_config['repeat']
    trainPerf, testPerf, trainMSE, testMSE, Epochs, Time = tuple(np.repeat([np.zeros(MaxRun)], 6, axis=0))

    # print (TrainData.shape, TestData.shape, Input, Output, TrSamples, TestSize, Topo)
    for run in range(0, MaxRun):
        results = {}
        print (run)
        fnnSGD = Network(Topo, TrainData, TestData, local_config['num_epoch'], TrSamples, local_config['max_eval']) # Stocastic GD
        start_time=time.time()
        (erEp,  trainMSE[run] , trainPerf[run] , Epochs[run], results) = fnnSGD.BP_GD(local_config['learning_rate'], local_config['mRate'], local_config['useNestmomen'],  local_config['useStocasticGD'], local_config['useVanilla'])   
        
        Time[run]  =time.time() - start_time
        (testMSE[run], testPerf[run]) = fnnSGD.TestNetwork(TestData, TestSize, local_config['test_dropout'])
        # Save result
        instance_id = get_instance_id(conn, db, instance, '{}hidden'.format(' '.join('{}-'.format(h) for h in [hidden])))
        for result in results:
            kwargs = {  'method_id': sgd_method_id,
                        'instance_id': instance_id,
                        'best': result.fun,
                        'rmp': result.message,
                        'best_solution': serialize(result.x),
                        'num_iteration': result.nit,
                        'num_evaluation': result.nfev,
                        'seed': run,
                        }
            add_iteration(conn, db, **kwargs)
                
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
    run_sgd(instance='ionosphere', hidden=10, is_n_bit=False)
    run_sgd(instance='ticTacToe', hidden=24, is_n_bit=False)
    run_sgd(instance='creditScreening', hidden=22, is_n_bit=False)
    run_sgd(instance='breastCancer', hidden=8, is_n_bit=False)
    
    