import os
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
BASEDIR = os.path.dirname(__file__)

def ionosphere(link):
    with open(os.path.join(BASEDIR, link), "r") as f:
        reader = csv.reader(f, delimiter=',')

        in_tmp = []
        out_tmp = []

        data_list = list(reader)
        for row in data_list:
            tmp = row[:-1:]
            in_tmp.append(tmp)
            if(row[-1] == 'g'):
                out_tmp.append(0)
            else:
                out_tmp.append(1)
        in_arr = np.array(in_tmp, dtype=float)
        out_arr = np.array(out_tmp)
        out_arr = out_arr.reshape((len(out_arr), 1))
        return in_arr, out_arr.astype(np.float64)

def ticTacToe(link):
        with open(link, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            #next(reader, None)

            input_tmp = []
            output_tmp = []
            data_list = list(reader)
            #print(data_list)
            np.random.shuffle(data_list)
            for row in data_list:
                tmp = []
                for i in range(0, len(row) - 1):
                    if(row[i] == 'x'):
                        tmp.append(1)
                    elif(row[i] == 'o'):
                        tmp.append(0)
                    else:
                        tmp.append(0)
                input_tmp.append(tmp)
                if(row[-1] == 'positive'):
                    output_tmp.append(1)
                else:
                    output_tmp.append(0)

            input_arr = np.array(input_tmp)
            output_arr = np.array(output_tmp)
            output_arr = output_arr.reshape((len(output_arr), 1))
            #print(input_arr)
            #print(output_arr)
            return input_arr, output_arr.astype(np.float64)
        pass

def creditScreening(link):
        df = pd.read_csv(link, header=None, index_col=None)

        min_max = preprocessing.MinMaxScaler()
        labelEncoder = preprocessing.LabelEncoder()
        delete_idx = []
        try:
            for ridx in df.index.values:
                #print(df.iloc[ridx])
                for field in df.iloc[ridx]:
                    #print(field)
                    if field == '?':
                        delete_idx.append(ridx)
                        break
            for i in reversed(delete_idx):
                df.drop(i, inplace=True)
            pass
            #print(df)
        except ValueError as e:
            #print(ridx)
            pass
        # except IndexError as e_i:
        #     print(ridx)
        #     pass
        #df = df.convert_objects(convert_numeric=True)
        for col in df.columns.values:
            #print(df[col].dtypes)
            if df[col].dtypes == 'object':
                #print(col)
                data = df[col].append(df[col])
                labelEncoder.fit(data.values)
                df[col] = labelEncoder.transform(df[col])
        
    
        data_minmax = np.array(min_max.fit_transform(df))
        #print(data_minmax)
        dlen = len(data_minmax[0])
        in_arr = np.array(data_minmax[::1, 0: dlen - 2]).T
        out_arr = np.array(data_minmax[::1, -1::]).T
        #out_arr = out_arr.reshape((len(data_minmax), 1))
        #print(in_arr)
        #print(out_arr.shape)
        return in_arr.T, out_arr.T.astype(np.float64)
        pass

def breastCancer(link):
        with open(link, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            in_tmp = []
            out_tmp = []

            data_list = list(reader)
            for row in data_list:
                for field in row:
                    if(field == '?'):
                        data_list.remove(row)
            
           # data_list = [map(int, i) for i in data_list]
            np_data_list = np.array(data_list, dtype=float)
            in_tmp = np_data_list[::1, 1:-1:]
            out_tmp = np_data_list[::1, -1::]
            #print(in_tmp)
            #print(out_tmp)

            minmax = preprocessing.MinMaxScaler()
            in_minmax = minmax.fit_transform(in_tmp).T
            out_minmax = minmax.fit_transform(out_tmp).T
            #print(in_minmax)
            #print(out_minmax)
            return in_minmax.T, out_minmax.astype(np.float64).T
        pass

def generateNbitDataSet(N):
    dim = [int(2 ** int(N)), int(N)] # 2^N x N
    sample = np.random.rand(dim[0], dim[1]) # 2N x N
    X = np.zeros(dim, dtype=bool) # 2N x N
    Y = np.empty([1, dim[0]], dtype=bool) # 1 x 2N

    for i in range(0, len(X)):
        increment(X[i], i + 1)
        # print(X[i].astype(float))

    # X = X.T

    np.random.shuffle(X)
    X = X.T # N x 2N
    for j in range(0, dim[0]):
        count = np.count_nonzero(X[::1, j])
        # print(count)
        if count % 2 == 0:
            Y[0][j] = 0 # 0
        else:
            Y[0][j] = 1 # 1
    # print('X')
    # print(X.astype(float))
    # print('Y')
    # print(Y.astype(float))
    X, Y = X.T, Y.T
    return X.astype(int), Y.astype(int)
    pass

def generateNbitOddDataSet(N):
    dim = [int(2 ** int(N)), int(N)] # 2^N x N
    sample = np.random.rand(dim[0], dim[1]) # 2N x N
    X = np.zeros(dim, dtype=bool) # 2N x N
    Y = np.empty([1, dim[0]], dtype=bool) # 1 x 2N

    for i in range(0, len(X)):
        increment(X[i], i + 1)
        # print(X[i].astype(float))

    # X = X.T

    np.random.shuffle(X)
    X = X.T # N x 2N
    for j in range(0, dim[0]):
        count = np.count_nonzero(X[::1, j])
        # print(count)
        if count % 2 == 0:
            Y[0][j] = 1 # 0
        else:
            Y[0][j] = 0 # 1
    # print('X')
    # print(X.astype(float))
    # print('Y')
    # print(Y.astype(float))
    X, Y = X.T, Y.T
    return X.astype(int), Y.astype(int)
    pass

def increment(X, times):
    cur_times = 0
    while cur_times != times:
        i = 0
        while i < len(X) and X[i] == 1:
            X[i] = 0
            i = i + 1
        if i < len(X):
            X[i] = 1
        cur_times += 1

mapping = {'ionosphere':ionosphere, 'ticTacToe': ticTacToe, 'creditScreening' : creditScreening, 
'breastCancer' : breastCancer, 'generateNbitDataSet' : generateNbitDataSet, 'generateNbitOddDataSet': generateNbitOddDataSet}

if __name__ == '__main__':
    input, out  = generateNbitDataSet(8)
