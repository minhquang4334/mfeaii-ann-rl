import csv
import numpy as np

def ionosphere(link):
    with open(link, "r") as f:
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

mapping = {'ionosphere':ionosphere}
