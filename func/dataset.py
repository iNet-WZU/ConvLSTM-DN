import h5py#存放数据库和组
import pickle
import numpy as np
from pandas import to_datetime
import pandas as pd



class MinMaxNorm01(object):
    """scale data to range [0, 1]"""
    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()

    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x


def get_date_feature(idx):
    a = idx.weekday()
    b = idx.hour
    c = idx.minute
    d = idx.weekday() // 6
    e = idx.weekday() // 7
    return a, b, c, d, e


def traffic_loader(f, opt):
    data = f['data'][()]

    result = data.reshape((-1, 1, opt.height, opt.width))

    return result




def read_data(path, opt):
    f = h5py.File(path, 'r')
    data = traffic_loader(f, opt)

    index = f['idx'][()].astype(str)
    index = to_datetime(index)

    mmn = MinMaxNorm01()
    data_scaled = mmn.fit_transform(data)
    X, y = [], []
    X_meta = []
    predict_steps = 11
    # for each in range(len(data)-opt.close_size-predict_steps):
    #     xc_ = data_scaled[each:each+opt.close_size]
    #     yc_ = data_scaled[each+opt.close_size+predict_steps]
    #     a, b, c, d, e = get_date_feature(index[each+opt.close_size+predict_steps])
    #     X_meta.append((a, b, c, d, e))
    for i in range(opt.close_size, len(data)):
        xc_ = [data_scaled[i - c] for c in range(1, opt.close_size + 1)]
        a, b, c, d, e = get_date_feature(index[i])
        X_meta.append((a, b, c, d, e))
        if opt.close_size > 0:
            X.append(xc_)
        # y.append(yc_)
        y.append(data_scaled[i])

    X = np.asarray(X)
    X_meta = np.asarray(X_meta)
    y = np.asarray(y)

    print('X shape:' + str(X.shape))
    print('X meta shape:' + str(X_meta.shape))

    return X, X_meta, y, mmn