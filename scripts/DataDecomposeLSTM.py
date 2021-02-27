import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
import pandas as pd
import matplotlib.pyplot as plt

N_STEPS = 3
N_FEATURES = 1


def splitSequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def dealData(tableName,columnName):
    date_csv = pd.read_csv(tableName)
    dataset = date_csv[columnName].values
    dataset = dataset.astype('float32')
    max_value = np.max(dataset)  # 获得最大值
    min_value = np.min(dataset)
    scalar = max_value - min_value  # 获得间隔数量
    dataset = list(map(lambda x: (x - min_value) / scalar, dataset))
    X_all, y_all = splitSequence(dataset, N_STEPS)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], N_FEATURES))
    train_size = 32
    test_size = len(X_all) - train_size
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_test = X_all[train_size:]
    y_test = y_all[train_size:]
    return X_train, y_train, X_test, y_test, scalar, min_value

def trainAndPred(X_train, y_train, X_test):
    modelDouble = Sequential()
    modelDouble.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(N_STEPS, N_FEATURES)))
    modelDouble.add(Dense(1))
    modelDouble.compile(optimizer='adam', loss='mse')
    modelDouble.fit(X_train, y_train, epochs=200, verbose=0)
    x_input = X_test
    x_input = x_input.reshape((3, N_STEPS, N_FEATURES))
    yhat = modelDouble.predict(x_input, verbose=0)
    return yhat


def returnNormal(scalar, minValue, y_hat, y_test):
    pred_y = list(map(lambda x: x * scalar + minValue, y_hat))
    real_y = list(map(lambda x: x * scalar + minValue, y_test))
    return pred_y, real_y


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, scalar, minValue = dealData('Yingkou_Ningbo_deals.csv', 'deals')
    yhatDouble = trainAndPred(X_train, y_train, X_test)
    pred_y, real_y = returnNormal(scalar, minValue, yhatDouble, y_test)
    print('this is pred_y', pred_y)
    print('this is real_y', real_y)
