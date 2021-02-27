import numpy as np
from numpy import array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

N_STEPS = 3
N_FEATURES = 1
PRED_DAYS = 3


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

def readData(tableName,columnName):
    date_csv = pd.read_csv(tableName)
    dataset = date_csv[columnName].values
    dataset = dataset.astype('float32')
    maxValue = np.max(dataset)  # 获得最大值
    minValue = np.min(dataset)
    scalar = maxValue - minValue  # 获得间隔数量
    #dataset = list(map(lambda x: (x - minValue) / scalar, dataset))
    return dataset

def normalizeData(dataset):
    maxValue = np.max(dataset)  # 获得最大值
    minValue = np.min(dataset)
    scalar = maxValue - minValue  # 获得间隔数量
    dataset = list(map(lambda x: (x - minValue) / scalar, dataset))
    return dataset, scalar, minValue

def getTrainAndTest(dataset):
    X_all, y_all = splitSequence(dataset, N_STEPS)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], N_FEATURES))
    train_size = 36
    test_size = len(X_all) - train_size
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_test = X_all[train_size:]
    y_test = y_all[train_size:]
    return X_train, y_train, X_test, y_test

def trainAndPred(X_train, y_train, X_test):
    modelDouble = Sequential()
    modelDouble.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(N_STEPS, N_FEATURES)))
    modelDouble.add(Dense(1))
    modelDouble.compile(optimizer='adam', loss='mse')
    callback = EarlyStopping(monitor='loss', patience=10)
    modelDouble.fit(X_train, y_train, epochs=1500, verbose=0, callbacks=[callback], batch_size=5)
    x_input = X_test
    x_input = x_input.reshape((PRED_DAYS, N_STEPS, N_FEATURES))
    yhat = modelDouble.predict(x_input, verbose=0)
    return yhat

def trainAndPred_old(X_train, y_train, X_test):
    modelDouble = Sequential()
    modelDouble.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(N_STEPS, N_FEATURES)))
    modelDouble.add(Dense(1))
    modelDouble.compile(optimizer='adam', loss='mse')
    #callback = EarlyStopping(monitor='loss', patience=10)
    modelDouble.fit(X_train, y_train, epochs=1500, verbose=0, batch_size=5)
    x_input = X_test
    x_input = x_input.reshape((PRED_DAYS, N_STEPS, N_FEATURES))
    yhat = modelDouble.predict(x_input, verbose=0)
    return yhat

def returnNormal(scalar, minValue, y_hat, y_test):
    pred_y = list(map(lambda x: x * scalar + minValue, y_hat))
    real_y = list(map(lambda x: x * scalar + minValue, y_test))
    return pred_y, real_y

def emdLSTM(dataset):
    emd = EMD()
    IMFs = emd.emd(dataset)
    print(type(IMFs))
    [rows, columns] = IMFs.shape
    yhatResult = 0
    for n, imf in enumerate(IMFs):
        tempDataSet = imf
        #print('--------------------------------------')
        #myDataSet, myScalar, myMinValue = normalizeData(tempDataSet)
        X_train, y_train, X_test, y_test = getTrainAndTest(tempDataSet)
        yhat = trainAndPred(X_train, y_train, X_test)
        #pred_y, _ = returnNormal(myScalar, myMinValue, yhat, y_test)
        #print(yhat)
        yhatResult = yhat + yhatResult
    print(yhatResult)
    return yhatResult

def metrics(test,predict):
    # MSE均方误差,越小越好
    mse = mean_squared_error(test, predict)
    print("MSE=", mse)

    # MAE数值越小越好，可以通过对比判断好坏
    mae = mean_absolute_error(test, predict)
    print("MAE=", mae)

    # R平方值，越接近1越好
    r2 = r2_score(test, predict)
    print("R_square=", r2)

if __name__ == '__main__':
    dataset = readData('../data/1.csv', 'size')
    _, _, _, y_test = getTrainAndTest(dataset)
    yhat = emdLSTM(dataset)
    metrics(y_test, yhat)
    plt.title('Qinzhou_Ningbo_deals_decompose_array')
    plt.plot(yhat, 'r', label='prediction')
    plt.plot(y_test, 'b', label='real')
    plt.legend(loc='best')
    plt.savefig('Qinzhou_Ningbo_deals_decompose_array.png')
    plt.show()
    '''
    X_train, y_train, X_test, y_test = getTrainAndTest(dataset)
    yhat = trainAndPred(X_train, y_train, X_test)
    yhat_old = trainAndPred_old(X_train, y_train, X_test)
    print('this is no call_back')
    metrics(y_test, yhat_old)
    plt.title('Qinzhou_Ningbo_deals_no_callback')
    plt.plot(yhat_old, 'r', label='prediction')
    plt.plot(y_test, 'b', label='real')
    plt.legend(loc='best')
    plt.savefig('Qinzhou_Ningbo_deals_no_callback.png')
    plt.show()
    print('this is with call_back')
    metrics(y_test, yhat)
    plt.title('Qinzhou_Ningbo_deals_with_callback')
    plt.plot(yhat, 'r', label='prediction')
    plt.plot(y_test, 'b', label='real')
    plt.legend(loc='best')
    plt.savefig('Qinzhou_Ningbo_deals_with_callback.png')
    plt.show()
    '''