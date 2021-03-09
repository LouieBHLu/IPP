import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import array

DAYS = 7
BATCH_SIZE = 1
HIDDEN_UNITS = 128
LEARNING_RATE = 0.1
EPOCH = 500
HIDDEN_UNITS1 = 128
N_STEPS = 7
N_FEATURES = 1
TIME_STEPS = 7
# ------------- parameters for callback --------------
CALLBACK = True
MONITOR = 0.01
PATIENCE = 10
# ----------------------------------------------------

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


def getData(csvName, colName):
    dataset = pd.read_csv(csvName)
    dataset = dataset[colName].values
    dataset = dataset.astype('float32')
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    #dataset = list(map(lambda x: (x - min_value) / scalar, dataset))
    X_all, y_all = splitSequence(dataset, DAYS)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], N_FEATURES))
    train_size = 265
    test_size = len(X_all) - train_size
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_test = X_all[train_size:]
    y_test = y_all[train_size:]
    return X_train, y_train, X_test, y_test


def doLSTM(X_train, y_train, X_test, y_test):
    # graph object
    graph = tf.Graph()

    train_length = X_train.shape[0]
    test_length = X_test.shape[0]
    with graph.as_default():
        inputs = tf.placeholder(np.float32, shape=(BATCH_SIZE, DAYS, 1))
        preds = tf.placeholder(np.float32, shape=(BATCH_SIZE, 1))
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=HIDDEN_UNITS,
            name="LSTM_CELL"
        )

        # multi_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell, lstm_cell2])
        # multi_lstm = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell])

        # 自己初始化state
        # 第一层state
        lstm_layer1_c = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS1))
        lstm_layer1_h = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS1))
        layer1_state = tf.contrib.rnn.LSTMStateTuple(c=lstm_layer1_c, h=lstm_layer1_h)
        '''
        # 第二层state   
        lstm_layer2_c = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS))
        lstm_layer2_h = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS))
        layer2_state = tf.contrib.rnn.LSTMStateTuple(c=lstm_layer2_c, h=lstm_layer2_h)
        '''
        # init_state = (layer1_state, layer2_state)
        init_state = (layer1_state)
        print(init_state)

        # 自己展开RNN计算
        outputs = list()  # 用来接收存储每步的结果
        state_list = list()
        state = init_state
        with tf.variable_scope('RNN'):
            for timestep in range(DAYS):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                # 这里的state保存了每一层 LSTM 的状态
                # (cell_output, state) = multi_lstm(inputs[:, timestep, :], state)
                (cell_output, state) = lstm_cell(inputs[:, timestep, :], state)
                outputs.append(cell_output)
                state_list.append(state)

        # h = outputs[-1]
        h = tf.layers.dense(outputs[-1], 1)

        '''
        init_state = multi_lstm.zero_state(batch_size=BATCH_SIZE, dtype = np.float32)

        output, state = tf.nn.dynamic_rnn(
        cell = multi_lstm,
        inputs = inputs,
        dtype = tf.float32,
        initial_state = init_state
        )
        h = tf.layers.dense(output[:,:,:], 1)
        '''
        # ---------------------------------define loss and optimizer-------------
        mse = tf.losses.mean_squared_error(labels=preds, predictions=h)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=mse)

        init = tf.global_variables_initializer()

        # -----------------------------define session--------------------------------

    sess = tf.Session(graph=graph)
    # with tf.Session(graph = graph) as sess:
    sess.run(init)
    callback_array = []
    earlyStop = False
    patience = 0
    train_losses = []
    test_losses = []
    for epoch in range(1, EPOCH + 1):
        if earlyStop:
            break

        for j in range(train_length):
            # X_max = np.max(X_test_label)
            # X_min = np.min(X_test_label)
            # scalar = X_max - X_min
            # X_test_label = ~(X_test_label-X_min)/scalar
            _, train_loss, LSTMtuple, _, _ = sess.run(
                fetches=(optimizer, mse, state_list, h, outputs),
                feed_dict={
                    inputs: X_train[j:j + 1],  # 用25天闭盘价预测26天开盘价
                    # preds : X_test_label.reshape(1,25,1)
                    # preds : np.array((np_price[-1]-X_min)/scalar).reshape(1,1)
                    preds: y_train[j:j + 1].reshape(1, 1)
                }
            )


        for j in range(test_length):
            # X_max = np.max(X_test_label)
            # X_min = np.min(X_test_label)
            # scalar = X_max - X_min
            # X_test_label = ~(X_test_label-X_min)/scalar
            _, test_loss, _, _, outputs = sess.run(
                fetches=(optimizer, mse, state_list, h, outputs),
                feed_dict={
                    inputs: X_test[j:j + 1],  # 用25天闭盘价预测26天开盘价
                    # preds : X_test_label.reshape(1,25,1)
                    # preds : np.array((np_price[-1]-X_min)/scalar).reshape(1,1)
                    preds: y_test[j:j + 1].reshape(1, 1)
                }
            )
        print("this is ", epoch)
        print(train_loss)
        print(test_loss)
        if CALLBACK and len(train_losses) > 0:
            if train_losses[-1] - train_loss <= MONITOR:
                patience = patience + 1
                train_losses.append(train_loss)
                if patience == PATIENCE:
                    earlyStop = True
            else:
                train_losses.append(train_loss)
                patience = 0
        if len(train_losses) == 0:
            train_losses.append(train_loss)

    CT = []
    for cts in LSTMtuple:
        CT.append(cts.c.reshape(HIDDEN_UNITS1))




if __name__ == "__main__":
    X_train, y_train, X_test, y_test = getData("0226mean-yingkou-qinzhou.csv", "size")
    doLSTM(X_train, y_train, X_test, y_test)
