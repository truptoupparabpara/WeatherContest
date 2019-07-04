import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
df_init = pd.read_csv('datalab.csv')

#about X
timesteps = sequence_length = 7
data_dim = 5
data_lim = 5

#about Y and hidden size
hidden_size = 1
hidden_dim = 10
output_dim = 5


#about training
learning_rate = 0.3
epoch = 1


df_xy = df_init.iloc[:,1:]
np_xy = df_xy.to_numpy()
batch_size = np_size = len(np_xy)




def MinMaxScaler(data):
     numerator = data - np.min(data, 0)
     denominator = np.max(data, 0) - np.min(data, 0)
     # noise term prevents the zero division
     return numerator / (denominator + 1e-7)
#
#
scaled_np_xy = MinMaxScaler(np_xy)
# #scaled_np_xy = StandardScaler(np_xy)
# # train/test split
train_size = int(len(scaled_np_xy) * 0.7)
train_set = scaled_np_xy[0:train_size]
test_set = scaled_np_xy[train_size - sequence_length:]
# Index from [train_size - seq_length] to utilize past sequence


# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, :]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, sequence_length)
testX, testY = build_dataset(test_set, sequence_length)

X = tf.placeholder(tf.float32, [None, sequence_length, data_dim])
Y = tf.placeholder(tf.float32, [None, data_dim])
# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim,
    state_is_tuple=True,
    activation=tf.tanh)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, 5],
    output_dim,
    activation_fn=None)  # We use the last cell's output


# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 5])
predictions = tf.placeholder(tf.float32, [None, 5])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(500):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
        targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))


    # Plot predictions
    ax = plt.figure(figsize = [25,10])
    plt.plot(testY[:,1])
    plt.plot(test_predict[:,1], label='predict')
    plt.xlabel("Time Period")
    plt.ylabel("Americano Sail")
    ax.legend(loc = 8)
    plt.show()

    # Plot predictions
    ax2 = plt.figure(figsize=[25, 10])
    plt.plot(testY[:, 1])
    plt.plot(test_predict[:, 1], label='predict')
    plt.xlabel("Time Period")
    plt.ylabel("Caffe Latte Sail")
    ax2.legend(loc=8)
    plt.show()

    # Plot predictions
    ax3 = plt.figure(figsize=[25, 10])
    plt.plot(testY[:, 2])
    plt.plot(test_predict[:, 2], label='predict')
    plt.xlabel("Time Period")
    plt.ylabel("Caffe Moca Sail")
    ax3.legend(loc=8)
    plt.show()

    # Plot predictions
    ax4 = plt.figure(figsize=[25, 10])
    plt.plot(testY[:, 3])
    plt.plot(test_predict[:, 3], label='predict')
    plt.xlabel("Time Period")
    plt.ylabel("Caramel Matki Sail")
    ax4.legend(loc=8)
    plt.show()

    # Plot predictions
    plt.figure(figsize=[25, 10])
    plt.plot(testY[:, 4])
    plt.plot(test_predict[:, 4], label='predict')
    plt.xlabel("Time Period")
    plt.ylabel("iced Americano Sail")
    ax.legend(loc=8)
    plt.show()