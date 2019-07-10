import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

def set_activation_function(string):
    if string == 'tanh' :
        return tf.tanh, string
    elif string == 'relu' :
        return tf.nn.relu, string


#about X
timesteps = sequence_length = 7
data_dim = 7

#about Y and hidden size
hidden_layer_size = 5
hidden_dim = 10
output_dim = 2

#about training
learning_rate = 0.015
epoch = 1

#activation function
activation_function, activation_functionname = set_activation_function('relu') #tanh




df_init = pd.read_csv('coffee_weather_janghoo.csv')
df_init = df_init.drop(['weather_mintemp', 'weather_maxwindspeed', 'weather_mintemp'], axis = 1)
df_weather_columns = df_init.iloc[:, 6:]
df_americano_and_icedamericano = df_init.drop(['caramelmatk', 'cafemoca', 'cafelatte'],axis = 1)
df_caffelatte_and_americano = df_init.drop(['caramelmatk', 'cafemoca', 'icedamericano'], axis = 1)


df_xy = df_americano_and_icedamericano.iloc[:,1:]
np_xy = df_xy.to_numpy()
batch_size = np_size = len(np_xy)


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


def StandardScaler(data):
    #    avg = []
    #    for i in range(0, np_xy[0], 1) :
    #        np.mean(np_xy[:,i])
    avg = np.mean(np_xy, axis=0)
    std = np.std(np_xy, axis=0)

    final = (data - avg) / std

    return final


# scaled_np_xy = MinMaxScaler(np_xy)
scaled_np_xy = StandardScaler(np_xy)

# train/test split
train_size = int(len(scaled_np_xy) * 0.7)
train_set = scaled_np_xy[0:train_size]
test_set = scaled_np_xy[train_size - sequence_length:]
print('size : ', train_size)
# Index from [train_size - seq_length] to utilize past sequence


# build datasets
def build_dataset(time_series, seq_length): # seq_length : count of time series (timesteps)
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length): # 7개씩 묶는 과정
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, 0:2]  # Next close
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, sequence_length)
testX, testY   = build_dataset(test_set , sequence_length)
X = tf.placeholder(tf.float32, [None, sequence_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])
# build a LSTM network
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_dim,
                                        state_is_tuple=True,
                                        activation = activation_function)
    return cell


#cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,
#                                    state_is_tuple=True, activation=activation_function)

cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hidden_layer_size)],
                                   state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1],
    output_dim,
    activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 2])
predictions = tf.placeholder(tf.float32, [None, 2])
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


filename1 = str(learning_rate * 1000) + '_' + \
           str(activation_functionname) + '_' + \
           str(hidden_dim) + '_' + \
           str(hidden_layer_size) + '_' + 'Americano.png'

filename2 = str(learning_rate * 1000) + '_' + \
           str(activation_functionname) + '_' + \
           str(hidden_dim) + '_' + \
           str(hidden_layer_size) + '_' + 'IcedAmericano.png'

# Plot predictions
ax = plt.figure(figsize = [25,10])
plt.plot(testY[:,0])
plt.plot(test_predict[:,0], label='predict')
plt.xlabel("Time Period")
plt.ylabel("Americano Sail")
ax.legend(loc = 8)
plt.savefig('./Graph/' + filename1, dpi=600)
plt.show()


# Plot predictions
ax2 = plt.figure(figsize = [25,10])
plt.plot(testY[:,1])
plt.plot(test_predict[:,1], label = 'predict')
plt.xlabel("Time Period")
plt.ylabel("Iced_Americano Sail")
ax2.legend(loc = 8)
plt.savefig('./Graph/' + filename2, dpi=600)
plt.show()




