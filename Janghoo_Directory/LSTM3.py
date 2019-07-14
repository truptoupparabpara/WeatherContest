#!/usr/bin/env python
# coding: utf-8

# # Import Library, Load Data
# 
# ## LSTM2 > LSTM3
# 
# LSTM 3에서는 내일의 날씨를 함께 학습시킵니다.

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


import pandas as pd
from pandas import DataFrame, Series


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df_init = pd.read_csv('coffee_weather_janghoo.csv')
df_init = df_init.drop(['weather_mintemp', 'weather_maxwindspeed', 'weather_mintemp'], axis = 1)
df_weather_columns = df_init.iloc[:, 6:]
df_init.head()


# In[5]:


df_weather_columns.head()


# In[6]:


df_americano_and_icedamericano = df_init.drop(['caramelmatk', 'cafemoca', 'cafelatte'],axis = 1)
df_caffelatte_and_americano = df_init.drop(['caramelmatk', 'cafemoca', 'icedamericano'], axis = 1)



# # Model

# In[8]:


# 1. RNN
# 2. LSTM


# ## Hyper parameters

# In[9]:


#about X
timesteps = sequence_length = 365
data_dim = 5

#about Y and hidden size
hidden_layer_size = 5
hidden_dim = 20
output_dim = 1

#about training
learning_rate = 0.04
epoch = 1


def set_activation_function(string):
    if string == 'tanh' :
        return tf.tanh, string
    elif string == 'relu' :
        return tf.nn.relu, string

#activation function
activation_function, activation_functionname = set_activation_function('relu') #relu
activation_function, activation_functionname


# <h4>Kinds of Hyper parameters</h4>
# 
# - timesteps : <br>
# - data_dim : data dimension <br>
# - hidden_layer_size <br>
# - hidden_dim = hidden_dimension <br>
#     - https://stackoverflow.com/questions/37901047/what-is-num-units-in-tensorflow-basiclstmcell <br>
#     - more hidden dimension, more memorize <br>
#     
#     
# 

# ## Data Pre-Processing

# In[10]:


#df_xy = df_americano_and_icedamericano.iloc[:,1:]
df_xy = pd.concat([df_weather_columns, df_init['icedamericano']], axis = 1)
np_xy = df_xy.to_numpy()
batch_size = np_size = len(np_xy)
np_xy


# In[11]:


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
    avg = np.mean(np_xy, axis = 0)
    std = np.std(np_xy, axis = 0)
    
    final = (data - avg) / std
    
    return final




#scaled_np_xy = MinMaxScaler(np_xy)
scaled_np_xy = StandardScaler(np_xy)

scaled_np_xy


# In[12]:


# train/test split
train_size = int(len(scaled_np_xy) * 0.7)
train_set = scaled_np_xy[0:train_size]
test_set = scaled_np_xy[train_size - sequence_length:]
print('size : ', train_size)
# Index from [train_size - seq_length] to utilize past sequence


# In[13]:


# build datasets
def build_dataset(time_series, seq_length): # seq_length : count of time series (timesteps)
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length - 1): # 7개씩 묶는 과정
        _x = time_series[i + 1:i + seq_length + 1, :-1]
        _y = time_series[i + seq_length, -1]  # Next close
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, sequence_length)
testX, testY   = build_dataset(test_set , sequence_length)


# In[14]:


print(np.shape(trainX), np.shape(trainY))
print(np.shape(testX), np.shape(testY))


# ## RNN Model

# In[15]:


X = tf.placeholder(tf.float32, [None, sequence_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])


# In[16]:


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


# In[17]:


# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)


# In[28]:


# RMSE
targets = tf.placeholder(tf.float32, [None, output_dim])
predictions = tf.placeholder(tf.float32, [None, output_dim])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

if (output_dim == 1) :
    trainY = trainY.reshape([-1,1])
    testY = testY.reshape([-1,1])
    print(trainY.shape)


# In[29]:


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


# In[31]:


# Plot predictions
    ax = plt.figure(figsize = [25,10])
    plt.plot(testY[:,:])
    plt.plot(test_predict[:,:], label='predict')
    plt.xlabel("Time Period")
    plt.ylabel("Americano Sail")
    ax.legend(loc = 8)
    plt.show()




