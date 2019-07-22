#!/usr/bin/env python
# coding: utf-8

# # Import Library, Load Data
# 
# ## FULC
# 
# *Fully connected layer* 을 주로 이용한 이 방법은
# 
# input에 이용되는 Feature : 날씨-평균기온, 날씨-평균습도, 날씨-평균바람, 날씨-강수량, 날씨-최고기온 <br>
# 
# <br>
# ** EDA 결과, 날씨-최고기온 / 날씨-평균기온과 커피 검색량은 + correlation 을 보였고, 
# 바람과 강수량에서는 명확한 관계는 보이지 않았으나, log 곡선을 타지 않을까 조심스럽게 예측해 본다. 윤영님같은 경우, 강수량을 0,1,2 로 재정의해서 사용하는 아이디어를 제안하였는데, 다양한 방법을 다 적용해 보는걸로.<br>
# 
# try 1 : 당일 날씨-평균기온, 당일 날씨-평균습도, 당일 날씨-평균바람, 당일 날씨-강수량, 당일 날씨-최고기온 <br>
# try 2 : 당일 날씨-평균기온, 당일 날씨-평균습도, 당일 날씨-평균바람(log), 당일 날씨-강수량(log), 당일 날씨-최고기온 <br>
# try 3 : 당일 날씨-평균기온, 당일 날씨-평균습도, 당일 날씨-평균바람(012), 당일 날씨-강수량(012), 당일 날씨-최고기온 <br>
# try 4 : 요일별도 적용해 보는걸로.

# In[36]:


import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns


# ## Hyper Parameters

# In[37]:


kind_of_dataset = 'juice'
#kind of dataset : all, coco, ia (iced americano), icetea, juice, latte


kind_of_activation_function = 'lrelu'
#kind of this parameter : lrelu (leak - relu), relu, tanh

learning_rate = 0.01
step = 50000

#coco : 60000 , [fri, sat, sun] delete recommend
#ia : 13000 or 100000, [fri, sat, sun] delete recommend
#icetea : [fri, sat, sun] delete recommend
#juice : recommend
#latte : 10000 recommend



## Note1 : 뒤에 "선택하지 않을 요일" 조절하는 함수 있음.
## Note2 : 뒤에 "넣지 않는 column" 조절하는 함수 있음.
## Note3 : 뒤에 scaling 하는 함수 있음.


# In[38]:


def importDataset(name) :
    print('kind of dataset : all, coco, ia, icetea, juice, latte')
    print('import dataset : ', name , '\n\n')
    dataset = pd.read_csv('bev_' + name + '_weather.csv', index_col = 0)
    print(dataset.head())
    print('\n\n')
    return dataset

def removeUnimportantColumns(dataframe) : 
    new_dataframe = dataframe.drop([
        'datetime',
        'weather.stn_id',
        'weather.min_ta',
    ], axis = 1)
    print(new_dataframe.columns)
    return new_dataframe

df_dataset = importDataset(kind_of_dataset)
df_dataset = removeUnimportantColumns(df_dataset)


# In[39]:


plt.figure(figsize = [15, 8])
sns.lineplot(x = df_dataset['weekday'], y = df_dataset.iloc[:,0])


# # Model

# In[40]:


# DNN : Deep Neural Network


# In[41]:


#about training

def set_activation_function(string):
    print('activation function :',string, '\n\n')
    if string == 'tanh' :
        return tf.tanh, string
    elif string == 'relu' :
        return tf.nn.relu, string
    elif string == 'lrelu' :
        return tf.nn.leaky_relu, string

#activation function
activation_function, activation_functionname = set_activation_function(kind_of_activation_function) #relu


# ## Data Pre-Processing

# In[42]:


def select_weekday(data, *weekname_args) :
    print('function input : \n' , data.head())
    select_weekday_val = []
    original_weekday_val = [0,1,2,3,4,5,6]
    data_return = data.copy()
    
    if 'mon' in weekname_args :
        select_weekday_val.append(0)
    if 'tue' in weekname_args :
        select_weekday_val.append(1)
    if 'wed' in weekname_args :
        select_weekday_val.append(2)
    if 'thu' in weekname_args :
        select_weekday_val.append(3)
    if 'fri' in weekname_args :
        select_weekday_val.append(4)
    if 'sat' in weekname_args :
        select_weekday_val.append(5)    
    if 'sun' in weekname_args :
        select_weekday_val.append(6)

    print(data_return.head())
    for i in select_weekday_val :
        tmpdata = data_return.drop(index = (data_return.loc[(data_return['weekday'] == i) == True]).index)
        data_return = tmpdata

    print(data_return.weekday.unique())
    return data_return
    

def select_for_train_and_test(data, *columnname_args) :        
    drop_column = []
    for column in columnname_args :
        drop_column.append(column)
    
    df_xy = data.drop(drop_column, axis = 1)
    print(df_xy.columns)
    return df_xy


#df_dataset_weekday_selected = select_weekday(df_dataset, 'fri', 'sat', 'sun')
df_dataset_weekday_selected = df_dataset
df_xy = select_for_train_and_test(df_dataset_weekday_selected, 'weather.max_ws', 'weather.max_ta', 'rn_label', 'weekday')
np_xy = df_xy.to_numpy()
batch_size = np_size = len(np_xy)
np_xy


# In[43]:


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


def LogScaler(data) :
    pass


#scaled_np_xy = MinMaxScaler(np_xy)
scaled_np_xy = StandardScaler(np_xy)

scaled_np_xy


# In[44]:


# train/test split
train_size = int(len(scaled_np_xy) * 0.7)
train_set = scaled_np_xy[0:train_size]
test_set = scaled_np_xy[train_size:]
print('train size : ', train_size)
print('test size : ', len(test_set))
# Index from [train_size - seq_length] to utilize past sequence


# In[45]:


# build datasets
def build_dataset(dataset, target_column):
    
    dataset_X = []
    column_array = []
    column_array = list(range(0, len(dataset[0])))
    column_array.remove(target_column)
    print(column_array)
    
    return dataset[:,column_array], dataset[:,target_column]

trainX, trainY = build_dataset(train_set, 0)
testX, testY   = build_dataset(test_set , 0)


# In[46]:


trainY = trainY.reshape(-1, 1)
print(np.shape(trainX), np.shape(trainY))
testY = testY.reshape(-1,1)
print(np.shape(testX), np.shape(testY))


# ## DNN Model

# In[47]:


X = tf.placeholder(tf.float32, [None, len(trainX[0])])
Y = tf.placeholder(tf.float32, [None, 1])


# In[48]:


w1 = tf.Variable(tf.random_normal([len(trainX[0]), len(trainX[0])]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([len(trainX[0])]), name = 'bias1')
layer1 = activation_function(tf.add(tf.matmul(X, w1), b1))

w2 = tf.Variable(tf.random_normal([len(trainX[0]), len(trainX[0])]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([len(trainX[0])]), name = 'bias2')
layer2 = activation_function(tf.add(tf.matmul(layer1, w2), b2))


w3 = tf.Variable(tf.random_normal([len(trainX[0]), len(trainX[0])]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([len(trainX[0])]), name = 'bias3')
layer3 = activation_function(tf.add(tf.matmul(layer2, w3), b3))


w4 = tf.Variable(tf.random_normal([len(trainX[0]), len(trainX[0])]), name = 'weight4')
b4 = tf.Variable(tf.random_normal([len(trainX[0])]), name = 'bias4')
layer4 = activation_function(tf.add(tf.matmul(layer3, w4), b4))


w5 = tf.Variable(tf.random_normal([len(trainX[0]), len(trainX[0])]), name = 'weight5')
b5 = tf.Variable(tf.random_normal([len(trainX[0])]), name = 'bias5')
layer5 = activation_function(tf.add(tf.matmul(layer4, w5), b5))



finalw = tf.Variable(tf.random_normal([len(trainX[0]), 1]), name = 'finalweight')
finalb = tf.Variable(tf.random_normal([1]), name = 'finalbias')
output_layer = activation_function(tf.add(tf.matmul(layer5, finalw), finalb))


print(output_layer)


# In[49]:


cost = tf.reduce_mean(tf.square(output_layer - Y))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(cost)


# In[50]:


# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
print(targets, '\n', predictions, '\n', rmse)

np.shape(testX)
np.shape(testY)


# In[51]:


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(step):
        _, step_loss = sess.run([train, cost], feed_dict={
                                X: trainX, Y: trainY})

    test_predict = sess.run(output_layer, feed_dict={X: testX})
    train_predict = sess.run(output_layer, feed_dict= {X: trainX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict.reshape(-1,1)})
    print("RMSE: {}".format(rmse_val))


# In[62]:


ax = plt.figure(figsize = [25,10])
plt.plot(trainY[:,0])
plt.plot(train_predict[:,0], label = 'training_set(learned) + test_set(predict)')
#plt.scatter(x = range(0,len(trainY[:,0])), y = trainY[:])
#plt.scatter(x = range(0,len(train_predict[:,0])), y = train_predict[:], label = 'training_set')
plt.plot(trainX[:,1], label = 'average_temp', alpha= 0.5, linestyle='dashed')
plt.xlabel("Time Period")
plt.ylabel(kind_of_dataset + "Sail")
ax.legend(loc = 8)
plt.savefig('./DNNGraph/' + kind_of_dataset + '_' + str(step) + '_trainandtest.png', dpi = 800)
plt.show()



# In[69]:


# plot predictions
ax = plt.figure(figsize= [25,10])
plt.plot(testY[:, 0])
plt.plot(test_predict[:, 0], label = 'testset predict')
plt.plot(testX[:,1], label = 'average_temp', alpha = 0.5, linestyle='dashed' )
plt.xlabel("Time Period")
plt.ylabel(kind_of_dataset + "Sail")
ax.legend(loc = 8)
plt.savefig('./DNNGraph/' + kind_of_dataset + '_' + str(step) + '_test.png', dpi = 800)
plt.show()



ax = plt.figure(figsize= [25,10])
plt.plot(testX[:,1], label = 'average_temp', alpha = 0.5, linestyle='dashed', color = 'green' )
plt.scatter(x = range(0,len(testY[:,0])), y = testY[:])
plt.scatter(x = range(0,len(test_predict[:,0])), y = test_predict[:], label = 'testset predict')
plt.xlabel("Time Period")
plt.ylabel(kind_of_dataset + "Sail")
ax.legend(loc = 8)
plt.savefig('./DNNGraph/' + kind_of_dataset + '_' + str(step) + '_scatter_test.png', dpi = 800)
plt.show()



# ## 아무리 학습을 시켜도...
# 
# 저기 내려가는 부분이 경향을 파악하지 못함. 하지만 놀라울 정도로 예측을 잘하는 경우도 존재함. 분명히 날씨변수만 넣었음에도 이런 추세를 따라가는 것은 분명히 어느정도 상관관계가 있다는 것을 알 수 있음.
