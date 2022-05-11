# -*- coding: utf-8 -*-
"""
Created on Sat May  7 19:14:51 2022

@author: WeijieXia
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

#%% 
'normal neural network'
data = pd.read_csv(r'../data/learning_data_2.csv')
data = data.dropna()
x_data = data.iloc[:,1:16]
y_data = data.loc[:,['charging']]

'split train test'
x_data1 = MinMaxScaler((-100,100)).fit_transform(x_data) 
y_data1 = MinMaxScaler((-100,100)).fit_transform(y_data)
x_train,x_test,y_train,y_test = train_test_split(x_data1,y_data1,test_size=0.3,random_state=0)


def scalar(y,y_data):
    z = y-y_data.min()
    z = z/( y_data.max()-y_data.min())
    return z 

'Network parameters'
n_input=15
h1=15
h2=15
n_output=1
learning_rate = 0.003  
training_epochs = 15   
max_epoch = 200
display_step = 1      

x = tf.placeholder("float",[None, n_input])
y = tf.placeholder("float",[None, n_output])

def multilayer_perceptron(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    out_layer = tf.matmul(layer_2,weights['out'] + biases['out'])
    return out_layer

weights ={
    'h1':tf.Variable(tf.random_normal([n_input, h1])),
    'h2':tf.Variable(tf.random_normal([h1, h2])),
    'out':tf.Variable(tf.random_normal([h2, n_output]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([h1])),
    'b2': tf.Variable(tf.random_normal([h2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

y_hat = multilayer_perceptron(x, weights, biases)
cost = tf.square(y-y_hat)
mse = tf.reduce_mean(tf.cast(cost,'float'))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)
init = tf.global_variables_initializer()

Loss_epoch_nn_t = []
Loss_epoch_nn_p = []
violation_nn = []
boundary = 101.557707
with tf.Session() as  sess:
    sess.run(init)
    for epoch in range(2000):
        _, c = sess.run([train, mse], feed_dict = {x: x_train, y: y_train})
        Loss_epoch_nn_t.append(c)
        pre_mse, pre_y = sess.run([mse, y_hat], feed_dict = {x:x_test, y:y_test})
        Loss_epoch_nn_p.append(pre_mse)
        violation_nn.append(np.sum(pre_y>boundary)+np.sum(pre_y<-boundary))
        print("Epoch %02d, Loss = %.6f" %(epoch, c))
    print('Training Done')
    pre_mse, pre_y = sess.run([mse, y_hat], feed_dict = {x:x_test, y:y_test})
print('Optimization Finished!')

#%% 
'normal neural network'
data = pd.read_csv(r'../data/learning_data_2.csv')
data = data.dropna()
x_data = data.iloc[:,1:16]
y_data = data.loc[:,['charging']]

'split train test'
x_data1 = MinMaxScaler((-100,100)).fit_transform(x_data) 
y_data1 = MinMaxScaler((-100,100)).fit_transform(y_data)
x_train,x_test,y_train,y_test = train_test_split(x_data1,y_data1,test_size=0.3,random_state=0)


def scalar(y,y_data):
    z = y-y_data.min()
    z = z/( y_data.max()-y_data.min())
    return z 

'Network parameters'
n_input=15
h1=15
h2=15
n_output=1
learning_rate = 0.003  
training_epochs = 15   
max_epoch = 200
display_step = 1       
lam = 1/100.0

x = tf.placeholder("float",[None, n_input])
y = tf.placeholder("float",[None, n_output])

def multilayer_perceptron(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    out_layer = tf.matmul(layer_2,weights['out'] + biases['out'])
    return out_layer

weights ={
    'h1':tf.Variable(tf.random_normal([n_input, h1])),
    'h2':tf.Variable(tf.random_normal([h1, h2])),
    'out':tf.Variable(tf.random_normal([h2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([h1])),
    'b2': tf.Variable(tf.random_normal([h2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# y_hat = multilayer_perceptron(x, weights, biases)
# cost = tf.square(y-y_hat)
# mse = tf.reduce_mean(tf.cast(cost,'float'))
# train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)
# init = tf.global_variables_initializer()
'physical rules'
p_simulation1 = tf.multiply(tf.constant(0.0014),tf.pow(y,6))
p_simulation2 = tf.multiply(tf.constant(0.0346),tf.pow(y,5))
p_simulation3 = tf.multiply(tf.constant(-21.065),tf.pow(y,4))
p_simulation4 = tf.multiply(tf.constant(-346.54),tf.pow(y,3))
p_simulation5 = tf.multiply(tf.constant(73368.0),tf.pow(y,2))
p_simulation6 = tf.multiply(tf.constant(608868.0),tf.pow(y,1))
p_simulation = tf.add(p_simulation1,tf.constant(-3e+7))
p_simulation = tf.add(p_simulation,p_simulation2)
p_simulation = tf.add(p_simulation,p_simulation3)
p_simulation = tf.add(p_simulation,p_simulation4)
p_simulation = tf.add(p_simulation,p_simulation5)
p_simulation = tf.add(p_simulation,p_simulation6)
physical_correction = tf.multiply(lam,p_simulation) 

y_hat = multilayer_perceptron(x, weights, biases)
cost = tf.square(y-y_hat)
cost_pinn = tf.add(tf.square(y-y_hat),physical_correction)
mse = tf.reduce_mean(tf.cast(cost,'float'))
mse_pinn = tf.reduce_mean(tf.cast(cost_pinn,'float'))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse_pinn)
init = tf.global_variables_initializer()

Loss_epoch_pinn_t = []
Loss_epoch_pinn_p = []
violation_pinn = []
boundary = 101.557707
with tf.Session() as  sess:
    sess.run(init)
    for epoch in range(2000):
        _, c_pinn,loss_pinn = sess.run([train, mse, mse_pinn], feed_dict = {x: x_train, y: y_train})
        Loss_epoch_pinn_t.append(c_pinn)
        pre_mse_pinn, pre_y_pinn, mse_pinn_out = sess.run([mse, y_hat, mse_pinn], feed_dict = {x:x_test, y:y_test})
        Loss_epoch_pinn_p.append(pre_mse_pinn)
        violation_pinn.append(np.sum(pre_y_pinn>boundary)+np.sum(pre_y_pinn<-boundary))
        print('Epoch loss loss2 ',epoch,c_pinn,loss_pinn)
        # print("Epoch %02d, Loss = %.6f, loss2 = %" %(epoch, c, loss2))
    print('Training Done')
    pre_mse, pre_y = sess.run([mse, y_hat], feed_dict = {x:x_test, y:y_test})
print('Optimization Finished!')

#%%  
plt.title('Violation of physical limitation comparison INN & PINs')
plt.ylabel('Times of violating physical limitation')
plt.xlabel('Times of training')
plt.plot(violation_nn)
plt.plot(violation_pinn)
plt.legend(labels=['NN','PINNs'],loc='best')
plt.show()

plt.title('Loss comparison INN & PINs')
plt.ylabel('Loss ')
plt.xlabel('Times of training')
plt.plot(Loss_epoch_nn_t[:200])
plt.plot(Loss_epoch_pinn_t[:200])
plt.legend(labels=['NN','PINNs'],loc='best')
plt.show()


