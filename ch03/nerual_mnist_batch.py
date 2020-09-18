import numpy as np
import sys,os
sys.path.append(os.pardir)
import pickle
from dataset.mnist import load_mnist

def get_data():
    (x_train,y_train),(x_test,y_test) = load_mnist()
    return x_test,y_test

def init_network():
    with open('sample_weight.pkl','rb') as f:
        network=pickle.load(f)
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    c = np.max(x)
    x_c = x-c
    sum_x = np.sum(np.exp(x_c))
    return np.exp(x_c)/sum_x

def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = softmax(a3)
    return y

x,y = get_data()
network = init_network()

batch_size = 100

count = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    count = count + np.sum(y[i:i+batch_size]==np.argmax(y_batch,axis=1))

print('accu:',count/len(x))
