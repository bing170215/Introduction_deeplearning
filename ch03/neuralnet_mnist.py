import sys,os
sys.path.append(os.pardir) #为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    c = np.max(x)
    x_c = x -c
    sum_x = np.sum(np.exp(x_c))
    return np.exp(x_c)/sum_x

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test,t_test

def init_network():
    with open('sample_weight.pkl','rb') as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) +b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y

x,t = get_data()

network = init_network()
count = 0
for i in range(len(x)):
    if i%100==0:
        print('当前已执行:',100*i)

    if t[i]==np.argmax(predict(network,x[i])):
        count = count +1

print('Accu:',count/len(x))
