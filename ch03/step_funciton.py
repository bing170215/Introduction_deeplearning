import numpy as np
import matplotlib.pyplot as plt

#可以接受numpy数组
def step_function(x):
    #数组的各个元素都会进行不等号运算，生成一个布尔型数组
    #dtype=np.int 将元素类型将布尔型转换为int型
    return np.array(x>0,dtype=np.int)
X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
plt.show()
