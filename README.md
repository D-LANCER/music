# music
music generation

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers
#导入数据集
from keras.datasets import mnist

#加载数据集，并设置数据集的变量名
(x_train,y_train),(x_test,y_test) = mnist.load_data()
#将一张28*28的图片内的灰度值由0~255压缩到0~1（0~1是预测的概率）
x_train = x_train.reshape(-1,28*28).astype("float32")/255.0
x_test = x_test.reshape(-1,28*28).astype("float32")/255.0

#顺序API(非常方便，非常灵活)
#建立网络，设置一个全连接神经网络
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),#用于打印模型
        layers.Dense(512,activation='relu'),#第一层
        layers.Dense(256,activation='relu'),#第二层
        layers.Dense(10),#最后是10个输出节点，分别对应0~9数字对应的预测概率
#Dense是创造节点，即神经元
    ]
)

print(model.summary())#打印模型的输出

"""
import sys
sys.exit()
"""#这个代码可以使程序只进行到此，不再运行下去，若用了此代码，而下面还有程序的话，就会警告“this code is unreachable”



#告述keras如何配置训练网络
model.compile(
    #指定所用的损失函数(分类交叉熵)
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),#指定优化器和设置学习率
    metrics=["accuracy"],
)

#具体培训数据集和发送数据
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)
# batch_size是一次训练的样本数
#1个epoch等于使用数据集的全部样本训练一次
# .fit中的verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
#默认为1


#对模型的评估
model.evaluate(x_test,y_test,batch_size=32,verbose=2)
#.evaluate中的verbose = 0 为不在标准输出流输出日志信息
#verbose = 1 为输出进度条记录
#注意： 只能取 0 和 1；默认为 1
