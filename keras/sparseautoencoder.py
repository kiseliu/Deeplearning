#coding:utf-8
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

encoding_dim = 32  #即编码的大小，假设输入是784 floats，那么32 floats 的压缩率是24.5

#建立自编码器模型
input_img = Input(shape=(784,))  # this is our input placeholder
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.activity_l1(10e-7))(input_img)  # "encoded" 是输入的编码表示，系数是－7次才能得到官方示例的结果
decoded = Dense(784, activation='sigmoid')(encoded)   # "decoded" 是输入的有损失的重构
autoencoder = Model(input=input_img, output=decoded)   # 把输入映射到它的重构上

#建立单独的编码器模型
encoder = Model(input=input_img, output=encoded)   #该模型把输入映射到它的编码表示

#构建解码器模型
encoded_input = Input(shape=(encoding_dim,))   #create a placeholder for an encoded (32-dimensional) input 因为解码器的输入是编码的维度encoding_dim
decoder_layer = autoencoder.layers[-1]  #自编码器模型的最后一层就是解码器
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))   #把编码后的数据映射到最后的输出，感觉这里换成autoencode(input_img)也可以

#训练自编码器来对MNIST数字进行重构
(x_train, _), (x_test, _) = mnist.load_data() #准备输入数据，丢弃数据的标签
x_train = x_train.astype('float32') / 255.  # 对数据进行归一化
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  #把28*28的图像平展成784大小的向量
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape, x_test.shape

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')  #配置模型参数，基于元素的二元交叉熵损失，和Adadelta优化算子
autoencoder.fit(x_train, x_train, nb_epoch=100, batch_size=256,
            shuffle=True, validation_data=(x_test, x_test))  #训练自编码器，测试集作为交叉验证集

encoded_imgs = encoder.predict(x_test)  #对测试数据进行编码
decoded_imgs = decoder.predict(encoded_imgs)   #对测试数据进行解码

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)  # display original
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)  # display reconstruction
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
