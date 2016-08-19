本节，我们将回答关于自编码的一些常见的问题，我们将给出下面几个模型的代码示例：
- 基于全连接层的简单的自编码器
- 稀疏自编码器
- 深度全连接自编码器
- 深度卷积自编码器
- 图像去噪模型
- 序列到序列的自编码器
- 变自编码器

## What are autoencoders?
“Autoencoder”是一种数据压缩算法，它的压缩和解压函数是1)数据专用的，2)有损失的，3)自动从样本中学习而不是人工设计的。此外，在文章中术语”autoencoder”出现的地方，表示压缩和解压都是由神经网络实现的。

1) 自编码器是数据专用的，表示它们只能用来压缩和训练数据相似的数据。基于脸部图片训练的自编码在压缩树的图片效果很差，因为它学习到的特征是脸部专用的。

2) 自编码器是有损的，即和原始的输入相比，压缩后的输出降级了，这不同于无损算法的压缩。

3) 自编码器自动从数据样本中学习，这是一个有用的性质。it means that it is easy to train specialized instances of the algorithm that will perform well on a specific type of input.它不需要任何新的设计，只需要合适的训练数据。

要想构建一个自编码器，你需要三件事：编码函数，解码函数，和衡量数据的压缩表示和解压表示间的信息损失的距离函数(i.e. a "loss" function)。编码器和解码器需要选择是参数方程(尤其是神经网络)，并且对于距离函数是可微分的，这样编码/解码函数的参数可以通过使用SGD减小重构损失进行优化。这非常简单，你甚至不需要理解任何文字就可以开始使用自编码器。

## Are they good at data compression?
通常不是这样的，比如在图像压缩中，很难训练一个自编码器做的比基础算法，像JPEG效果好。它能做到的唯一的方法就是，限制在一种非常特殊的图片(比如，JPEG做得不好的图片上)。自编码器是数据专用的的事实让它们对于现实世界的数据压缩问题通常都是不实际的：你只能把它们用在和训练集相似的数据上，因此想要让它们变得通用需要大量的训练数据。但是未来的进步可能会改变这一现状，谁知道呢？


## What are autoencoders good for?
自编码器很少用在实际的应用中。In 2012 they briefly found an application in greedy layer-wise pretraining for deep convolutional neural networks [1]，但是它很快就过时了，因为我们开始意识到，更好的随机权重初始化方案足以让我们从头开始训练深度网络。在2014年，batch normalization [2] 开始允许更深的网络，从2015年末我们可以使用残差学习随意从头训练深度网络[3]。

如今自编码器的两个有趣的实际应用是**数据去噪**(本文后续会说)，和用于**数据可视化的降维**。有了合适的维度和稀疏性限制，自编码器可以学习数据投影，它比PCA或者其它基本的技术更有趣。

For 2D visualization specifically, t-SNE (pronounced "tee-snee") is probably the best algorithm around, but it typically requires relatively low-dimensional data. So a good strategy for visualizing similarity relationships in high-dimensional data is to start by using an autoencoder to compress your data into a low-dimensional space (e.g. 32 dimensional), then use t-SNE for mapping the compressed data to a 2D plane. Note that a nice parametric implementation of t-SNE in Keras was developed by Kyle McDonald and is available on Github. Otherwise scikit-learn also has a simple and practical implementation.



## So what's the big deal with autoencoders?

## Let's build the simplest possible autoencoder
我们先从最简单的单层全连接神经网络开始建立编码器和解码器：

	#coding:utf-8
	from keras.layers import Input, Dense
	from keras.models import Model
	from keras.datasets import mnist
	import numpy as np
	import matplotlib.pyplot as plt
	
	encoding_dim = 32  #即编码的大小，假设输入是784 floats，那么32 floats 的压缩率是24.5
	
	#建立自编码器模型
	input_img = Input(shape=(784,))  # this is our input placeholder
	encoded = Dense(encoding_dim, activation='relu')(input_img)  # "encoded" 是输入的编码表示
	decoded = Dense(784, activation='sigmoid')(encoded)   # "decoded" 是输入的有损失的重构
	autoencoder = Model(input=input_img, output=decoded)   # 把输入映射到它的重构上
	
	#建立单独的编码器模型
	encoder = Model(input=input_img, output=encoded)   #该模型把输入映射到它的编码表示
	
	#构建解码器模型
	encoded_input = Input(shape=(encoding_dim,))   #create a placeholder for an encoded (32-dimensional) input 因为解码器的输入是编码的维度encoding_dim
	decoder_layer = autoencoder.layers[-1]  #自编码器模型的最后一层就是解码器
	decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))   #把编码后的数据映射到最后的输出，感觉这里换成autoencode(input_img)也可以
	
	#训练自编码器来对MNIST数字进行重构
	(x_train, _), (x_test, _) = mnist.load_data() # 准备输入数据，丢弃数据的标签
	x_train = x_train.astype('float32') / 255.  # 对数据进行归一化
	x_test = x_test.astype('float32') / 255.
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  #把28*28的图像平展成784大小的向量
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	print x_train.shape, x_test.shape
	
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')  #配置模型参数，基于元素的二元交叉熵损失，和Adadelta优化算子
	autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=256, 
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
最终经过压缩再解压后的图片效果如下：
![][image-1]   
如果我们把编码后的经过压缩的图像用matplotlib画出来(也就是可视化提取的特征)如下：
![](https://github.com/kiseliu/MarkDownPictures/blob/master/dl/%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E6%8F%90%E5%8F%96%E7%9A%84%E7%89%B9%E5%BE%81.png)	  
表示看到之后，只想说什么鬼？
 
## Adding a sparsity constraint on the encoded representations

## Deep autoencoder
我们不必限制编码器或者解码器为单层，相反我们可以多叠几层，比如：

	#coding:utf-8
	from keras.layers import Input, Dense
	from keras.models import Model
	from keras.datasets import mnist
	import numpy as np
	import matplotlib.pyplot as plt
	
	encoding_dim = 32  #即编码的大小，假设输入是784 floats，那么32 floats 的压缩率是24.5
	
	#建立自编码器模型
	input_img = Input(shape=(784,))  # this is our input placeholder
	encoded = Dense(128, activation='relu')(input_img) 
	encoded = Dense(64, activation='relu')(encoded)
	encoded = Dense(32, activation='relu')(encoded)  # "encoded" 是输入的编码表示
	
	decoded = Dense(64, activation='relu')(encoded)
	decoded = Dense(128, activation='relu')(decoded)
	decoded = Dense(784, activation='sigmoid')(decoded)  # "decoded" 是输入的有损失的重构
	
	autoencoder = Model(input=input_img, output=decoded)   # 把输入映射到它的重构上
	
	#建立单独的编码器模型
	encoder = Model(input=input_img, output=encoded)   #该模型把输入映射到它的编码表示
	
	#构建解码器模型
	encoded_input = Input(shape=(encoding_dim,))   #create a placeholder for an encoded (32-dimensional) input 因为解码器的输入是编码的维度encoding_dim
	decoder_layer1 = autoencoder.layers[-3]   #⚠️
	decoder_layer2 = autoencoder.layers[-2]  #⚠️
	decoder_layer3 = autoencoder.layers[-1]   #⚠️
	decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))   #⚠️
	
	#训练自编码器来对MNIST数字进行重构
	(x_train, _), (x_test, _) = mnist.load_data() # 准备输入数据，丢弃数据的标签
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
注意，由于构建了深度解码器，在单独构建解码器的时候，不能再取最后一层作为解码器了，因为此时最后一层的输入是128维，而此时encoding dim是32维，所以程序会报错，具体可以参考[stack overflow上的回答][4]，开始我也犯了同样的错误。
最终经过压缩和解压后的图像如下：
![](https://github.com/kiseliu/MarkDownPictures/blob/master/dl/%E6%B7%B1%E5%BA%A6%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.png)


[4]:	https://stackoverflow.com/questions/37758496/python-keras-theano-wrong-dimensions-for-deep-autoencoder

[image-1]:	https://github.com/kiseliu/MarkDownPictures/blob/master/dl/%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.png
