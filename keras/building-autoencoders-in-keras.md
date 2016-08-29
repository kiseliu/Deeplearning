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
自编码器很少用在实际的应用中。In 2012 they briefly found an application in greedy layer-wise pretraining for deep convolutional neural networks [1][1]，但是它很快就过时了，因为我们开始意识到，更好的随机权重初始化方案足以让我们从头开始训练深度网络。在2014年，batch normalization [2][2] 开始允许更深的网络，从2015年末我们可以使用残差学习随意从头训练深度网络[3][3]。

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
 
## Adding a sparsity constraint on the encoded representations
在前面的例子中，编码表示只受到隐藏层大小(32)的限制。在这样的情况下，隐藏层做的事情是学习PCA的近似表示。另一种限制表示被压缩的方法是对隐藏层表示的行为加入稀疏性限制，在给定的时刻只有少数的单元被启动。在Keras，可以通过在Dense 层加入 activity\_regularizer 做到：\_
	from keras import regularizers
	
	encoding_dim = 32
	
	input_img = Input(shape=(784,))
	# add a Dense layer with a L1 activity regularizer
	encoded = Dense(encoding_dim, activation='relu',
	                activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
	decoded = Dense(784, activation='sigmoid')(encoded)
	
	autoencoder = Model(input=input_img, output=decoded)

我们让模型训练迭代100次(模型加入正则化更不容易过拟合，并且可以被训练地更久)，结束后训练损失是0.11，测试损失是0.10。这两者的差别主要是因为在训练过程中，被加入到损失中的正则化项(大概有0.01)。

这里是新的可视化结果：
![][image-2]

They look pretty similar to the previous model, the only significant difference being the sparsity of the encoded representations. encoded\_imgs.mean() yields a value 3.33 (over our 10,000 test images), whereas with the previous model the same quantity was 7.30. So our new model yields encoded representations that are twice sparser. 

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
![][image-3]

## Convolutional autoencoder
由于我们的输入是图像，使用卷积神经网络作为编码器和解码器是合理的。在实际的场景中，应用在图像上的自编码器通常是卷积自编码器，它们很容易表现地更好。

让我们实现一个。编码器将由Convolution2D 和 MaxPooling2D (max pooling being used for spatial down-sampling) 层堆叠组成，解码器将由Convolution2D 和 UpSampling2D 层堆叠成。
	#coding:utf-8
	import numpy as np
	from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
	from keras.models import Model
	from keras.datasets import mnist
	
	input_img = Input(shape=(1, 28, 28))
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8 ,3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2,2), border_mode='same')(x)
	
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2,2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2,2))(x)
	x = Convolution2D(16, 3, 3, activation='relu')(x)
	x = UpSampling2D((2,2))(x)
	decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
	
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	
	(x_train, _), (x_test, _) = mnist.load_data()
	
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
	x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

为了说明如何可视化训练过程中模型的结果，我们将使用TensorFlow backendand 和 TensorBoard callback。

首先打开终端，然后开启TensorBoard服务器，它会从你指定的文件夹读取日志。

	tensorboard --logdir=/Users/lyj/DeepLearning/tmp/autoencoder

然后训练模型，在callbacks列表中，我们传递一个TensorBoard callback实例。每一轮迭代过后，该callback会把日志写入你指定的文件夹，然后它会被TensorBoard服务器读取。

	from keras.callbacks import TensorBoard
	autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=128, shuffle=True,validation_data=(x_test, x_test),
	                 callbacks=[TensorBoard(log_dir='/Users/lyj/DeepLearning/tmp/autoencoder')])
这里需要注意，Keras默认是把then当作后端的，如果我们要用TensorBoard，需要把后端改成tensorflow，具体见[http://keras-cn.readthedocs.io/en/latest/backend/][5]。

我们可以在TensorBoard 页面监控训练(通过http://0.0.0.0:6006 网址)：

模型收敛时损失为0.094，比我们前面的模型好得多(主要是因为编码表示的更大的熵能力，128维 vs 32维)。让我们看下重塑的数字：
![][image-4]
我们也看一下128维的编码表示，这些表示是8x4x4的，所以可以把它们转变成4x32大小，然后用灰度图像展示出来。

## Application to image denoising
让我们把卷积自编码器用在图像去噪问题上，这非常简单，我们会训练自编码器，然后把噪音数字图像映射到干净的数字图像。

现在让我们产生合成噪音数字：我们只需要应用高斯噪音矩阵，然后clip0～1之间的图像。

	#coding:utf-8
	import numpy as np
	import matplotlib.pyplot as plt
	from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
	from keras.models import Model
	from keras.datasets import mnist
	
	input_img = Input(shape=(1, 28, 28))
	
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)
	
	encoder = Model(input_img, x)
	# at this point the representation is (8, 4, 4) i.e. 128-dimensional
	
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(16, 3, 3, activation='relu')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
	
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	
	(x_train, _), (x_test, _) = mnist.load_data()
	
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
	x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
	
	noise_factor = 0.5
	x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
	x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
	
	x_train_noisy = np.clip(x_train_noisy, 0., 1.)
	x_test_noisy = np.clip(x_test_noisy, 0., 1.)
	
	n = 10
	plt.figure(figsize=(20, 2))
	for i in range(n):
	    ax = plt.subplot(1, n, i+1)
	    plt.imshow(x_test_noisy[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	plt.show()
下面是噪音数字的样子：
![][image-5]

如果你仔细看仍然能认出他们，但是很难。我们的自编码器能够恢复原始的数字吗？让我们看看。

和前面的卷积自编码器相比，为了提高重构的质量，我们用一个稍微不同的模型，它的每层都有更多的过滤：

	#coding:utf-8
	import numpy as np
	import matplotlib.pyplot as plt
	from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
	from keras.models import Model
	from keras.datasets import mnist
	from keras.callbacks import TensorBoard
	
	input_img = Input(shape=(1, 28, 28))
	
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)
	
	encoder = Model(input_img, x)
	# at this point the representation is (32, 7, 7) 
	
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
	
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	
	(x_train, _), (x_test, _) = mnist.load_data()
	
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
	x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
	
	noise_factor = 0.5
	x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
	x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
	
	x_train_noisy = np.clip(x_train_noisy, 0., 1.)
	x_test_noisy = np.clip(x_test_noisy, 0., 1.)
	
	autoencoder.fit(x_train_noisy, x_train,
	                nb_epoch=100,
	                batch_size=128,
	                shuffle=True,
	                validation_data=(x_test_noisy, x_test),
	                callbacks=[TensorBoard(log_dir='/Users/lyj/DeepLearning/tmp/convolutionalantuencoder', histogram_freq=0, write_graph=False)])
	decoded_imgs = autoencoder.predict(x_test_noisy)
	
	n = 10
	plt.figure(figsize=(20, 4))
	for i in range(n):
	    # display original
	    ax = plt.subplot(2, n, i+1)
	    plt.imshow(x_test_noisy[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	
	    # display reconstruction
	    ax = plt.subplot(2, n, i + 1 + n)
	    plt.imshow(decoded_imgs[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	plt.show()

现在我们看一看结果。digits are reconstructed by the network. 上面是我们喂给网络的噪音数字，下面是卷积自编码器网络重构的数字。

看起来该网络工作的很好。如果你使用更大的卷积网络，你可以开始构建文档去噪或者音频去噪模型。[ Kaggle has an interesting dataset to get you started.][6]

[1]:	https://stackoverflow.com/questions/37758496/python-keras-theano-wrong-dimensions-for-deep-autoencoder
[2]:	https://stackoverflow.com/questions/37758496/python-keras-theano-wrong-dimensions-for-deep-autoencoder
[3]:	http://keras-cn.readthedocs.io/en/latest/backend/
[4]:	https://stackoverflow.com/questions/37758496/python-keras-theano-wrong-dimensions-for-deep-autoencoder
[5]:	http://keras-cn.readthedocs.io/en/latest/backend/
[6]:	https://www.kaggle.com/

[image-1]:	https://github.com/kiseliu/MarkDownPictures/blob/master/dl/%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.png
[image-2]:	https://github.com/kiseliu/MarkDownPictures/blob/master/dl/%E7%A8%80%E7%96%8F%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.png
[image-3]:	https://github.com/kiseliu/MarkDownPictures/blob/master/dl/%E6%B7%B1%E5%BA%A6%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.png
[image-4]:	https://github.com/kiseliu/MarkDownPictures/blob/master/dl/%E5%8D%B7%E7%A7%AF%E8%87%AA%E7%BC%96%E7%A0%81.png
[image-5]:	https://github.com/kiseliu/MarkDownPictures/blob/master/dl/%E5%99%AA%E9%9F%B3%E6%95%B0%E5%AD%97.png