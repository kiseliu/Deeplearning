#coding:utf-8
import os
import numpy as np
from keras.utils.np_utils import to_categorical
#pad_sequences: Pads each sequence to the same length, the length of the longest sequence.

texts = []
labels_index = {}
labels = []
text_data_dir = "/Users/lyj/Programs/Data/20news-bydate/20news-bydate-train"
for name in sorted(os.listdir(text_data_dir)):
    path = os.path.join(text_data_dir, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath)
                texts.append(f.read())
                f.close()
                labels.append(label_id)

print labels
print texts[0]
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# MAX_NB_WORDS = 20000
# tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(texts)
# word_index = tokenizer.word_index
# print "Found %s unique tokens." % len(word_index)
# # print word_index
#
# sequences = tokenizer.texts_to_sequences(texts)
#
# MAX_SEQUENCE_LENGTH = 1000
# data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# labels = to_categorical(np.asarray(labels))
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)
#
#
# # shuffle数据
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# # 然后把数据分成训练集和测试集
# VALIDATION_SPLIT = 0.2
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
# x_train = data[:-nb_validation_samples]
# y_train = labels[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]
#
# #从预训练好的GloVe文件中解析出每个词和它所对应的词向量，并用字典的方式存储
# embeddings_index = {}
# glove_text = '/Users/lyj/Programs/Data/glove.6B/glove.6B.100d.txt'
# f = open(glove_text)
# for line in f:
#     values = line.split()
#     word = values[0]  #取词
#     coefs = np.asarray(values[1:], dtype='float32')  #取向量
#     embeddings_index[word] = coefs  #将词和对应的向量存到字典里
# f.close()
# print('Found %s word vectors.' % len(embeddings_index))
#
# EMBEDDING_DIM = 100
# embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
# for word, i in word_index.items():  #借助字典embeddings_index[词,向量]将"词和索引"转化为"索引和向量"
#     embedding_vector = embeddings_index.get(word)   # 在embedding index 中没有找到的词,其对应的向量会是全0
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
#
# from keras.layers import Embedding
# embeddings_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix],
#                              input_length=MAX_SEQUENCE_LENGTH, trainable=False)
#
# from keras.layers import Dense, Input, Flatten
# from keras.layers import Conv1D, MaxPooling1D, Embedding
# from keras.models import Model
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embeddings_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(35)(x)
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# preds = Dense(len(labels_index), activation='softmax')(x)
#
# model = Model(sequence_input, preds)
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
#
# model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=2, batch_size=128)
