#coding:utf-8
import numpy as np
import itertools
from collections import Counter
from data_processing import whole_process

def load_data_and_labels(pos_path, neg_path):
    #加载数据和标签
    positive_examples = whole_process(pos_path)
    negative_examples = whole_process(neg_path)
    text = positive_examples + negative_examples

    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    return [text, np.concatenate([positive_labels, negative_labels], 0)]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    #对数据集生成一个批迭代器
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            # shuffle_indices = np.random.shuffle(np.arange(data.shape[0]))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
