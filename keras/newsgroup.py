#coding:utf-8
import os

texts = []
labels_index = {}
labels = []
text_data_dir = "/Users/lyj/Programs/Data/20news-bydate/20news-bydate-train"
for name in sorted(os.listdir(text_data_dir)):
    path = os.path.join(text_data_dir, name)
    if os.path.isdir(path):
	label_id = len(labels_index)
	labels_index[name] = label_id

