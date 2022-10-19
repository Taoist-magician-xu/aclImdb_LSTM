"""
加载数据集
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import re
import hyper_parameters
import numpy as np

# 清洗文本数据
def tokenlize(content):
    # 在 .!? 之前添加一个空格
    content = re.sub(r"([.!?])", r" \1", content)
    # 去除掉不是大小写字母及 .!? 符号的数据
    content = re.sub(r"[^a-zA-Z.!?]+", r" ", content)
    # 全部转换为小写，然后去除两边空格，将字符串转换成list,
    token = [i.strip().lower() for i in content.split()]
    # return ['word', 'word', ...]
    return token

# 字符， 数字 之间的转换
class transform():
    def __init__(self):
        self.PAD_TAG = "PAD"
        self.UNK = 0

    def trans(self, sentence, ws, max_len=None):
        if max_len is not None:    # 补齐，切割 句子固定长度
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        return [ws.get(word, self.UNK) for word in sentence]
    def inverse_trans(self, ws_inverse, indices):
        return [ws_inverse.get(idx) for idx in indices]
temp_transform = transform()

# 读取训练数据
class ImdbDataset(Dataset):
    def __init__(self, train_data_path, test_data_path, train=True):
        self.data_path = train_data_path if train else test_data_path
        # 获取所有文件的路径
        self.temp_data_path = [os.path.join(self.data_path, "pos"),
                               os.path.join(self.data_path, "neg")]
        self.total_file_path = []
        for path in self.temp_data_path:
            self.file_name_list = os.listdir(path)
            self.file_path_list = [os.path.join(path, file_name) for file_name in self.file_name_list
                                   if file_name.endswith("txt")]
            self.total_file_path.extend(self.file_path_list)

    def __getitem__(self, item):
        # 读取文件路径
        file_path = self.total_file_path[item]
        # 获取 label
        label_str = file_path.split("/")[-2]                      # 二分类问题，-1为文件名，-2为正负类别
        label = 0 if label_str == "neg" else 1                    # neg -> 0; pos -> 1
        token = tokenlize(open(file_path).read())                 # 获取 经过 token 处理以后的 内容
        ws = np.load("./model/aclImdb_trans.npy", allow_pickle=True).item()    # 加载的是 numpy.ndarray 的形式，加个item()返回元数据
        # ws = pickle.load(open("./model/aclImdb_ws.pkl", "rb"))    # 加载 ws 表
        token = temp_transform.trans(sentence=token, ws=ws, max_len=hyper_parameters.max_len)
        return np.array(token), np.array(label)                   # 返回 tuple (token, label)

    def __len__(self):
        return len(self.total_file_path)

# 建立 DataLoader
def get_dataloader(train=True):
    temp = ImdbDataset(hyper_parameters.train_data_path, hyper_parameters.test_data_path, train=train)
    data_loader = DataLoader(temp, batch_size=hyper_parameters.train_batch_size,
                             shuffle=hyper_parameters.train_shuffer)
    return data_loader




