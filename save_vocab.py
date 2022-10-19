"""
建立词表，实现字符与数字之间的相互转换并保存词表
利用 训练集 中的数据建立词表，跟测试集没有关系
"""
import os
from tqdm import tqdm
from hyper_parameters import train_data_path
from load_dataset import tokenlize
import numpy as np
from hyper_parameters import max_features

class Word2Sequence:
    UNK_TAG = "UNK"    # 遇到未知字符，用UNK表示
    PAD_TAG = "PAD"    # 用PAD补全句子长度
    UNK     = 0        # UNK字符对应的数字
    PAD     = 1        # PAD字符对应的数字

    def __init__(self):
        self.dict = {                   # 用 dict 表示最终要保存的词表
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}                 # 统计词频

    # 统计一个句子中的词频
    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1    # 统计完一个句子，self.count 是不清空的

    # 建立词表
    def build_vocab(self, min=0, max=None, max_features=None):
        if min is not None:
            self.count = {word: value for word, value in self.count.items()
                          if value>min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items()
                          if value<max}

        # 根据词频排序，选择最大的 max_features 个
        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x:x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        # 建立词表
        for word in self.count:
            self.dict[word] = len(self.dict)    # 如果没有经过 max_features, 无序的

        # 翻转词表：数字->字符
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

        return self.dict, self.inverse_dict

    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    ws = Word2Sequence()
    path = train_data_path
    temp_data_path = [os.path.join(path, "pos"),
                      os.path.join(path, "neg")]    # 训练集中包含 正类数据pos 负类数据neg
    for data_path in temp_data_path:
        file_paths = [os.path.join(data_path, file_name)
                      for file_name in os.listdir(data_path)
                      if file_name.endswith("txt")]
        for file_path in tqdm(file_paths):
            sentence = tokenlize(open(file_path).read())    # 逐条文件路径打开文件读取，并且使用使用 tokenlize 清晰句子
            ws.fit(sentence)                                # 清洗完的句子放入 fit , 统计词频，count不会更新

    # 所有文件的句子都进行统计词频以后，建立词表 dict
    trans, inverse_trans = ws.build_vocab(max_features=(max_features-2))                      # 假设 使用 词频数最大的 20000 个词
    # 保存词表
    np.save("./model/aclImdb_trans.npy", trans)             # 保存为 numpy.ndarray 的形式
    np.save("./model/aclImdb_intrans.npy", inverse_trans)
    # pickle.dump(ws, open("./model/aclImdb_ws.pkl", "wb"))
    print(len(ws))                                          # 观察下词表的维度是否正确

