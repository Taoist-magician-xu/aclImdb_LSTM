"""
定义超参数
"""

train_data_path = "./data/aclImdb/train"    # 训练集
test_data_path  = "./data/aclImdb/train"    # 测试集

max_len            = 200                      # 对齐每个句子的长度
train_batch_size   = 1280                     # 训练 batch_size
test_batch_size    = 1000                     # 测试 batch_size
train_shuffer      = True                     # 训练进行 shuffer
test_shuffer       = False                    # 测试不进行 shuffer
max_features       = 20000                    # 词表的长度
embedding_size     = 100                      # embeeding 的大小
lstm_hidden_size   = 128                      # lstm 的单元数
lstm_num_layer     = 2                        # lstm 的层数
bidriectional      = True                     # lstm 是否为双向
lstm_dropout       = 0.5                      # 随机 dropout
lstm_out_hidden    = 10                       # 输出层LSTM的单元数
lstm_out_num       = 1                        # 输出层LSTM的层数
lstm_bidriectional = False                    # 输出层LSTM为单向
lstm_out_dropout   = 0.5                      # 输出层LSTM的dropout
epoch              = 100                      # 训练迭代次数
train              = True                     # 训练模式
test               = False                    # 测试模式
output_class       = 2                        # 判别类别
learning_rate      = 0.001                    # 学习率



