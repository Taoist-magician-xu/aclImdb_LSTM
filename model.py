"""
建立模型，使用双向两层 LSTM，然后把输出传入第三层 LSTM
"""
import torch.nn as nn
import torch.nn.functional as F

import hyper_parameters

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(hyper_parameters.max_features, hyper_parameters.embedding_size)
        self.lstm      = nn.LSTM(input_size=hyper_parameters.embedding_size,
                                 hidden_size=hyper_parameters.lstm_hidden_size,
                                 num_layers=hyper_parameters.lstm_num_layer,
                                 batch_first=True,
                                 bidirectional=hyper_parameters.bidriectional,
                                 dropout=hyper_parameters.lstm_dropout)
        self.lstm_out = nn.LSTM(input_size=hyper_parameters.lstm_hidden_size*2,
                                hidden_size=hyper_parameters.lstm_out_hidden,
                                num_layers=hyper_parameters.lstm_out_num,
                                batch_first=True,
                                bidirectional=hyper_parameters.lstm_bidriectional)
                                # dropout=hyper_parameters.lstm_out_dropout 单词 RNN 中不允许添加dropout
        self.fc = nn.Linear(hyper_parameters.lstm_out_hidden, hyper_parameters.output_class)

    def forward(self, input):
        x = self.embedding(input)        # [batch_size, max_len] -> [bathc_size, max_len, embedding_size]
        x, (h_n, c_n) = self.lstm(x)     # out: [batch_size, max_len, hidden_size*2] h_n: [2*2, batch_size, hidden_size]
        x_out, (h_out_n, c_out_n) = self.lstm_out(x)     # out: [batch_size, max_len, hidden_size]
        # 获取最后一次输出                                  # h_n [1, batch_size, hidden_size]
        output = h_out_n[0]                              # [batch_size, hidden_size]
        out = self.fc(output)
        return F.log_softmax(out, dim=-1)




