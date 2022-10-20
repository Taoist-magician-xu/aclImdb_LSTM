# aclImdb_LSTM
dataset: aclImdb; model: biLSTM_TwoLayer+LSTM

1. 下载文件
  git clone https://github.com/Taoist-magician-xu/aclImdb_LSTM.git
2. 建立 data 文件夹
  mkdir data
3. 将 aclImdb_v1.tar.gz 文件移动到 data 文件夹下
  mv aclImdb_v1.tar.gz ./data/
4. 进入 data 文件夹 并解压 aclImdb_v1.tar.gz 文件
  cd data & tar -xvzf aclImdb_v1.tar.gz
5. 返回主文件夹
  cd ~
6. 建立 model 文件夹 用于保存 模型文件 和 词表
  mkdir model
7. 运行 save_vocab.py 文件，建立并保存词表
  python save_vocab.pu
8. 运行 train.py 文件
  python train.py
9. end

运行的速度属实有点慢
