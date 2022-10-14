# 使用pytorch 完成手写数字的识别
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

BATCH_SIZE = 128


# 1. 准备数据集
def get_dataloader(train=True, download_=False):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))    # mean and std 形状与通道数相同
    ])
    dataset = MNIST(root="./data", train=train, download=download_,
                    transform=transform_fn)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    return data_loader

# 2. 构建模型
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28*1)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)

        return output

# 3. 损失函数
def get_loss(output, target):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    return loss

# 4. 训练模型
model = MnistNet()
optimizer = Adam(model.parameters(), lr=0.001)

if os.path.exists("./model/model.pkl"):
    model.load_state_dict(torch.load("./model/model.pkl"))
    optimizer.load_state_dict(torch.load("./results/optimizer.pkl"))

def train(epoch):
    data_loader = get_dataloader()
    for idx, (input, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(input)
        loss = get_loss(output, target)
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(epoch, idx, loss.item())

        if idx % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./results/optimizer.pkl")

def test():
    test_loss = []
    correct = []
    model.eval()    # 设置模型为评估模式
    test_dataloader = get_dataloader(train=False, download_=True)
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):
            output = model(data)
            cur_loss = get_loss(output, target).item()
            test_loss.append(cur_loss)
            pred = output.max(dim=-1)[-1]    # [-1]表示获取最大值的位置
            cur_acc = pred.eq(target).float().mean()
            correct.append(cur_acc)
    print("平均准确率，平均损失：", np.mean(correct), np.mean(test_loss))




if __name__ == '__main__':
    test()


