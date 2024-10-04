import torch
from torch import nn
from torch import optim
from model import Network
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 图像的预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),        # 转换为单通道灰度图
        transforms.ToTensor()                               # 转换为张量
    ])

    # 读入并构造数据集
    train_dataset = datasets.ImageFolder(root='./mnist_images/train', transform=transform)
    print("train_dataset length:", len(train_dataset))      # 训练集总数

    # 每次读入小批量数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    print("train_loader length:", len(train_loader))        # 将训练集全部遍历一遍需要的批次数

    # 在使用pytorch训练模型时，需要创建三个对象：
    model = Network()                                       # 1.模型本身，即我们设计的神经网络
    optimizer = optim.Adam(model.parameters())              # 2.优化器，优化模型中的参数
    criterion = nn.CrossEntropyLoss()                       # 3.损失函数，此处为分类问题，使用交叉熵损失误差

    # 初始化最低损失值为无穷大
    best_loss = float('inf')

    # 迭代次数
    epochs = 10

    # 进入模型的循环迭代
    # 外层循环，代表整个训练数据集的遍历次数
    for epoch in range(epochs):
        for batch_idx, (data, label) in enumerate(train_loader):
            # 内层循环使用train_loader，每批次处理64个数据
            # 内层每循环一次，就会进行一次梯度下降算法
            # 包含5个步骤
            output = model(data)                            # 1.计算神经网络前向传播的结果
            loss = criterion(output, label)                 # 2.计算output和标签label之间的损失loss
            loss.backward()                                 # 3.使用backward计算梯度
            optimizer.step()                                # 4.使用optimizer.step更新参数
            optimizer.zero_grad()                           # 5.将梯度清零

            # 每迭代100个小批量，就打印一次模型的损失，观察训练的过程
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss {loss.item():.4f}")

        # 每个epoch结束后，检查模型的loss值是否最低，是则保存为best_model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_mnist.pth')

    print(f"Best loss: {best_loss:.4f}")