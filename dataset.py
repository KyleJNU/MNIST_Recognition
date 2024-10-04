from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 实现图像的预处理pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),    # 转换为单通道灰度图
        transforms.ToTensor()   # 转换为张量
    ])

    # 使用ImageFolder函数，读取数据文件夹，并保存数据文件夹的名字作为数据的标签，构建数据集dataset
    # 例如，对于名字为“3”的文件夹，会将“3”作为文件夹中图像数据的标签和图像配对，用于后续的训练
    train_dataset = datasets.ImageFolder(root='./mnist_images/train', transform=transform)
    test_dataset = datasets.ImageFolder(root='./mnist_images/test', transform=transform)
    # 打印它们的长度
    print("train_dataset length:", len(train_dataset))
    print("test_dataset length:", len(test_dataset))

    # 使用train_loader，实现小批量的数据读取
    # 这里设置小批量的大小，batch_size=64。也就是每个批次处理64个数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    # 打印train_loader的长度
    # 60000个训练数据，如果每次读入64个数据，那么这60000个数据会被分成938组
    print("train_loader length:", len(train_loader))