from model import Network
from torchvision import datasets, transforms
import torch

if __name__ == '__main__':
    # 图像的预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),            # 转换为单通道灰度图
        transforms.ToTensor()                                   # 转换为张量
    ])

    # 读取测试数据集
    test_dataset = datasets.ImageFolder(root='./mnist_images/test', transform=transform)
    print("test_dataset length:", len(test_dataset))

    # 定义神经网络模型并加载训练好的模型文件
    model = Network()
    model.load_state_dict(torch.load('best_mnist.pth', weights_only=True))

    right = 0                                                   # 保存正确识别的数量
    for i, (x, y) in enumerate(test_dataset):
        output = model(x)                                       # 将其中的数据x输入到模型
        predict = output.argmax(1).item()                       # 选择概率最大的标签作为预测结果
        # 对比预测值predict和真实标签y
        if predict == y:
            right += 1
        # 将识别错误的样例打印出来
        else:
            img_path = test_dataset.samples[i][0]
            print(f"wrong case: predict = {predict} , truth = {y} , img_path = {img_path}")

    # 计算出测试结果
    sample_num = len(test_dataset)
    acc = right * 1.0 / sample_num
    print(f"test accuracy = %d / %d = %.3lf" % (right, sample_num, acc))