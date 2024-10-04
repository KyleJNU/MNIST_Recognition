'''
1. 通过 torchvision.datasets.MNIST 下载、解压和读取 MNIST 数据集；
2. 使用 PIL.Image.save 将 MNIST 数据集中的灰度图片以 PNG 格式保存。
'''
import sys, os
from torchvision.datasets import MNIST
from tqdm import tqdm
sys.path.insert(0, os.getcwd())     # 将当前工作目录添加到模块搜索路径的开头

if __name__ == "__main__":

    # 图片保存路径
    root = 'mnist_images'                   # 定义保存图片的根目录
    if not os.path.exists(root):            # 如果根目录不存在
        os.makedirs(root)                   # 创建根目录

    # 训练集60K、测试集10K
    # torchvision.datasets.MNIST接口下载数据
    training_dataset = MNIST(               # 实例化torchvision.datasets.MNIST 类，加载MNIST数据集
        root='mnist',                       # 数据集将被下载到当前工作目录下的 mnist 文件夹中
        train=True,                         # 指定要下载的是训练集
        download=True,                      # 如果本地路径中没有找到数据集，则联网下载；如果数据集已经存在于指定的 root 目录中，则不会重新下载
    )
    test_dataset = MNIST(                   # 实例化torchvision.datasets.MNIST 类，加载MNIST数据集
        root='mnist',                       # 数据集将被下载到当前工作目录下的 mnist 文件夹中
        train=False,                        # 指定要下载的是测试集
        download=True,                      # 如果本地路径中没有找到数据集，则联网下载；如果数据集已经存在于指定的 root 目录中，则不会重新下载
    )

    # 保存训练集图片
    with tqdm(total=len(training_dataset), ncols=150) as pro_bar:   # 创建进度条，宽度为150个字符
        for idx, (X, y) in enumerate(training_dataset):             # 遍历训练集，enumerate函数为training_dataset的每个元素生成一个包含索引（idx）和元素本身（X,y）的元组，X代表图像数据，y则为对应标签
            # 创建目标文件夹
            train_dir = os.path.join(root, "train", str(y))         # 定义保存训练集图片的目录
            if not os.path.exists(train_dir):                       # 如果目录不存在
                os.makedirs(train_dir)                              # 创建目录
            f = os.path.join(train_dir, f"training_{y}_{idx}.png")  # 保存的文件名
            X.save(f)                                               # 保存图片，torchvision.datasets.MNIST默认将图像加载为PIL图像格式，.save() 是PIL库中图像对象的一个方法，用于将图像保存到文件
            pro_bar.update(n=1)                                     # 更新进度条

    # 保存测试集图片
    with tqdm(total=len(test_dataset), ncols=150) as pro_bar:       # 创建进度条，宽度为150个字符
        for idx, (X, y) in enumerate(test_dataset):                 # 遍历测试集，enumerate函数为training_dataset的每个元素生成一个包含索引（idx）和元素本身（X,y）的元组，X代表图像数据，y则为对应标签
            # 创建目标文件夹
            test_dir = os.path.join(root, "test", str(y))           # 定义保存测试集图片的目录
            if not os.path.exists(test_dir):                        # 如果目录不存在
                os.makedirs(test_dir)                               # 创建目录
            f = os.path.join(test_dir, f"test_{y}_{idx}.png")       # 保存的文件名
            X.save(f)                                               # 保存图片，torchvision.datasets.MNIST默认将图像加载为PIL图像格式，.save() 是PIL库中图像对象的一个方法，用于将图像保存到文件
            pro_bar.update(n=1)                                     # 更新进度条