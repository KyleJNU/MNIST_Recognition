# MNIST_Recognition
简单的MNIST手写数字识别，B站 https://www.bilibili.com/video/BV134421U77t “从零设计并训练一个神经网络，你就能真正理解它了”对应的手打代码，包含数据集下载脚本，有详细注释，适合新手拿来学习和理解。
# 下载数据集
首先运行download_data.py，下载MNIST数据集并转为png格式，png数据集会自动保存在项目目录下的mnist_images文件夹中。
# 训练
运行train.py，最佳权重会保存在项目目录下，为.pth文件。
# 测试
运行test.py，会将识别错误图片的预测结果、真实标签以及对应的文件名打印出来。
