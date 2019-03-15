import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
from CNNFramework import *
from Resnet import *
from Drawing import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Hyper parameters


def default_loader(path):
    return Image.open(path).convert('RGB')


# 数据集的划分，将数据划分为训练集以及测试集
class DataSplit():
    def __init__(self, path, train_size):
        fh = open(path, 'r')  # 读取全部的文件
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        length = imgs.__len__()
        flag = int(length * train_size)  # 前面的flag作为训练集
        rand_list = np.random.randint(0, length, length)  # 获得随机数组
        train_image = []
        test_image = []
        for i in range(length):
            if i < flag:
                train_image.append(imgs[rand_list[i]])
            else:
                test_image.append(imgs[rand_list[i]])
        self.train_imgs = train_image
        self.test_imgs = test_image
        self.train_imgs_length = train_image.__len__()
        self.test_imgs_length = test_image.__len__()


class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, imgs, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def run():
    # data_transforms = transforms.Compose([
    #     # transforms.CenterCrop(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # data_transforms_test = transforms.Compose([
    #     # transforms.CenterCrop(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])   #图像风格图像
    start_time = time.time()
    imgs = DataSplit(path='./dataset/train_data_labels.txt', train_size=0.9)
    train_data = MyDataset(imgs.train_imgs, transform=transforms.ToTensor())  # 作为训练集
    test_data = MyDataset(imgs.test_imgs, transform=transforms.ToTensor())  # 作为测试集
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    def show_batch(imgs):
        grid = utils.make_grid(imgs)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')

    # for i, (batch_x, batch_y) in enumerate(data_loader):
    #     if(i<4):
    #         print(i, batch_x.size(),batch_y.size())
    #         show_batch(batch_x)
    #         plt.axis('off')
    #         plt.show()

    model = ResNet(num_classes=num_classes).to(device)
    print(model)
    # model = ConvNet(num_classes)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model

    total_step = len(train_loader)
    Acc_h = []  # 用于画图的数据记录
    Loss_h = []  # 用于画图的数据记录
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            correct = 0
            images = images.to(device)
            labels = labels.to(device)

            # images = images
            # labels = labels

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, prediction = torch.max(outputs.data, 1)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (prediction == labels).sum().item()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), correct / batch_size))
                Acc_h.append(correct / batch_size)
                Loss_h.append(loss.item())

    # Test the model
    Acc = 0.0
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # images = images
            # labels = labels
            outputs = model(images)  # 直接获得模型的结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(imgs.test_imgs_length,
                                                                              batch_size * correct / total))
        Acc = batch_size * correct / total

    # Save the model checkpoint
    timestamp = str(int(time.time()))
    name = str("./model/model-{}-{}-{:.4f}.ckpt".format(learning_rate, timestamp, Acc))
    torch.save(model.state_dict(), name)
    end_time = time.time()
    run_time = end_time - start_time
    print("Running Time {:.2f}".format(run_time))
    show_plt(Acc_h, Loss_h)


run()
