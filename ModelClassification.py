#!/usr/bin/python3
'''
主要是加载模型来验证测试集?
'''
import torch
import glob
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import sys
import torch.nn.functional as F
from PIL import Image
from submit_result import write_submit
from CNNFramework import *
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_path = "dataset/test"  # 验证集的位置
mode_path = "./model/model.ckpt"  # 训练好的模型的文件位置
save_path = "./dataset/submit.csv"  # 生成需要提交的文件
write_path = "./result/result.csv"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ----------------------------------网络结构特征---------------------------------------
class ConvNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(24 * 24 * 32, num_classes)  # 注意参数的调整，神经网络的参数要保持相同结构?

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def default_loader(path):
    return Image.open(path).convert('RGB')


def get_all_path(path):  # 获取这个文件夹下面的所有文件?
    names = []
    paths = glob.glob(os.path.join(path, "*.tif"))
    for p in paths:
        name = p.split('.')[0]
        name = name.split('/')[-1]
        names.append(name)
    return paths, names


class MyDataLoader(Dataset):
    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        paths, names = get_all_path(path)
        self.paths = paths
        self.names = names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.paths[index]
        ids = self.names[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, ids

    def __len__(self):
        return self.paths.__len__()


def run():
    test_data = MyDataLoader(test_path, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)  # 标准数据集的构造?

    net_r = ConvNet(num_classes=num_classes).to(device)  # 保持和之前的神经网络相同的结构特征?
    net_r.load_state_dict(torch.load("model/model.ckpt"))
    names = []
    results = []

    # Test the model
    net_r.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for (images, ids) in test_loader:
            images = images.to(device)
            # print(ids)
            # images = images
            # labels = labels
            outputs = net_r(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            for r in predicted:
                results.append(r.cpu().numpy())
            for tmp in ids:
                names.append(tmp)
    fp = open(write_path, 'w', encoding="UTF-8")
    fp.write("id,label\n")
    for index in range(names.__len__()):
        result = names[index] + "," + str(results[index]) + "\n"
        fp.write(result)
    fp.close()
    print("Successfully Write!!!")
    write_submit()


run()
