#!/usr/bin/python3
'''
主要是加载模型来验证测试集。
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
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_path = "dataset/test"  #验证集的位置
mode_path = "./model/model.ckpt"  #训练好的模型的文件位置
batch_size = 100
num_classes = 2
save_path = "./dataset/submit.csv"#生成需要提交的文件
write_path = "./result.csv"


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#----------------------------------网络结构特征---------------------------------------
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
        self.fc = nn.Linear(24 * 24 * 32, num_classes)  #注意参数的调整，神经网络的参数要保持一致
     
    def forward(self, x):
        # x = x.view(x.size(0), -1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def default_loader(path):
    return Image.open(path).convert('RGB')


def get_all_path(path):  #获取这个文件夹下面的所有文件
    paths = glob.glob(os.path.join(path, "*.tif"))
    return paths



def get_name_list(path):# 获取全部的名字列表
    names = []
    for root, dir, files in os.walk(path):
        for f in files:
            n = f.split('.')[0]
            names.append(n)
    return names



class MyDataLoader(Dataset):
    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        paths = get_all_path(path)
        self.paths = paths
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.paths[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.paths.__len__()


def run():
    test_data = MyDataLoader(test_path, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)  # 标准数据集的构造

    net_r = ConvNet(num_classes=num_classes).to(device)  # 保持和之前的神经网络相同的结构
    net_r.load_state_dict(torch.load("model/model.ckpt"))
    names = get_name_list(test_path)
    results = []

    # Test the model
    net_r.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            # images = images
            # labels = labels
            outputs = net_r(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            for r in predicted:
                results.append(r)
    fp = open(write_path, 'w', encoding="UTF-8")
    fp.write("id,label\n")
    for index in names.__len__():
        result = names[index] + "," + str(result[index])+"\n"
        fp.write(result)
    fp.close()
    print("Successfully Write!!!")



run()







