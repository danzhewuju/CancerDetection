#!/usr/bin/python3
'''
主要是加载模型来验证测试集。
'''


import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_path = "./dataset/train"  #验证集的位置
mode_path = "./model/model.ckpt"  #训练好的模型的文件位置
from PIL import Image

batch_size = 100

image_path = "dataset/train/0a1779a202e1a35eca405720fe35966dbde59b4c.tif"

transform1 = transforms.Compose([transforms.ToTensor()])


def read_img():
    img = Image.open(image_path).convert("RGB")
    img = transform1(img)
    return img


def testing(test_path, mode_path):
    # Validation_data = DataLoader(test_path, shuffle=True, batch_size=batch_size)
    model = torch.load(mode_path, map_location="cpu")
    model.eval()
    img = read_img()
    output = model(img)
    _, pre = torch.max(output.data, 1)
    print(pre)
    return pre
    # model.eval()
    # for images, _ in Validation_data:
    #     # images = images.to(device)
    #     outputs = model(images)
    #     _, predicted = torch.max(outputs.data, 1)
    #     print(predicted)


testing(test_path, mode_path)