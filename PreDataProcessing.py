#!/usr/bin/python
import pandas as pd
import numpy as np

import torchvision
import torch
import torchvision.transforms as transforms
import os

path_label = "dataset/train_labels.csv"


def get_lables(path):                          #获取训练图片数据的labels
    data = pd.read_csv(path, sep=',')
    id_list = data['id']
    labels_list = data['label']
    dict_labels = dict(zip(id_list, labels_list))
    return dict_labels                         #返回的是dict_labels 的映射关系


#图片的读取
def read_img(path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = torchvision.datasets.ImageFolder(root=path, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True, num_workers=2)

    dataiter = iter(loader)
    images, labels = dataiter.next()

    return images, labels


def write_image_labels(path_write, path_labels):        #注意pytorch数据的加载方式，我们需要写成  “****.png 1” 这样的格式
    index = 0
    dict_labels = get_lables(path_labels)
    f = open(path_write, 'w', encoding="UTF-8")
    for d in dict_labels.items():
        name = '.'.join([d[0], "tif"])
        name_pre = "dataset/train"
        path_name = os.path.join(name_pre, name)  #获得的完整路径
        label = d[1]
        result = path_name+" "+str(label)+"\n"
        f.write(result)
        index += 1
        if index % 100 == 0:
            print("Processing %d data" % index)
    f.close()
    return True


# id_t = "f38a6374c348f90b587e046aac6079959adf3835"
# data = get_lables(path_label)
# print(data[id_t])
# path_image = "dataset/train"
# read_img(path_image)
path_labels="dataset/train_labels.csv"
path_write="dataset/train_data_labels.txt"
write_image_labels(path_write, path_labels)

