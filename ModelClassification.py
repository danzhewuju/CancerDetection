#!/usr/bin/python3
'''
主要是加载模型来验证测试集
'''
import torch
import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from submit_result import write_submit
from ModelLoad import *
import time
from model_cnn import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_path = "dataset/test"  # 验证集的位置
mode_path = get_new_model("model")  # 获得最新的模板
# mode_path = "model/model-0.0005-1552599033-98.0185.ckpt"  # 训练好的模型的文件位置
save_path = "./dataset/submit.csv"  # 生成需要提交的文件
write_path = "./result/result.csv"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ----------------------------------网络结构特征---------------------------------------


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
    start_time = time.time()  # 图像风格图像
    test_data = MyDataLoader(test_path, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)  # 标准数据集的构造?

    net_r = VGGNet(num_classes=num_classes).to(device)  # 保持和之前的神经网络相同的结构特征?
    net_r.load_state_dict(torch.load(mode_path))
    print("Loading {} model!".format(mode_path))
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
            re = outputs.data[:, 1].cpu().numpy()
            # _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
            print(re)
            for r in re:
                results.append(r)
                # results.append(r.cpu())
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
    end_time = time.time()
    run_time = end_time - start_time
    print("Running Time {:.2f}".format(run_time))


run()  # 程序运行
