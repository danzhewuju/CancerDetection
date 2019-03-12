#!/usr/bin/Python3
'''
根据时间戳来自动的加载模型加载最新的模型
'''
import os
import re

path_model_dir = "model"


def get_info(path):
    names = []
    paths = []
    for root, dir, files in os.walk(path):
        for f in files:
            names.append(f)
            p = os.path.join(root, f)
            paths.append(p)
    return names, paths


def get_best_model(path):  # 获取最好的模型
    names, paths = get_info(path)
    names_paths = dict(zip(names, paths))
    accs = []
    for na in names:
        result = re.findall(r"-(.*).c", na)[0]
        result = result.split("-")[-1]
        accs.append(float(result))
    name_accs = dict(zip(names, accs))
    name_accs = sorted(name_accs.items(), key=lambda x: -x[1])
    top_name = name_accs[0]
    top_name = top_name[0]
    path = names_paths[top_name]
    print("Loading model path:{}".format(path))
    return path


def get_new_model(path):  # 获得的是最新的模型
    names, paths = get_info(path)
    names_paths = dict(zip(names, paths))
    accs = []
    for na in names:
        result = re.findall(r"-(.*).c", na)[0]
        result = result.split('-')[1]
        accs.append(float(result))
    name_accs = dict(zip(names, accs))
    name_accs = sorted(name_accs.items(), key=lambda x: -x[1])
    top_name = name_accs[0]
    top_name = top_name[0]
    path = names_paths[top_name]
    print("Loading model path:{}".format(path))
    return path


# print(get_new_model(path_model_dir))


