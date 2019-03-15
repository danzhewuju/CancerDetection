#!/usr/bin/python
'''
主要是画图的工具类
'''

import matplotlib.pyplot as plt
import numpy as np


def show_plt(accuracy, loss):  # 画出accuracy,loss的趋势图
    acc = np.asarray(accuracy)
    loss = np.asarray(loss)
    plt.xlabel("epoch")
    plt.ylabel("Acc/loss")
    plt.plot(acc, label="Accuracy")
    plt.plot(loss, label="Loss")
    plt.show()



