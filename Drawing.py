#!/usr/bin/python
'''
主要是画图的工具类
'''

import matplotlib.pyplot as plt
import numpy as np
import time

path = "./Drawing/"  # 将最后的散点图保存的位置


def show_plt(accuracy, loss):  # 画出accuracy,loss的趋势图
    time_stamp = time.time()
    time_struct = time.localtime(time_stamp)
    time_stamp = "Accuracy-Loss%d-%d-%d %d:%d:%d" % (time_struct[0], time_struct[1], time_struct[2],
                                                     time_struct[3], time_struct[4], time_struct[5])
    name = path + time_stamp + ".png"
    acc = np.asarray(accuracy)
    loss = np.asarray(loss)
    plt.title("Accuracy&&Loss")
    plt.xlabel("epoch")
    plt.ylabel("Acc/loss")
    plt.plot(acc, label="Accuracy")
    plt.plot(loss, label="Loss")
    plt.legend(loc='upper right')
    plt.savefig(name)
    plt.show()

    print("Saved Image!")
    return True

