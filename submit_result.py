#!/usr/bin/python

import pandas as pd
import os

path_result = "./result/result.csv"
path_sample_submit = "./result/sample_submission.csv"  # 需要按照sample的顺序提交
save_submit = "./submit/submit_result.csv"


def dic_result():  # 获得字典类型
    data = pd.read_csv(path_result)
    ids = data['id']
    labels = data['label']
    dict_r = dict(zip(ids, labels))
    return dict_r


def write_submit(save_path=save_submit, path_sample_submit=path_sample_submit):
    w_ids = pd.read_csv(path_sample_submit, sep=',')
    w_ids = list(w_ids["id"])
    data_dict = dic_result()
    fp = open(save_path, 'w', encoding="UTF-8")
    fp.write("id,label\n")
    for index in range(w_ids.__len__()):
        l = data_dict[w_ids[index]]
        r = w_ids[index] + "," + str(l) + "\n"
        fp.write(r)
    fp.close()
    print("Successfully Write!!!")
