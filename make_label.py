import os
import pickle
import random

area_map = {"辽宁": 0, "吉林": 0, "黑龙江": 0,
            "河北": 1, "山西": 1, "内蒙古": 1, "北京": 1, "天津": 1,
            "山东": 2, "江苏": 2, "安徽": 2, "浙江": 2, "台湾": 2, "福建": 2, "江西": 2, "上海": 2,
            "河南": 3, "湖北": 3, "湖南": 3,
            "广东": 4, "广西": 4, "海南": 4, "香港": 4, "澳门": 4,
            "云南": 5, "重庆": 5, "贵州": 5, "四川": 5, "西藏": 5,
            "新疆": 6, "陕西": 6, "宁夏": 6, "青海": 6, "甘肃": 6}

area_train_uid = []
area_train_label = {}
train_uid = []
sex_train_label = {}
age_train_label = {}

with open("./data/train/train_labels.txt", "r", encoding="utf-8") as f:
    line = f.readline()
    while line:
        a = line[:-1].split("||")
        train_uid.append(a[0])
        if a[1] == 'm':
            sex_train_label[a[0]] = 1
        else:
            sex_train_label[a[0]] = 0
        if int(a[2]) <= 1979:
            age_train_label[a[0]] = 2
        elif 1980 <= int(a[2]) <= 1989:
            age_train_label[a[0]] = 1
        else:
            age_train_label[a[0]] = 0
        area = a[3].split(" ")[0]
        if not area == 'None':
            area_train_uid.append(a[0])
            if area not in area_map:
                area_train_label[a[0]] = 7
            else:
                area_train_label[a[0]] = area_map[area]
        line = f.readline()

test_uid = []
area_test_label = {}
sex_test_label = {}
age_test_label = {}

with open("./data/test/test_labels.txt", "r", encoding="utf-8") as f:
    line = f.readline()
    while line:
        a = line[:-1].split("||")
        test_uid.append(a[0])
        if a[1] == 'm':
            sex_test_label[a[0]] = 1
        else:
            sex_test_label[a[0]] = 0
        if int(a[2]) <= 1979:
            age_test_label[a[0]] = 2
        elif 1980 <= int(a[2]) <= 1989:
            age_test_label[a[0]] = 1
        else:
            age_test_label[a[0]] = 0
        area = a[3].split(" ")[0]
        if area not in area_map:
            area_test_label[a[0]] = 7
        else:
            area_test_label[a[0]] = area_map[area]
        line = f.readline()

random.seed(30)
dev_uid = random.sample(train_uid, int(len(train_uid) * 0.3))
train_uid = list(set(train_uid) - set(dev_uid))
area_dev_uid = random.sample(area_train_uid, int(len(area_train_uid) * 0.3))
area_train_uid = list(set(area_train_uid) - set(area_dev_uid))