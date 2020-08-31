import numpy as np
import os
import copy
import time
import sys
import math

# mobilenet 9 blocks
MOBILE_TBS = [[3, 1], [3, 3], [3, 6], [5, 1], [5, 3], [5, 6], [7, 1], [7, 3], [7, 6]]
MOBILE_TYPICAL_CHOICE = [1] * 12
# VGGnet 13 blocks
VGG_TBS = [
    [1],
    [3],
    [1, 3],
    [5],
    [1, 5],
    [3, 3],
    [1, 3, 3],
    [7],
    [1, 7],
    [3, 5],
    [1, 3, 5],
    [3, 3, 3],
    [1, 3, 3, 3],
]
VGG_TYPICAL_CHOICE = [5] * 12
# Resnet 9 blocks
RES_TBS = [[1, 3], [1, 5], [1, 7], [2, 3], [2, 5], [2, 7], [4, 3], [4, 5], [4, 7]]
RES_TYPICAL_CHOICE = [0] * 12

# Cost function parameters
thres = 0.80
scale = 2
lamb = 0.01
block_percentage = 0.5

# 外层循环为layer，内层循环为block，当block == typical_block，则跳过
# CHOICE_TOTAL = [TYPICAL_CHOICE]
def generate_total_choice(tbs_list, typical_choice):
    total_choice = [typical_choice]
    for i in range(len(typical_choice)):
        for j in range(len(tbs_list)):
            if j == typical_choice[i]:
                continue
            tmp = copy.deepcopy(typical_choice)
            tmp[i] = j
            total_choice.append(tmp)
    return total_choice


def load_accuracy(accuracy_file, tbs_list, typical_choice, total_choice):
    layer_num = len(typical_choice)
    block_num = len(tbs_list)
    accuracy = np.loadtxt(accuracy_file, dtype=np.str)
    assert accuracy.shape[0] == len(total_choice)
    accuracy_array = np.zeros((layer_num, block_num), dtype=np.float32)
    for i in range(layer_num):
        for j in range(block_num):
            cur_choice = copy.deepcopy(typical_choice)
            cur_choice[i] = j
            dir_path = "_".join([str(v) for v in cur_choice])
            dir_path = dir_path + "_deploy.prototxt"
            # print(dir_path)
            index_1, index_2 = np.where(accuracy == dir_path)
            assert len(index_1) == 1
            assert index_2[0] == 0
            index = index_1[0]
            # get accuracy
            accuracy_array[i, j] = accuracy[index, 1]
            # print(accuracy_array[i, j])
    return accuracy_array


def load_latency(latency_file, tbs_list, typical_choice, total_choice):
    layer_num = len(typical_choice)
    block_num = len(tbs_list)
    latency = np.loadtxt(latency_file, dtype=np.str)
    assert latency.shape[0] == len(total_choice)
    latency_array = np.zeros((layer_num, block_num), dtype=np.float32)
    for i in range(layer_num):
        for j in range(block_num):
            cur_choice = copy.deepcopy(typical_choice)
            cur_choice[i] = j
            dir_path = "_".join([str(v) for v in cur_choice])
            dir_path = dir_path + "_deploy.txt"
            # print(dir_path)
            index_1, index_2 = np.where(latency == dir_path)
            assert len(index_1) == 1
            assert index_2[0] == 0
            index = index_1[0]
            # get latency
            latency_array[i, j] = latency[index, 1]
            # print(latency_array[i, j])
    return latency_array


def process_final_lat(lat_array, tbs_list, typical_choice, typical_choice_latency):
    layer_num = len(typical_choice)
    block_num = len(tbs_list)
    final_lat_array = np.zeros((layer_num, block_num), dtype=np.float32)
    for i in range(layer_num):
        for j in range(block_num):
            if j == typical_choice[i]:
                final_lat_array[i, j] = typical_choice_latency[i]
                continue
            lat_temp = lat_array[i, j] - lat_array[i, typical_choice[i]]
            final_lat_array[i, j] = typical_choice_latency[i] + lat_temp
    return final_lat_array


def compute_acc_cost(accuracy_array):
    [rows, cols] = accuracy_array.shape
    cost_array = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            cost_array[i, j] = math.exp(-(accuracy_array[i, j] - thres) / scale)
    return cost_array


def compute_lat_cost(latency_array):
    [rows, cols] = latency_array.shape
    cost_array = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            cost_array[i, j] = -1 / latency_array[i, j]
    return cost_array


def genereta_ss(id_list, lat_array, type, tbs_list, typical_choice):
    sort_id_list = np.sort(id_list)
    layer_num = len(typical_choice)
    block_num = len(tbs_list)
    block_list = []
    block_lat_list = []
    last_layer_id = 0
    tmp_list = []
    lat_list = []
    layer = []
    for id in sort_id_list:
        layer_id = int(id / block_num)
        block_id = int(id % block_num)
        if layer_id == last_layer_id:
            if layer_id == 0 and len(layer) == 0:
                layer.append(0)
            block_name = type + "block_" + str(block_id)
            tmp_list.append(block_name)
            lat_list.append(lat_array[layer_id, block_id])
        else:
            layer.append(layer_id)
            miss_layer_num = layer_id - last_layer_id
            block_list.append(tmp_list)
            block_lat_list.append(lat_list)
            tmp_list = []
            lat_list = []
            if miss_layer_num > 1:
                for i in range(miss_layer_num - 1):
                    block_list.append([type + "block_" + str(typical_choice[0])])
                    block_lat_list.append([lat_array[0, typical_choice[0]]])
            last_layer_id = layer_id
            block_name = type + "block_" + str(block_id)
            tmp_list.append(block_name)
            lat_list.append(lat_array[layer_id, block_id])
    block_list.append(tmp_list)
    block_lat_list.append(lat_list)
    return block_list, block_lat_list, layer


def get_ss(vgg_cost, res_cost, mobile_cost):
    vgg_array = vgg_cost.flatten()
    vgg_len = len(vgg_array)
    print("vgg_len = ", vgg_len)
    res_array = res_cost.flatten()
    res_len = len(res_array)
    print("res_len = ", res_len)
    mobile_array = mobile_cost.flatten()
    mobile_len = len(mobile_array)
    print("mobile_len = ", mobile_len)
    total_len = vgg_len + res_len + mobile_len
    print("total_len = ", total_len)
    cost_array = np.concatenate((vgg_array, res_array, mobile_array))
    sort_id = np.argsort(cost_array)
    # vgg_thres = math.ceil(vgg_len * block_percentage)
    # res_thres = math.ceil(res_len * block_percentage)
    # mobile_thres = math.ceil(mobile_len * block_percentage)
    vgg_thres = 60
    res_thres = 60
    mobile_thres = 60
    print("vgg_thres = ", vgg_thres, " res/mobile_thres =", res_thres)
    vgg_cnt = 0
    vgg_id = []
    res_cnt = 0
    res_id = []
    mobile_cnt = 0
    mobile_id = []
    for id in sort_id:
        if id >= 0 and id < vgg_len:
            vgg_cnt = vgg_cnt + 1
            vgg_id.append(id)
        elif id >= vgg_len and id < (vgg_len + res_len):
            res_cnt = res_cnt + 1
            res_id.append(id - vgg_len)
        else:
            mobile_cnt = mobile_cnt + 1
            mobile_id.append(id - vgg_len - res_len)
        if vgg_cnt >= vgg_thres:
            print(vgg_cnt, res_cnt, mobile_cnt)
            return vgg_id, "VGG"
        elif res_cnt >= res_thres:
            print(vgg_cnt, res_cnt, mobile_cnt)
            return res_id, "Res"
        elif mobile_cnt >= mobile_thres:
            print(vgg_cnt, res_cnt, mobile_cnt)
            return mobile_id, "Mobile"


# ResNet
res_total_choice = generate_total_choice(RES_TBS, RES_TYPICAL_CHOICE)
res_acc = load_accuracy(
    "./perf/res-12l-imgnet-acc.txt", RES_TBS, RES_TYPICAL_CHOICE, res_total_choice
)
# res_acc = load_accuracy('./perf/resnet-10l-cifar10-80ep-acc.txt', RES_TBS, RES_TYPICAL_CHOICE, res_total_choice)
res_acc_mean = np.mean(res_acc)

print("res_acc_mean = ", res_acc_mean)
res_lat = load_latency(
    "./perf/res-12l-imgnet-lat.txt", RES_TBS, RES_TYPICAL_CHOICE, res_total_choice
)
# res_lat = load_latency('./perf/res-10l-lat.txt', RES_TBS, RES_TYPICAL_CHOICE, res_total_choice)
res_lat_mean = np.mean(res_lat)
print("res_lat_mean = ", res_lat_mean)

# VGGNet
vgg_total_choice = generate_total_choice(VGG_TBS, VGG_TYPICAL_CHOICE)
vgg_acc = load_accuracy(
    "./perf/vgg-12l-imgnet-acc.txt", VGG_TBS, VGG_TYPICAL_CHOICE, vgg_total_choice
)
# vgg_acc = load_accuracy('./perf/VGG-10l-cifar10-80ep-acc.txt', VGG_TBS, VGG_TYPICAL_CHOICE, vgg_total_choice)
vgg_acc_mean = np.mean(vgg_acc)
print("vgg_acc_mean = ", vgg_acc_mean)
vgg_lat = load_latency(
    "./perf/vgg-12l-imgnet-lat.txt", VGG_TBS, VGG_TYPICAL_CHOICE, vgg_total_choice
)
# vgg_lat = load_latency('./perf/vgg-10l-lat.txt', VGG_TBS, VGG_TYPICAL_CHOICE, vgg_total_choice)
vgg_lat_mean = np.mean(vgg_lat)
print("vgg_lat_mean = ", vgg_lat_mean)

# MobileNet
mobile_total_choice = generate_total_choice(MOBILE_TBS, MOBILE_TYPICAL_CHOICE)
mobile_acc = load_accuracy(
    "./perf/mobile-12l-imgnet-acc.txt",
    MOBILE_TBS,
    MOBILE_TYPICAL_CHOICE,
    mobile_total_choice,
)
# mobile_acc = load_accuracy('./perf/mobile-10l-cifar10-80ep-acc.txt', MOBILE_TBS, MOBILE_TYPICAL_CHOICE, mobile_total_choice)
mobile_acc_mean = np.mean(mobile_acc)
print("mobile_acc_mean = ", mobile_acc_mean)
mobile_lat = load_latency(
    "./perf/mobile-12l-imgnet-lat.txt",
    MOBILE_TBS,
    MOBILE_TYPICAL_CHOICE,
    mobile_total_choice,
)
# mobile_lat = load_latency('./perf/mobile-10l-lat.txt', MOBILE_TBS, MOBILE_TYPICAL_CHOICE, mobile_total_choice)
mobile_lat_mean = np.mean(mobile_lat)
print("mobile_lat_mean = ", mobile_lat_mean)

res_acc_cost = compute_acc_cost(res_acc)
print("res_acc_cost\n")
print(res_acc_cost)
res_lat_cost = compute_lat_cost(res_lat)
print("res_lat_cost\n")
print(res_lat_cost)
res_cost = res_acc_cost + lamb * res_lat_cost
print("res_cost:\n")
print(res_cost)
# res_typical_lat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
res_typical_lat = [
    0.71,
    0.347,
    0.298,
    0.191,
    0.169,
    0.147,
    0.164,
    0.187,
    0.244,
    0.224,
    0.278,
    0.354,
]
# res_typical_lat = [0.103, 0.089, 0.086, 0.083, 0.119, 0.093, 0.104, 0.131, 0.161, 0.206]
res_final_lat = process_final_lat(res_lat, RES_TBS, RES_TYPICAL_CHOICE, res_typical_lat)
print("res_final_lat:\n")
print(res_final_lat)

vgg_acc_cost = compute_acc_cost(vgg_acc)
print("vgg_acc_cost\n")
print(vgg_acc_cost)
vgg_lat_cost = compute_lat_cost(vgg_lat)
print("vgg_lat_cost\n")
print(vgg_lat_cost)
vgg_cost = vgg_acc_cost + lamb * vgg_lat_cost
# vgg_typical_lat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print("vgg_cost:\n")
print(vgg_cost)
vgg_typical_lat = [
    0.228,
    0.323,
    0.122,
    0.124,
    0.065,
    0.116,
    0.068,
    0.109,
    0.115,
    0.218,
    0.115,
    0.255,
]
# vgg_typical_lat = [0.041, 0.047, 0.037, 0.039, 0.039, 0.048, 0.045, 0.073, 0.072, 0.137]
vgg_final_lat = process_final_lat(vgg_lat, VGG_TBS, VGG_TYPICAL_CHOICE, vgg_typical_lat)
print("vgg_final_lat:\n")
print(vgg_final_lat)


mobile_acc_cost = compute_acc_cost(mobile_acc)
print("mobile_acc_cost\n")
print(mobile_acc_cost)
mobile_lat_cost = compute_lat_cost(mobile_lat)
print("mobile_lat_cost\n")
print(mobile_lat_cost)
mobile_cost = mobile_acc_cost + lamb * mobile_lat_cost
# mobile_typical_lat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print("mobile_cost")
print(mobile_cost)
mobile_typical_lat = [
    0.71,
    0.321,
    0.327,
    0.114,
    0.140,
    0.08,
    0.121,
    0.126,
    0.193,
    0.148,
    0.221,
    0.239,
]
# mobile_typical_lat = [0.064, 0.048, 0.053, 0.045, 0.051, 0.053, 0.075, 0.089, 0.123, 0.152]
mobile_final_lat = process_final_lat(
    mobile_lat, MOBILE_TBS, MOBILE_TYPICAL_CHOICE, mobile_typical_lat
)
print("mobile_final_lat\n")
print(mobile_final_lat)


ss_id, target = get_ss(vgg_cost, res_cost, mobile_cost)
print(ss_id)
print("target = ", target)
if target == "VGG":
    block_list, block_lat_list, layer = genereta_ss(
        ss_id, vgg_final_lat, target, VGG_TBS, VGG_TYPICAL_CHOICE
    )
elif target == "Res":
    block_list, block_lat_list, layer = genereta_ss(
        ss_id, res_final_lat, target, RES_TBS, RES_TYPICAL_CHOICE
    )
else:
    block_list, block_lat_list, layer = genereta_ss(
        ss_id, mobile_final_lat, target, MOBILE_TBS, MOBILE_TYPICAL_CHOICE
    )
print(layer)
f = open("block_choice.txt", "w")
for layer_block in block_list:
    print(layer_block)
    layer_choice = ",".join([str(v) for v in layer_block])
    f.write("[" + layer_choice + "],")
f.close()
f = open("layer_latency.txt", "w")
for layer_lat in block_lat_list:
    print(layer_lat)
    layer_choice_lat = " ".join([str(v) for v in layer_lat])
    f.write(layer_choice_lat + "\n")
f.close()

f = open("allblock_final_lat.txt", "w")
for l in range(12):
    res_layer_lat = res_final_lat[l]
    mobile_layer_lat = mobile_final_lat[l]
    vgg_layer_lat = vgg_final_lat[l]
    res_all_lat = " ".join([str(v) for v in res_layer_lat])
    # mobile_all_lat = ' '.join([str(v) for v in mobile_layer_lat])
    # vgg_all_lat = ' '.join([str(v) for v in vgg_layer_lat])
    # layer_all_lat = res_all_lat + ' ' + mobile_all_lat + ' ' + vgg_all_lat
    # f.write(layer_all_lat + '\n')
    f.write(res_all_lat + "\n")
f.close()
