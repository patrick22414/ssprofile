# -*-coding:utf-8-*-
import caffe
from caffe import layers as Layers
from caffe import params as Params
import numpy as np
import os
import commands
import copy
import time
import sys

# prototxt and net define
net = caffe.NetSpec()

# function for generate data layer
def generate_data(method):
    if method == "train_test":
        net.data, net.label = Layers.Data(
            name="imagenet",
            ntop=2,
            transform_param=dict(
                mirror=True, crop_size=224, mean_value=[104, 117, 123],
            ),
            data_param=dict(
                source="/home/eva_share/imgNet/tiny_train_lmdb/",
                batch_size=64,
                backend=Params.Data.LMDB,
            ),
            include=dict(phase=caffe.TRAIN,),
        )
        # add repetive train part
        train_part = str(net.to_proto())
        ## test part
        net.data, net.label = Layers.Data(
            name="imagenet",
            ntop=2,
            transform_param=dict(
                mirror=False, crop_size=224, mean_value=[104, 117, 123],
            ),
            data_param=dict(
                source="/home/eva_share/imgNet/tiny_valid_lmdb/",
                batch_size=32,
                backend=Params.Data.LMDB,
            ),
            include=dict(phase=caffe.TEST),
        )
        return train_part
    elif method == "deploy":
        net.tops["data"] = Layers.Input(
            name="cifar", input_param=dict(shape=dict(dim=[1, 3, 224, 224]))
        )
        return ""
    else:
        raise NotImplementedError


# function for generate one conv module
def generate_conv(
    input_blob_name,
    layer_name,
    input_size,
    input_channel,
    kernel_size,
    stride,
    group,
    output_channel,
    batchnorm_flag,
    relu_flag,
):
    net.tops[layer_name + "_conv"] = Layers.Convolution(
        net.tops[input_blob_name],
        name=layer_name + "_conv",
        ntop=1,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        convolution_param=dict(
            num_output=output_channel,
            kernel_size=kernel_size,
            pad=(kernel_size - 1) / 2,
            stride=stride,
            group=group,
            weight_filler=dict(type="xavier",),
            bias_filler=dict(type="constant",),
        ),
    )
    input_blob_name = layer_name + "_conv"
    if batchnorm_flag:
        net.tops[layer_name + "_bn"] = Layers.BatchNorm(
            net.tops[input_blob_name],
            name=layer_name + "_bn",
            ntop=1,
            param=[
                dict(lr_mult=1, decay_mult=0),
                dict(lr_mult=1, decay_mult=0),
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0),
            ],
            batch_norm_param=dict(
                moving_average_fraction=0.9,
                scale_filler=dict(type="constant", value=1),
                bias_filler=dict(type="constant", value=0),
            ),
        )
        input_blob_name = layer_name + "_bn"
    if relu_flag:
        net.tops[layer_name + "_relu"] = Layers.ReLU(
            net.tops[input_blob_name], name=layer_name + "_relu", ntop=1
        )
        input_blob_name = layer_name + "_relu"
    output_size = input_size / stride
    return input_blob_name, output_size, output_channel


# function for generate one pooling module
def generate_pooling(
    input_blob_name,
    layer_name,
    input_size,
    input_channel,
    pool_method,
    kernel_size,
    stride,
):
    net.tops[layer_name + "_pool"] = Layers.Pooling(
        net.tops[input_blob_name],
        name=layer_name + "_pool",
        ntop=1,
        pooling_param=dict(pool=pool_method, kernel_size=kernel_size, stride=stride,),
    )
    output_size = input_size / stride
    output_blob_name = layer_name + "_pool"
    return output_blob_name, output_size, input_channel


# function for generate one fc
def generate_fc(input_blob_name, layer_name, feature_out, drop_flag, relu_flag):
    net.tops[layer_name + "_fc"] = Layers.InnerProduct(
        net.tops[input_blob_name],
        name=layer_name + "_fc",
        ntop=1,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        inner_product_param=dict(
            num_output=feature_out,
            weight_filler=dict(type="xavier",),
            bias_filler=dict(type="constant",),
        ),
    )
    input_blob_name = layer_name + "_fc"
    if drop_flag:
        net.tops[layer_name + "_drop"] = Layers.Dropout(
            net.tops[input_blob_name],
            name=layer_name + "_drop",
            ntop=1,
            dropout_param=dict(dropout_ratio=0.5,),
        )
        input_blob_name = layer_name + "_drop"
    if relu_flag:
        net.tops[layer_name + "_relu"] = Layers.ReLU(
            net.tops[input_blob_name], name=layer_name + "_relu", ntop=1
        )
        input_blob_name = layer_name + "_relu"
    return input_blob_name


# generate eltwise layer
def generate_eltwise(input_blob_name_1, input_blob_name_2, layer_name, eltwise_method):
    net.tops[layer_name + "_elt"] = Layers.Eltwise(
        net.tops[input_blob_name_1],
        net.tops[input_blob_name_2],
        name=layer_name + "_elt",
        ntop=1,
        eltwise_param=dict(operation=eltwise_method),
    )
    return layer_name + "_elt"


"""
MOBILE_TBS = [[3, 1], [3, 3], [3, 6], [5, 1], [5, 3], [5, 6]]
MOBILE_CHANNEL = [16]*1 + [24]*4 + [32]*4 + [64]*4 + [112]*4 + [184]*4 + [352]*1
MOBILE_STRIDE = [1] + [2] + [1]*3 + [2] + [1]*3 + [2] + [1]*3 + [1]*4 + [2] + [1]*3 + [1]

# function for generate mobilenet block
# 1x1conv + dwconv + 1x1conv + perhaps shotcut

def generate_mobile_block(input_blob_name, layer_name, input_size, input_channel, TBS_CHOICE, TBS_INDEX, method):
    batchnorm_flag = (method == 'train_test')
    kernel_size = MOBILE_TBS[TBS_CHOICE][0]
    expansion = MOBILE_TBS[TBS_CHOICE][1]
    tmp_input_blob_name = input_blob_name
    tmp_input_channel = input_channel
    input_blob_name, input_size, input_channel = generate_conv(input_blob_name, layer_name + '_1x1p', input_size, input_channel, \
                                                               1, 1, 1,  expansion * input_channel, batchnorm_flag, True, \
                                                               )
    input_blob_name, input_size, input_channel = generate_conv(input_blob_name, layer_name + '_dw', input_size, input_channel, \
                                                               kernel_size, MOBILE_STRIDE[TBS_INDEX], input_channel,  input_channel, batchnorm_flag, True, \
                                                               )
    input_blob_name, input_size, input_channel = generate_conv(input_blob_name, layer_name + '_1x1l', input_size, input_channel, \
                                                               1, 1, 1,  MOBILE_CHANNEL[TBS_INDEX], batchnorm_flag, False, \
                                                               )
    if MOBILE_STRIDE[TBS_INDEX] == 1 and tmp_input_channel == input_channel:
        input_blob_name = generate_eltwise(input_blob_name, tmp_input_blob_name, layer_name, Params.Eltwise.SUM)
    net.tops[layer_name + '_relu'] = \
        Layers.ReLU(net.tops[input_blob_name], name = layer_name + '_relu', ntop = 1)
    input_blob_name = layer_name + '_relu'
    return input_blob_name, input_size, input_channel
"""

RES_TBS = [[1, 3], [1, 5], [1, 7], [2, 3], [2, 5], [2, 7], [4, 3], [4, 5], [4, 7]]
RES_CHANNEL = (
    [16] * 1 + [24] * 2 + [32] * 2 + [64] * 2 + [112] * 2 + [184] * 2 + [352] * 1
)
RES_STRIDE = [1] + [2] + [1] + [2] + [1] + [2] + [1] + [1] * 2 + [2] + [1] + [1]

# function for generate resnet block
def generate_resnet_block(
    input_blob_name,
    layer_name,
    input_size,
    input_channel,
    TBS_CHOICE,
    TBS_INDEX,
    method,
):
    batchnorm_flag = method == "train_test"
    expansion = 1.0 / RES_TBS[TBS_CHOICE][0]
    kernel_size = RES_TBS[TBS_CHOICE][1]
    res_blob_name, res_size, res_channel = generate_conv(
        input_blob_name,
        layer_name + "_bdw",
        input_size,
        input_channel,
        kernel_size,
        RES_STRIDE[TBS_INDEX],
        1,
        input_channel,
        batchnorm_flag,
        True,
    )
    res_blob_name, res_size, res_channel = generate_conv(
        res_blob_name,
        layer_name + "_b1x1",
        res_size,
        res_channel,
        1,
        1,
        1,
        RES_CHANNEL[TBS_INDEX],
        batchnorm_flag,
        False,
    )
    input_blob_name, input_size, input_channel = generate_conv(
        input_blob_name,
        layer_name + "_1x1p",
        input_size,
        input_channel,
        1,
        1,
        1,
        int(expansion * input_channel),
        batchnorm_flag,
        True,
    )
    input_blob_name, input_size, input_channel = generate_conv(
        input_blob_name,
        layer_name + "_dw",
        input_size,
        input_channel,
        kernel_size,
        RES_STRIDE[TBS_INDEX],
        1,
        input_channel,
        batchnorm_flag,
        True,
    )
    input_blob_name, input_size, input_channel = generate_conv(
        input_blob_name,
        layer_name + "_1x1l",
        input_size,
        input_channel,
        1,
        1,
        1,
        RES_CHANNEL[TBS_INDEX],
        batchnorm_flag,
        False,
    )
    input_blob_name = generate_eltwise(
        input_blob_name, res_blob_name, layer_name, Params.Eltwise.SUM
    )
    net.tops[layer_name + "_relu"] = Layers.ReLU(
        net.tops[input_blob_name], name=layer_name + "_relu", ntop=1
    )
    input_blob_name = layer_name + "_relu"
    return input_blob_name, input_size, input_channel


# 22 block
TYPICAL_CHOICE = [0] * 12


def generate_mobile(method, CHOICE_LIST):
    # new net and new data
    global net
    net = caffe.NetSpec()
    train_part = generate_data(method)
    input_blob_name = "data"
    input_size = 224
    input_channel = 3
    # conv0
    input_blob_name, input_size, input_channel = generate_conv(
        input_blob_name,
        "conv0",
        input_size,
        input_channel,
        3,
        2,
        1,
        16,
        (method == "train_test"),
        True,
    )
    # assert
    assert len(CHOICE_LIST) == len(TYPICAL_CHOICE)
    assert sum([int(v1 != v2) for v1, v2 in zip(CHOICE_LIST, TYPICAL_CHOICE)]) in [0, 1]
    for i in range(12):
        layer_name = (["cL%d", "L%d"][int(CHOICE_LIST[i] == TYPICAL_CHOICE[i])]) % (
            i + 1
        )
        input_blob_name, input_size, input_channel = generate_resnet_block(
            input_blob_name,
            layer_name,
            input_size,
            input_channel,
            CHOICE_LIST[i],
            i,
            method,
        )
    # conv_tile
    input_blob_name, input_size, input_channel = generate_conv(
        input_blob_name,
        "conv_tile",
        input_size,
        input_channel,
        1,
        1,
        1,
        1504,
        (method == "train_test"),
        True,
    )
    # avg_pool
    input_blob_name, input_size, input_channel = generate_pooling(
        input_blob_name,
        "conv_tile",
        input_size,
        input_channel,
        Params.Pooling.AVE,
        7,
        7,
    )
    # fc loss and accuracy layer
    input_blob_name = generate_fc(input_blob_name, "FC", 1000, False, False)
    if method == "train_test":
        ## loss and accuracy
        net.tops["loss"] = Layers.SoftmaxWithLoss(
            net.tops[input_blob_name],
            net.label,
            name="loss",
            ntop=1,
            include=dict(phase=caffe.TRAIN),
        )
        net.tops["accuracy"] = Layers.Accuracy(
            net.tops[input_blob_name],
            net.label,
            name="accuracy",
            ntop=1,
            include=dict(phase=caffe.TEST),
        )
        return train_part + str(net.to_proto())
    else:
        return str(net.to_proto())


# write to dir
def generate_write_mobile(CHOICE_LIST):
    file_name = "_".join([str(v) for v in CHOICE_LIST])
    os.system("mkdir -p %s" % file_name)
    train_test_file = os.path.join(file_name, "train_test.prototxt")
    with open(train_test_file, "w") as f:
        f.write(generate_mobile("train_test", CHOICE_LIST))
    deploy_file = os.path.join(file_name, "deploy.prototxt")
    with open(deploy_file, "w") as f:
        f.write(generate_mobile("deploy", CHOICE_LIST))
    return file_name


# generate_write_mobile(TYPICAL_CHOICE)
# XXX

# occupy
occupy_list = ["", "", "", "", ""]
# total CHOICE_LIST
CHOICE_TOTAL = [TYPICAL_CHOICE]
for i in range(len(TYPICAL_CHOICE)):
    for j in range(len(RES_TBS)):
        if j == TYPICAL_CHOICE[i]:
            continue
        tmp = copy.deepcopy(TYPICAL_CHOICE)
        tmp[i] = j
        CHOICE_TOTAL.append(tmp)

# traverse
for CHOICE_LIST in CHOICE_TOTAL:
    # look up for no occupy gpu
    gpu = -1
    while True:
        for i, occupy in enumerate(occupy_list):
            if i == 0:
                continue
            if occupy == "":
                gpu = i
                break
            else:
                status, output = commands.getstatusoutput("ps -ef | grep %s" % occupy)
                output = len(output.split("\n"))
                assert output in [2, 3]
                if output == 2:
                    # cal accuracy
                    status, output = commands.getstatusoutput(
                        "grep accuracy %s/finetune.log | tail -n 1" % occupy
                    )
                    output = output.strip().split()[-1]
                    print("%s accuracy %s" % (occupy, output))
                    sys.stdout.flush()
                    gpu = i
                    break
        else:
            time.sleep(10)
            continue
        break
    # free gpu
    print("gpu %d free" % gpu)
    sys.stdout.flush()
    # generate and finetune, snapshot path, solver, train.sh
    dir_path = generate_write_mobile(CHOICE_LIST)
    os.system("mkdir -p %s/snapshot" % dir_path)
    os.system(
        'sed "s/A/%s/g" solver/finetune_template.prototxt > %s/finetune_solver.prototxt'
        % (dir_path, dir_path)
    )
    cmd = (
        "~/tool/caffe_dev/build/tools/caffe train -solver %s/finetune_solver.prototxt -weights typical/model/baseline.model -gpu %d >& %s/finetune.log &"
        % (dir_path, gpu, dir_path)
    )
    os.system('echo "%s"> %s/finetune.sh' % (cmd, dir_path))
    # finetune and occupy
    os.system("bash %s/finetune.sh &" % dir_path)
    print(cmd)
    sys.stdout.flush()
    occupy_list[gpu] = dir_path
# last
while True:
    flag = False
    for i, occupy in enumerate(occupy_list):
        if occupy != "":
            flag = True
            status, output = commands.getstatusoutput("ps -ef | grep %s" % occupy)
            output = len(output.split("\n"))
            assert output in [2, 3]
            if output == 2:
                # cal accuracy
                status, output = commands.getstatusoutput(
                    "grep accuracy %s/finetune.log | tail -n 1" % occupy
                )
                output = output.strip().split()[-1]
                print("%s accuracy %s" % (occupy, output))
                sys.stdout.flush()
                occupy_list[i] = ""
    if flag:
        time.sleep(10)
    else:
        break
