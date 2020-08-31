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

MOBILE_TBS = [[1, 3], [1, 5], [1, 7], [2, 3], [2, 5], [2, 7], [4, 3], [4, 5], [4, 7]]
TYPICAL_CHOICE = [1] * 22
CHOICE_TOTAL = [TYPICAL_CHOICE]
for i in range(len(TYPICAL_CHOICE)):
    for j in range(len(MOBILE_TBS)):
        if j == TYPICAL_CHOICE[i]:
            continue
        tmp = copy.deepcopy(TYPICAL_CHOICE)
        tmp[i] = j
        CHOICE_TOTAL.append(tmp)

# load accuracy.txt
accuracy = np.loadtxt("./accuracy.txt", dtype=np.str)
assert accuracy.shape[0] == len(CHOICE_TOTAL)

# generate
f = open("accuracy_prototxt.txt", "w")
for CHOICE_LIST in CHOICE_TOTAL:
    dir_path = "_".join([str(v) for v in CHOICE_LIST])
    index_1, index_2 = np.where(accuracy == dir_path)
    assert len(index_1) == 1
    assert index_2[0] == 0
    index = index_1[0]
    # get accuracy
    this_acc = accuracy[index, 2]
    # copy dir_path
    os.system(
        "cp %s/deploy.prototxt ./search_space/%s_deploy.prototxt" % (dir_path, dir_path)
    )
    # write to file
    f.write("%s_deploy.prototxt %s\n" % (dir_path, this_acc))
f.close()
