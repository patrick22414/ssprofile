import torch
from torch import nn


def primitive_str_to_module_cls(primitive: str):
    pass


# === Copied from https://github.com/walkerning/aw_nas/blob/master/examples/research/bbssp/ssp_plugin/ssp_plugin.py ===
class MobileBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, expansion, bn=True):
        super(MobileBlock, self).__init__()
        # assert not bn, "not support bn for now"
        bias_flag = not bn
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        else:
            raise ValueError("Not supported kernel_size %d" % kernel_size)
        inner_dim = int(C_in * expansion)
        if inner_dim == 0:
            inner_dim = 1
        self.op = nn.Sequential(
            nn.Conv2d(C_in, inner_dim, 1, stride=1, padding=0, bias=bias_flag),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                inner_dim,
                inner_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias_flag,
            ),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(inner_dim, C_out, 1, stride=1, padding=0, bias=bias_flag),
            nn.BatchNorm2d(C_out),
        )
        self.relus = nn.ReLU(inplace=False)
        self.res_flag = (C_in == C_out) and (stride == 1)

    def forward(self, x):
        if self.res_flag:
            return self.relus(self.op(x) + x)
        else:
            return self.relus(self.op(x))


class ResBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, expansion, bn=True):
        super(ResBlock, self).__init__()
        # assert not bn, "not support bn for now"
        bias_flag = not bn
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        else:
            raise ValueError("Not supported kernel_size %d" % kernel_size)
        inner_dim = int(C_in * expansion)
        if inner_dim == 0:
            inner_dim = 1
        self.opa = nn.Sequential(
            nn.Conv2d(C_in, inner_dim, 1, stride=1, padding=0, bias=bias_flag),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                inner_dim,
                inner_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias_flag,
            ),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(inner_dim, C_out, 1, stride=1, padding=0, bias=bias_flag),
            nn.BatchNorm2d(C_out),
        )
        self.opb = nn.Sequential(
            nn.Conv2d(
                C_in, C_in, kernel_size, stride=stride, padding=padding, bias=bias_flag
            ),
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias_flag),
            nn.BatchNorm2d(C_out),
        )
        self.relus = nn.ReLU(inplace=False)

    def forward(self, x):
        a = self.opa(x)
        b = self.opb(x)
        return self.relus(a + b)


class VGGBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_list, stride, bn=True):
        super(VGGBlock, self).__init__()
        bias_flag = not bn
        tmp_block = []
        for kernel_size in kernel_list:
            padding = int((kernel_size - 1) / 2)
            tmp_block.append(
                nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=bias_flag)
            )
            tmp_block.append(nn.BatchNorm2d(C_out))
            tmp_block.append(nn.ReLU(inplace=False))
            C_in = C_out
        if stride == 2:
            tmp_block.append(nn.MaxPool2d(2, stride))
        self.op = nn.Sequential(*tmp_block)

    def forward(self, x):
        return self.op(x)


# === End of copied code ===
