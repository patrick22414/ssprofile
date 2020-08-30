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
        if kernel_size not in (1, 3, 5, 7):
            raise ValueError("Not supported kernel_size %d" % kernel_size)
        padding = kernel_size // 2
        inner_dim = int(C_in * expansion)
        if inner_dim == 0:
            inner_dim = 1  # FIXME: what is this?
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
        if kernel_size not in (1, 3, 5, 7):
            raise ValueError("Not supported kernel_size %d" % kernel_size)
        padding = kernel_size // 2
        inner_dim = int(C_in * expansion)
        if inner_dim == 0:
            inner_dim = 1  # FIXME: what is this?
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


VGG_BLOCK_PARAMS = [
    {"kernel_list": [1]},
    {"kernel_list": [3]},
    {"kernel_list": [1, 3]},
    {"kernel_list": [5]},
    {"kernel_list": [1, 5]},
    {"kernel_list": [3, 3]},
    {"kernel_list": [1, 3, 3]},
    {"kernel_list": [7]},
    {"kernel_list": [1, 7]},
    {"kernel_list": [3, 5]},
    {"kernel_list": [1, 3, 5]},
    {"kernel_list": [3, 3, 3]},
    {"kernel_list": [1, 3, 3, 3]},
]

RES_BLOCK_PARAMS = [
    {"expansion": 1, "kernel_size": 3},
    {"expansion": 1, "kernel_size": 5},
    {"expansion": 1, "kernel_size": 7},
    {"expansion": 2, "kernel_size": 3},
    {"expansion": 2, "kernel_size": 5},
    {"expansion": 2, "kernel_size": 7},
    {"expansion": 4, "kernel_size": 3},
    {"expansion": 4, "kernel_size": 5},
    {"expansion": 4, "kernel_size": 7},
]

MOBILE_BLOCK_PARAMS = [
    {"kernel_size": 3, "expansion": 1},
    {"kernel_size": 3, "expansion": 3},
    {"kernel_size": 3, "expansion": 6},
    {"kernel_size": 5, "expansion": 1},
    {"kernel_size": 5, "expansion": 3},
    {"kernel_size": 5, "expansion": 6},
    {"kernel_size": 7, "expansion": 1},
    {"kernel_size": 7, "expansion": 3},
    {"kernel_size": 7, "expansion": 6},
]


def primitive_factory(primitive: str, C_in: int, C_out: int, stride: int):
    ss, idx = primitive.split("block_")
    idx = int(idx)

    if ss == "VGG":
        return VGGBlock(C_in, C_out, stride=stride, **VGG_BLOCK_PARAMS[idx])
    elif ss == "Res":
        return ResBlock(C_in, C_out, stride=stride, **RES_BLOCK_PARAMS[idx])
    elif ss == "Mobile":
        return MobileBlock(C_in, C_out, stride=stride, **MOBILE_BLOCK_PARAMS[idx])
    else:
        raise NotImplementedError(f"Search space {ss} not implemented")


# TEST
if __name__ == "__main__":
    block = primitive_factory("VGGblock_3", C_in=10, C_out=20, stride=1)
    print(block)
