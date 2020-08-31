from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from _modules import primitive_factory


class SearchSpaceBaseNetwork(nn.Module):
    """
    Search space base network

    SSBN functions as a normal PyTorch network. It consists of a backbone (Sequential)
    and a classifier (Global Pool + Linear). The difference is that it remembers its
    default settings (default_block) and blocks in the backbone can be swapped using
    `swap_block`.

    Blocks are 'cell group' based, ie different layers are instances of the same block
    if they belong to the same cell group. When `swap_block` is called, all layers in
    that cell group will change to the new block (backups are made for each of them if
    they originally use the default block).

    Args:
        primitives (list of str): they are passed to `primitive_factory` to generate new
            modules
        default_block (int): index of the default block in `primitives`
        num_cell_groups (int): not really used except for checking `cell_layout`
        cell_layout (list of int): which cell group does each layer belong to
        reduce_cell_groups (list of int): which cell groups reduce the resolution (will
            have stride=2)
        c_in_list (list of int): list of input channels for each layer
        c_out_list (list of int): list of output channels for each layer
        num_classes (int): number of output features for classifier
        device (torch.device): the *active* backbone and classifer will be put on this
            device
    """

    def __init__(
        self,
        primitives: List[str],
        default_block: int,
        num_cell_groups: int,
        cell_layout: List[int],
        reduce_cell_groups: List[int],
        c_in_list: List[int],
        c_out_list: List[int],
        num_classes: int,
        device: torch.device,
    ):
        super().__init__()

        self.primitives = primitives
        self.default_block = default_block

        assert len(set(cell_layout)) == num_cell_groups

        self.num_cell_groups = num_cell_groups
        self.cell_layout = cell_layout
        self.num_layers = len(cell_layout)
        self.reduce_cell_groups = reduce_cell_groups

        assert len(c_in_list) == self.num_layers
        assert len(c_out_list) == self.num_layers

        self.c_in_list = c_in_list
        self.c_out_list = c_out_list
        self.num_classes = num_classes
        self.device = device

        self.backbone = nn.Module()
        self.classifier = nn.Module()
        self.backbone_backup = nn.Module()
        self.classifier_backup = nn.Module()
        self.has_backups = False

        self._make_backbone_and_classifier()

    def swap_block(self, cell_group: int, new_block: int):
        """
        Change the blocks in a cell group to new blocks.
        Make backups on first call.
        """
        if not self.has_backups:
            # keep the backups on CPU
            self.backbone_backup = self.backbone.requires_grad_(False).cpu()
            self.classifier_backup = self.classifier.requires_grad_(False).cpu()
            self.has_backups = True

        # make new modules
        self._make_backbone_and_classifier()
        self.backbone.load_state_dict(self.backbone_backup.state_dict())
        self.classifier.load_state_dict(self.classifier_backup.state_dict())

        # replace the new block
        stride = 2 if cell_group in self.reduce_cell_groups else 1

        for layer, layer_cg in enumerate(self.cell_layout):
            if layer_cg == cell_group:
                self.backbone[layer] = primitive_factory(
                    self.primitives[new_block],
                    self.c_in_list[layer],
                    self.c_out_list[layer],
                    stride,
                ).to(self.device)

    def _make_backbone_and_classifier(self):
        """Create new backbone and classifer instances using default blocks"""
        self.backbone = nn.Sequential()
        for cell_group, c_in, c_out in zip(
            self.cell_layout, self.c_in_list, self.c_out_list
        ):
            if cell_group in self.reduce_cell_groups:
                stride = 2
            else:
                stride = 1

            self.backbone.add_module(
                name=str(len(self.backbone)),
                module=primitive_factory(
                    self.primitives[self.default_block], c_in, c_out, stride=stride
                ),
            )

        self.classifier = nn.Linear(self.c_out_list[-1], self.num_classes, bias=False)

        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, x):
        y = self.backbone(x)
        y = F.adaptive_avg_pool2d(y, 1).squeeze()
        y = self.classifier(y)

        return y

    def extra_repr(self):
        active_device = next(self.backbone.parameters()).device
        backup_device = (
            next(self.backbone_backup.parameters()).device if self.has_backups else None
        )

        return f"(active device): {active_device}; (backup device): {backup_device}; (has backups): {self.has_backups}"

    def cuda(*args, **kwargs):
        raise RuntimeError(
            "Do not call cuda() on SearchSpaceBaseNetwork, as it needs to keep the submodules and backups on different devices"
        )

    def to(*args, **kwargs):
        raise RuntimeError(
            "Do not call to() on SearchSpaceBaseNetwork, as it needs to keep the submodules and backups on different devices"
        )


# TEST
if __name__ == "__main__":
    from time import perf_counter

    s = perf_counter()
    ssbn = SearchSpaceBaseNetwork(
        ["VGGblock_0", "VGGblock_1"],
        default_block=0,
        num_cell_groups=3,
        cell_layout=[0, 1, 1, 2],
        reduce_cell_groups=[1],
        num_classes=10,
        c_in_list=[3, 10, 20, 30],
        c_out_list=[10, 20, 30, 40],
        device=torch.device("cuda:3"),
    )
    print(ssbn)
    print("SSBN __init__ time:", perf_counter() - s)

    s = perf_counter()
    ssbn.swap_block(1, new_block=1)

    print(ssbn)

    for name, param in ssbn.named_parameters():
        print(name, list(param.shape), param.requires_grad, param.device)

    print("SSBN swap_block time:", perf_counter() - s)

    s = perf_counter()
    ssbn = SearchSpaceBaseNetwork(
        ["Mobileblock_0", "Resblock_1"],
        default_block=0,
        num_cell_groups=3,
        cell_layout=[0, 1, 1, 2],
        reduce_cell_groups=[1],
        num_classes=10,
        c_in_list=[3, 10, 20, 30],
        c_out_list=[10, 20, 30, 40],
        device=torch.device("cuda:3"),
    )
    ssbn.swap_block(1, new_block=1)
    print(ssbn)

    for name, param in ssbn.named_parameters():
        print(name, list(param.shape), param.requires_grad, param.device)

    print("SSBN __init__ & swap_block time:", perf_counter() - s)

    x = torch.randn(10, 3, 32, 32).to(torch.device("cuda:3"))
    y = ssbn(x)

    y.view(-1).mean().backward()
