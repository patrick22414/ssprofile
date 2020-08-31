import torch
from torch import nn

from _modules import primitive_factory


class SearchSpaceBaseNetwork(nn.Module):
    """
    Search space base network

    SSBN functions as a normal PyTorch network. It consists of a backbone (Sequential)
    and a classifier (Global Pool + Linear). The difference is that it remembers its
    default setting (default_block) and blocks in the backbone can be swapped using
    `swap_block`.

    When swapping blocks, SSBN makes backup(s) of the original block in that cell group,
    if it uses the default block, and restores all other cell groups to the default
    block. If the swapped cell group is not using the default block, it does not make a
    back up.

    Blocks are 'cell group' based, ie different layers are instances of the same block
    if they belong to the same cell group. When `swap_block` is called, all layers in
    that cell group will change to the new block (backups are made for each of them if
    they originally use the default block).

    FIXME: wait, I think I need to backup the whole default network

    Args:
        primitives: List[str], they are passed to `primitive_factory` to generate a new
            Module
        default_block: int, index of the default block in `primitives`
        num_cell_groups: int
        cell_layout: List[int]
    """

    def __init__(
        self,
        primitives,
        default_block,
        num_cell_groups,
        cell_layout,
        reduce_cell_groups,
        num_classes,
        c_in_list,
        c_out_list,
    ):
        super().__init__()

        self.primitives = primitives

        assert len(set(cell_layout)) == num_cell_groups

        self.num_cell_groups = num_cell_groups
        self.cell_layout = cell_layout
        self.num_layers = len(cell_layout)
        self.reduce_cell_groups = reduce_cell_groups

        assert len(c_in_list) == self.num_layers
        assert len(c_out_list) == self.num_layers

        self.c_in_list = c_in_list
        self.c_out_list = c_out_list

        self.backbone = nn.Sequential()
        for cell_group, c_in, c_out in zip(
            self.cell_layout, self.c_in_list, self.c_out_list
        ):
            if cell_group in reduce_cell_groups:
                stride = 2
            else:
                stride = 1

            module = primitive_factory(
                primitives[default_block], c_in, c_out, stride=stride
            )
            self.backbone.add_module(str(len(self.backbone)), module)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Linear(c_out_list[-1], num_classes, bias=False),
        )

        self.backbone_backup = nn.Module()
        self.classifier_backup = nn.Module()
        self.has_backups = False

    @torch.no_grad()
    def swap_block(self, cell_group: int, new_block: int, device: torch.device):
        if not self.has_backups:
            self.backbone_backup = self.backbone.cpu().requires_grad_(False)
            self.classifier_backup = self.classifier.cpu().requires_grad_(False)
            self.has_backups = True

        self.backbone = self.backbone_backup.to(device).requires_grad_(True)
        self.classifier = self.classifier_backup.to(device).requires_grad_(True)

        stride = 2 if cell_group in self.reduce_cell_groups else 1

        for layer, layer_cg in enumerate(self.cell_layout):
            if layer_cg == cell_group:
                self.backbone[layer] = primitive_factory(
                    self.primitives[new_block],
                    self.c_in_list[layer],
                    self.c_out_list[layer],
                    stride,
                )

    def forward(self, x):
        pass


# TEST
if __name__ == "__main__":
    ssbn = SearchSpaceBaseNetwork(
        ["VGGblock_0", "VGGblock_1"],
        default_block=0,
        num_cell_groups=3,
        cell_layout=[0, 1, 2],
        reduce_cell_groups=[1],
        num_classes=10,
        c_in_list=[3, 10, 20],
        c_out_list=[10, 20, 30],
    )

    ssbn.swap_block(2, new_block=1, device=torch.device("cpu"))

    print(ssbn)
