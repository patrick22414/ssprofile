import torch
from torch import nn

from _modules import primitive_factory


class SSBaseNetwork(torch.nn.Module):
    """
    Search space base network

    SSBN functions as a normal PyTorch network. It consists of a backbone (Sequential)
    and a classifier (Global Pool + Linear). The difference is that it remembers its
    default setting (default_blocks) and blocks in the backbone can be swapped using
    `swap_block`.

    When swapping blocks, SSBN makes backup(s) of the original block in that cell group,
    if it uses the default block, and restores all other cell groups to the default
    block. If the swapped cell group is not using the default block, it does not make a
    back up.

    Blocks are 'cell group' based, ie different layers are instances of the same block
    if they belong to the same cell group. When `swap_block` is called, all layers in
    that cell group will change to the new block (backups are made for each of them if
    they originally use the default block).

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
        self.num_cell_groups = num_cell_groups
        self.cell_layout = cell_layout
        self.num_layers = len(cell_layout)
        self.reduce_cell_groups = reduce_cell_groups

        assert len(c_in_list) == self.num_layers
        assert len(c_out_list) == self.num_layers

        self.c_in_list = c_in_list
        self.c_out_list = c_out_list

        # take note of active blocks in each cell group
        self.activate_blocks = [default_block] * self.num_cell_groups
        self.backup_block = nn.ModuleList()

        self.backbone = nn.Sequential()
        for cell_group, c_in, c_out in zip(
            self.cell_layout, self.c_in_list, self.c_out_list
        ):
            if cell_group in reduce_cell_groups:
                stride = 2
            else:
                stride = 1

            module = primitive_factory(primitives[default_block], c_in, c_out, stride)
            self.backbone.add_module(f"layer_{i}", module)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Linear(c_out_list[-1], num_classes, bias=False),
        )

    def swap_block(self, cell_group: int, new_block: int):
        pass

    def forward(self, x):
        pass


# TEST
if __name__ == "__main__":
    x = torch.nn.Sequential()
    for i in range(5):
        x.add_module(str(i), torch.nn.Conv2d(10, 20, i + 1))

    print(x)

    torch.nn.init.ones_(x[4].weight)
    backup = x[4]
    x[4] = nn.Linear(10, 20)
    print(x)

    x[4] = backup
    print(x)
    print(x[4].weight)
