from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from _networks import SearchSpaceBaseNetwork


class SearchSpaceProfiler:
    def __init__(
        self,
        # params defining search spaces
        search_spaces: Dict[str, List[str]],  # eg {"ss0": ["block0", ...], "ss1": ...}
        default_blocks: Dict[str, str],  # eg {"ss0": "blockx", "ss1": ...}
        num_cell_groups: int,
        cell_layout: List[int],
        reduce_cell_groups: List[int],
        # params for first training and finetuning
        first_train_epochs: int,
        first_train_optimizer_cfg: Dict,
        first_train_scheduler_cfg: Dict,
        finetune_epochs: int,
        finetune_optimizer_cfg: Dict,
        finetune_scheduler_cfg: Dict,
        # other params required to make SSBNs
        c_in_list: List[int],
        c_out_list: List[int],
        num_classes: int,
        # params for datasets
        dataloaders: Dict[str, DataLoader],  # eg {"train": ..., "test": ...}
        # others
        cost_latency_coeff: float,
        device: torch.device,
        profile_dir: str,
        # TODO: what else?
    ):
        super().__init__()

        self.search_spaces = search_spaces
        self.default_blocks = default_blocks

        self.num_cell_groups = num_cell_groups
        self.cell_layout = cell_layout
        self.reduce_cell_groups = reduce_cell_groups

        self.first_train_epochs = first_train_epochs
        self.first_train_optimizer_cfg = first_train_optimizer_cfg
        self.first_train_scheduler_cfg = first_train_scheduler_cfg

        self.finetune_epochs = finetune_epochs
        self.finetune_optimizer_cfg = finetune_optimizer_cfg
        self.finetune_scheduler_cfg = finetune_scheduler_cfg

        self.c_in_list = c_in_list
        self.c_out_list = c_out_list
        self.num_classes = num_classes

        self.dataloaders = dataloaders

        self.profile_accuracy_on_this_device = device

        # TODO: how do we name each SSBN in each search space, ie keys in these dicts?
        self.accuracy_table: Dict[str, float] = {}
        self.latency_table: Dict[str, float] = {}

    def profile_accuracy(self):
        # 1. For each search space
        # 2. Construct a SSBN
        # 3. SSBN first train and test
        # 4. SSBN swap blocks, finetune, and test
        # 5. write results to accuracy_table

        # SSBNs are local to this method
        # num of SSBNs == len(search_spaces)

        # we need multiple optimizers/schedulers each time the SSBN changes

        for ss in self.search_spaces:
            # make SSBN
            ssbn = SearchSpaceBaseNetwork(
                primitives=self.search_spaces[ss],
                default_block=self.default_blocks[ss],
                num_cell_groups=self.num_cell_groups,
                cell_layout=self.cell_layout,
                reduce_cell_groups=self.reduce_cell_groups,
                c_in_list=self.c_in_list,
                c_out_list=self.c_out_list,
                num_classes=self.num_classes,
                device=self.profile_accuracy_on_this_device,
            )
            # first training
            for epoch in range(self.first_train_epochs):
                pass
        pass

    def profile_latency(self):
        pass

    def profile(self):
        # profile accuracy and latency
        # compute costs
        # sort and select search space and blocks
        pass

    def get_ssbn_identifier(search_space_name, block_indices_for_each_cell_group):
        return f"{search_space_name}_{'_'.join(block_indices_for_each_cell_group)}"

    def init_optimizer_from_cfg(cfg: Dict, module_parameters):
        assert cfg["type"] in dir(torch.optim), f"Unknown optimizer type {cfg['type']}"

        optimizer_cls = getattr(torch.optim, cfg.pop("type"))
        optimizer = optimizer_cls(module_parameters, **cfg)

        return optimizer

    def init_scheduler_from_cfg(cfg: Dict, optimizer):
        assert cfg["type"] in dir(
            torch.optim.lr_scheduler
        ), f"Unknown scheduler type {cfg['type']}"

        scheduler_cls = getattr(torch.optim.lr_scheduler, cfg.pop("type"))
        scheduler = scheduler_cls(optimizer, **cfg)

        return scheduler


if __name__ == "__main__":
    pass
