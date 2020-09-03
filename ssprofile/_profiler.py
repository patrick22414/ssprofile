from typing import Dict, List, Optional

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from _networks import SearchSpaceBaseNetwork


class SearchSpaceProfiler:
    def __init__(
        self,
        # params defining search spaces
        search_spaces: Dict[str, List[str]],  # eg {"ss0": ["block0", ...], "ss1": ...}
        default_blocks: Dict[str, int],  # eg {"ss0": 0, "ss1": 1, ...}
        num_cell_groups: int,
        cell_layout: List[int],
        reduce_cell_groups: List[int],
        # other params required to make SSBNs
        c_in_list: List[int],
        c_out_list: List[int],
        num_classes: int,
        # params for datasets
        dataloaders: Dict[str, DataLoader],  # eg {"train": ..., "test": ...}
        # params for first training and finetuning
        first_train_epochs: int,
        first_train_optimizer_cfg: Dict,
        first_train_scheduler_cfg: Dict,
        finetune_epochs: int,
        finetune_optimizer_cfg: Dict,
        finetune_scheduler_cfg: Dict,
        # others
        cost_latency_coeff: float,
        profile_dir: str = None,
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

        # TODO: how do we name each SSBN in each search space, ie keys in these dicts?
        self.accuracy_table: Dict[str, float] = {}
        self.latency_table: Dict[str, float] = {}

    def profile_accuracy(self):
        criterion = nn.CrossEntropyLoss()

        for ss in self.search_spaces:
            primitives = self.search_spaces[ss]
            default_block = self.default_blocks[ss]

            # make SSBN_0
            ssbn_0_id = self.get_ssbn_identifier(ss, [str(default_block)] * self.num_cell_groups)
            ssbn_0 = SearchSpaceBaseNetwork(
                primitives=primitives,
                default_block=self.default_blocks[ss],
                num_cell_groups=self.num_cell_groups,
                cell_layout=self.cell_layout,
                reduce_cell_groups=self.reduce_cell_groups,
                c_in_list=self.c_in_list,
                c_out_list=self.c_out_list,
                num_classes=self.num_classes,
            )

            if torch.cuda.is_available():
                ssbn_0 = ssbn_0.cuda()
            optimizer = self.make_optimizer_from_cfg(
                self.first_train_optimizer_cfg, ssbn_0.parameters()
            )
            scheduler = self.make_scheduler_from_cfg(
                self.first_train_scheduler_cfg, optimizer
            )

            # first training
            for epoch in range(self.first_train_epochs):
                ssbn_0.train()
                for inputs, targets in self.dataloaders["train"]:
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        targets = targets.cuda()

                    optimizer.zero_grad()
                    outputs = ssbn_0(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                ssbn_0.eval()
                for inputs, targets in self.dataloaders["test"]:
                    outputs = ssbn_0(inputs)
                    # TODO
                    # accuracy = calc_accuracy(outputs, accuracy)
                    # self.accuracy_table[ssbn_0_id] = accuracy

            ssbn_0 = ssbn_0.cpu()

            # make other SSBNs
            for cell_group in range(self.num_cell_groups):
                for i_block in range(len(primitives)):
                    if i_block == self.default_blocks:
                        continue

                    block_indices = [str(default_block)] * self.num_cell_groups
                    block_indices[cell_group] = str(i_block)
                    ssbn_id = self.get_ssbn_identifier(ss, block_indices)

                    ssbn = ssbn_0.swap_block(cell_group, i_block)
                    if torch.cuda.is_available():
                        ssbn = ssbn.cuda()

                    optimizer = self.make_optimizer_from_cfg(
                        self.finetune_optimizer_cfg, ssbn.parameters()
                    )
                    scheduler = self.make_scheduler_from_cfg(
                        self.finetune_scheduler_cfg, optimizer
                    )

                    for epoch in range(self.finetune_epochs):
                        ssbn.train()
                        for inputs, targets in self.dataloaders["train"]:
                            if torch.cuda.is_available():
                                inputs = inputs.cuda()
                                targets = targets.cuda()

                            optimizer.zero_grad()
                            outputs = ssbn(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                        ssbn.eval()
                        for inputs, targets in self.dataloaders["test"]:
                            outputs = ssbn(inputs)
                            # TODO
                            # accuracy = calc_accuracy(outputs, accuracy)
                            # self.accuracy_table[ssbn_id] = accuracy
        pass

    def profile_latency(self):
        pass

    def profile(self):
        # profile accuracy and latency
        # compute costs
        # sort and select search space and blocks
        pass

    @staticmethod
    def get_ssbn_identifier(search_space_name, block_indices_for_each_cell_group):
        return f"{search_space_name}_{'_'.join(block_indices_for_each_cell_group)}"

    @staticmethod
    def make_optimizer_from_cfg(cfg: Dict, parameters) -> optim.Optimizer:
        assert cfg["type"] in dir(optim), f"Unknown optimizer type {cfg['type']}"

        optimizer_cls = getattr(optim, cfg.pop("type"))
        optimizer = optimizer_cls(parameters, **cfg)

        return optimizer

    @staticmethod
    def make_scheduler_from_cfg(cfg: Dict, optimizer) -> lr_scheduler._LRScheduler:
        assert cfg["type"] in dir(lr_scheduler), f"Unknown scheduler type {cfg['type']}"

        scheduler_cls = getattr(lr_scheduler, cfg.pop("type"))
        scheduler = scheduler_cls(optimizer, **cfg)

        return scheduler


if __name__ == "__main__":
    pass
