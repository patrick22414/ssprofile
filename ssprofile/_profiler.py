from typing import Dict, List
import torch

from _networks import SSBaseNetwork


class SSProfiler:
    def __init__(
        self,
        search_spaces: Dict[str, List[str]],  # eg {"ss0": ["block0", ...], "ss1": ...}
        default_blocks: Dict[str, str],  # eg {"ss0": "blockx", "ss1": ...}
        num_cell_groups: int,
        cell_layout: List[int],
        reduce_cell_groups: List[int],
        first_train_epochs: int,
        first_train_optimizer_cfg: Dict,
        finetune_epochs: int,
        finetune_optimizer_cfg: Dict,
        cost_latency_coeff: float,
        device: torch.device,
    ):
        super().__init__()

        self.search_spaces = search_spaces

        self.first_train_epochs = first_train_epochs
        # self.first_train_optimizer = init_optimizer_from_cfg(first_train_optimizer_cfg)
        self.finetune_epochs = finetune_epochs
        # self.finetune_optimizer = init_optimizer_from_cfg(finetune_optimizer_cfg)

        self.ssnbs = []
        for ss in self.search_spaces:
            self.ssnbs.append(
                SSBaseNetwork(
                    primitives=self.search_spaces[ss],
                    default_block=default_block[ss],
                    num_cell_groups=num_cell_groups,
                    cell_layout=cell_layout,
                    reduce_cell_groups=reduce_cell_groups,
                    # TODO
                )
            )

        self.accuracy_table: Dict[str, float] = {}
        self.latency_table: Dict[str, float] = {}

    def first_train(self):
        pass

    def finetune(self):
        pass


def init_optimizer_from_cfg(cfg: Dict, module_parameters):
    assert cfg["type"] in dir(torch.optim), f"Unknown optimizer type {cfg['type']}"

    optimizer_cls = getattr(torch.optim, cfg.pop("type"))
    optimizer = optimizer_cls(module_parameters, **cfg)

    return optimizer


def init_scheduler_from_cfg(cfg: Dict, optimizer, epochs):
    if not cfg is None:
        raise NotImplementedError("Scheduler cfg not implemented")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.001
    )

    return scheduler


if __name__ == "__main__":
    init_optimizer_from_cfg(
        {"type": "SGD", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0001,}, []
    )
