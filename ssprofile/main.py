import argparse
import re
import tempfile
import warnings
from datetime import datetime
from os import path
from pprint import pprint

import click
import torch
import yaml

from _data import DATALOADERS
from _profiler import SearchSpaceProfiler

# Block names must be {search space name}block_{idx}
DEFAULT_BLOCKS = {"Mobile": 1, "Res": 1, "VGG": 1}

# TODO: this cfg block should be written in YAML files
SS_PROFILER_CFG = {
    "first_train_epochs": 20,
    "first_train_optimizer_cfg": {
        "type": "SGD",
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0001,
    },
    "first_train_scheduler_cfg": {
        "type": "CosineAnnealingLR",
        "T_max": 20,
        "eta_min": 0.001,
    },
    "finetune_epochs": 5,
    "finetune_optimizer_cfg": {
        "type": "SGD",
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0001,
    },
    "finetune_scheduler_cfg": {
        "type": "CosineAnnealingLR",
        "T_max": 5,
        "eta_min": 0.001,
    },
    "cost_latency_coeff": 0.5,
}


def default_profile_dir():
    return path.join(
        tempfile.gettempdir(),
        "ssprofile-" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    )


def main():
    parser = argparse.ArgumentParser(description="Search space profiling")
    parser.add_argument("input_yaml", type=str, help="Input YAML file")
    parser.add_argument("-o", "--output-yaml", type=str, help="Output YAML file")
    parser.add_argument(
        "--gpu", type=int, help="The GPU device to run training and finetuning on"
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        help="Folder for storing profiling files",
        default=default_profile_dir(),
    )
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    # Get these stuff form YAML cfg
    # - c_in_list
    # - c_out_list
    # - reduce_layers
    with open(args.input_yaml, "r") as yi:
        cfg = yaml.safe_load(yi)

    profiler = init_profiler_from_cfg(cfg, args.profile_dir)

    print()
    if torch.cuda.is_available() and "gpu" in args:
        with torch.cuda.device(args.gpu):
            profiler.profile()
    else:
        raise RuntimeError("CUDA device not available or not specified")

    pprint(profiler.accuracy_table)


def init_profiler_from_cfg(cfg: dict, profile_dir=None):
    # TODO: add more cfg checks

    # Process search_space_cfg, get these stuff:
    # - primitives
    # - search_spaces
    search_space_cfg = cfg["search_space_cfg"]
    if search_space_cfg["shared_primitives"] is None:
        raise RuntimeError("Profiling needs shared_primitives specified")
    if search_space_cfg["cell_shared_primitives"] is not None:
        warnings.warn(
            "Profiling will ignore and overwrite `cell_shared_primitives`. Is this YAML already profiled?"
        )

    num_cell_groups = search_space_cfg["num_cell_groups"]
    cell_layout = search_space_cfg["cell_layout"]
    reduce_cell_groups = search_space_cfg["reduce_cell_groups"]

    primitives = search_space_cfg["shared_primitives"]

    search_spaces = dict()
    search_space_name_regex = re.compile(r"(.*)block_\d*")
    for prim in primitives:
        match = search_space_name_regex.match(prim)
        if match is None:
            raise RuntimeError(
                "Cannot find search space name from primitive {}".format(prim)
            )

        ss_name = match[1]

        if ss_name in search_spaces:
            search_spaces[ss_name].append(prim)
        else:
            search_spaces[ss_name] = [prim]

    for ss in search_spaces:
        # sort blocks in a SS by their idx
        search_spaces[ss] = list(
            sorted(search_spaces[ss], key=lambda name: int(name.split("_")[-1]))
        )

    print("Search spaces:")
    pprint(search_spaces)

    # Process search_space_cfg, get these stuff:
    # - c_in_list
    # - c_out_list
    # - num_classes
    weights_manager_cfg = cfg["weights_manager_cfg"]
    if not "cell_group_kwargs" in weights_manager_cfg:
        raise RuntimeError("Cannot find `cell_group_kwargs` in `weights_manager_cfg`")

    c_in_list = []
    c_out_list = []
    for kwargs in weights_manager_cfg["cell_group_kwargs"]:
        c_in_list.append(kwargs["C_in"])
        c_out_list.append(kwargs["C_out"])

    assert len(c_in_list) == len(cell_layout), "Not enough C_in for each layer"
    assert len(c_out_list) == len(cell_layout), "Not enough C_out for each layer"

    num_classes = weights_manager_cfg["num_classes"]

    return SearchSpaceProfiler(
        search_spaces=search_spaces,
        default_blocks=DEFAULT_BLOCKS,
        num_cell_groups=num_cell_groups,
        cell_layout=cell_layout,
        reduce_cell_groups=reduce_cell_groups,
        c_in_list=c_in_list,
        c_out_list=c_out_list,
        dataloaders=DATALOADERS,
        num_classes=num_classes,
        profile_dir=profile_dir,
        **SS_PROFILER_CFG,
    )


if __name__ == "__main__":
    main()
