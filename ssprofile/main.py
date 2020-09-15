import argparse
import os
import re
import shutil
import tempfile
import warnings
from datetime import datetime
from os import path
from pprint import pprint
from collections import OrderedDict
import torch
import yaml

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
    "latency_cost_coeff": 0.5,
    "accuracy_threshold": 0.92,
    "accuracy_cost_scale": 1.0,
    "select_ss_threshold": 5,
}


def default_profile_dir():
    return path.join(
        tempfile.gettempdir(),
        "ssprofile-" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    )


def main():
    parser = argparse.ArgumentParser(description="Search space profiling")
    parser.add_argument("input_yaml", type=str, help="Input .yaml file")
    parser.add_argument("-o", "--output-yaml", type=str, help="Output .yaml file")
    parser.add_argument(
        "--profile-dir",
        type=str,
        help="Folder for storing profiling files",
        default=default_profile_dir(),
    )
    parser.add_argument(
        "--arch-file",
        type=str,
        help="DPU architecture .json file as Vitis compile target",
        default="/opt/vitis_ai/compiler/arch/dpuv2/ZCU102/ZCU102.json",
    )
    parser.add_argument(
        "--dpu-url",
        type=str,
        help="DPU network location to post .elf file and get latency back",
        default="http://192.168.6.144:8055/test_latency/test_latency",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="GPU device to run training and finetuning on",
        default=0,
    )
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    if args.output_yaml is None:
        args.output_yaml = path.join(args.profile_dir, "output.yaml")

    print("CLI Args:")
    for name in vars(args):
        print(f"    {name:12s}:", vars(args)[name])

    if args.seed is not None:
        print(f"Turning on reproducibility with seed {args.seed}\n")
        print(
            "This may hurt performance, see https://pytorch.org/docs/stable/notes/randomness.html#reproducibility"
        )
        print("   ", "torch.manual_seed(args.seed)")
        torch.manual_seed(args.seed)
        print("   ", "torch.backends.cudnn.deterministic = True")
        torch.backends.cudnn.deterministic = True
        print("   ", "torch.backends.cudnn.benchmark = False")
        torch.backends.cudnn.benchmark = False

    if path.exists(args.profile_dir):
        shutil.rmtree(args.profile_dir)
        os.makedirs(args.profile_dir)
        shutil.copyfile(args.input_yaml, path.join(args.profile_dir, "input.yaml"))

    with open(args.input_yaml, "r") as yi:
        cfg = OrderedDict(yaml.safe_load(yi))

    if "ss_profiler_cfg" not in cfg:
        cfg["ss_profiler_cfg"] = SS_PROFILER_CFG

    cfg["ss_profiler_cfg"]["arch_file"] = args.arch_file
    cfg["ss_profiler_cfg"]["dpu_url"] = args.dpu_url
    cfg["ss_profiler_cfg"]["profile_dir"] = args.profile_dir

    profiler = init_profiler_from_cfg(cfg)

    print()
    if torch.cuda.is_available() and "gpu" in args:
        profiler.profile(args.gpu)
    else:
        raise RuntimeError("CUDA device not available or not specified")

    # print(profiler.accuracy_table)
    # print(profiler.latency_table)
    with open(path.join(args.profile_dir, "accuracy_table.yaml"), "w") as fo:
        for k in sorted(profiler.accuracy_table):
            yaml.safe_dump({k: profiler.accuracy_table[k]}, fo)

    with open(path.join(args.profile_dir, "latency_table.yaml"), "w") as fo:
        for k in sorted(profiler.latency_table):
            yaml.safe_dump({k: profiler.latency_table[k]}, fo)

    cell_shared_primitives = profiler.select_search_space()
    print(f"\n\033[1mcell_shared_primitives: {cell_shared_primitives}\033[0m")

    cfg["search_space_cfg"]["cell_shared_primitives"] = cell_shared_primitives
    cfg["search_space_cfg"]["shared_primitives"] = None

    with open(args.output_yaml, "w") as yo:
        for k, v in cfg.items():
            yaml.safe_dump({k: v}, yo)
            yo.write("\n")


def init_profiler_from_cfg(cfg: dict):
    # TODO: add more cfg checks

    # Process search_space_cfg
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

    # Process weights_manager_cfg
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

    ss_profiler_cfg = cfg["ss_profiler_cfg"]

    from _data import DATALOADERS

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
        **ss_profiler_cfg,
    )


if __name__ == "__main__":
    main()
