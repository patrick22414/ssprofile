import contextlib
import math
import os
import re
import subprocess
import sys
import warnings
from copy import deepcopy
from os import path
from typing import Dict, List

import requests
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from _networks import SearchSpaceBaseNetwork
from _utils import calc_accuracy, get_lr, pretty_size, total_parameters

RE_INPUT_LAYER = re.compile(r"^input.*\n(^input_dim.*\n){4}", re.MULTILINE)
RE_FIRST_BOTTOM = re.compile(r"^  bottom: \"blob1\"", re.MULTILINE)
RE_DEPLOY_INPUT_LAYER = re.compile(r"", re.MULTILINE)  # TODO


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
        dataloaders: Dict[str, DataLoader],  # eg {"train": ..., "val": ...}
        # params for first training and finetuning
        first_train_epochs: int,
        first_train_optimizer_cfg: Dict,
        first_train_scheduler_cfg: Dict,
        finetune_epochs: int,
        finetune_optimizer_cfg: Dict,
        finetune_scheduler_cfg: Dict,
        # search space selection
        latency_cost_coeff: float,
        accuracy_threshold: float,
        accuracy_cost_scale: float,
        select_ss_threshold: int,
        # misc
        arch_file: str,
        dpu_url: str,
        profile_dir: str,
    ):
        super().__init__()

        self.search_spaces = search_spaces
        self.default_blocks = default_blocks

        self.num_cell_groups = num_cell_groups
        self.cell_layout = cell_layout
        self.reduce_cell_groups = reduce_cell_groups

        self.c_in_list = c_in_list
        self.c_out_list = c_out_list
        self.num_classes = num_classes

        self.dataloaders = dataloaders

        self.first_train_epochs = first_train_epochs
        self.first_train_optimizer_cfg = first_train_optimizer_cfg
        self.first_train_scheduler_cfg = first_train_scheduler_cfg

        self.finetune_epochs = finetune_epochs
        self.finetune_optimizer_cfg = finetune_optimizer_cfg
        self.finetune_scheduler_cfg = finetune_scheduler_cfg

        self.latency_cost_coeff = latency_cost_coeff
        self.accuracy_threshold = accuracy_threshold
        self.accuracy_cost_scale = accuracy_cost_scale
        self.select_ss_threshold = select_ss_threshold

        self.arch_file = arch_file
        self.dpu_url = dpu_url

        self.profile_dir = path.abspath(profile_dir)
        self.checkpoints_dir = path.join(profile_dir, "checkpoints")
        self.caffemodels_dir = path.join(profile_dir, "caffemodels")
        self.vitis_dir = path.join(profile_dir, "vitis")
        self.log_dir = path.join(profile_dir, "log_files")
        os.makedirs(self.checkpoints_dir)
        os.makedirs(self.caffemodels_dir)
        os.makedirs(self.vitis_dir)
        os.makedirs(self.log_dir)

        self.accuracy_table: Dict[str, float] = {}
        self.latency_table: Dict[str, float] = {}

    def select_search_space(self):
        """Select search space based on self.accuracy_table and self.latency_table
        
        Return:
            cell_shared_primitives: str
        """
        # calc cost_total of every model
        cost_table: Dict[str, float] = {}
        for model_id in self.accuracy_table:
            acc = self.accuracy_table[model_id]
            lat = self.latency_table[model_id]

            cost_acc = math.exp(
                -(acc - self.accuracy_threshold) / self.accuracy_cost_scale
            )
            cost_lat = -1 / lat

            cost_total = cost_acc + self.latency_cost_coeff * cost_lat
            cost_table[model_id] = cost_total

        # sort models by ascending cost
        selected_models: Dict[str, List[str]] = {
            ss: list() for ss in self.search_spaces
        }
        selected_ss = None
        for model_id in sorted(cost_table, key=lambda k: cost_table[k]):
            for ss in self.search_spaces:
                if model_id.startswith(ss):
                    selected_models[ss].append(model_id)

                if len(selected_models[ss]) >= self.select_ss_threshold:
                    selected_ss = ss

            if selected_ss is not None:
                break
        else:
            raise RuntimeError(
                f"No search space contains more than {self.select_ss_threshold} SSBNs. Consider lower the `self.select_ss_threshold`"
            )

        selected_models: List[str] = selected_models[selected_ss]

        print(f"Selected search space: \033[1m{selected_ss}\033[0m")
        print("Selected search space base networks:")
        for model_id in selected_models:
            print("   ", model_id)

        # get the selected primitive indices of each cell group
        cell_shared_primitives = [set() for _ in range(self.num_cell_groups)]
        for model_id in selected_models:
            selected_primitives = [
                int(prim_idx) for prim_idx in model_id.split("_")[1:]
            ]

            assert len(selected_primitives) == self.num_cell_groups
            for i in range(self.num_cell_groups):
                cell_shared_primitives[i].add(selected_primitives[i])

        # convert the inner set of indices to list of primitives str
        shared_primitives = self.search_spaces[selected_ss]
        for i in range(self.num_cell_groups):
            cell_shared_primitives[i] = [
                shared_primitives[idx] for idx in sorted(cell_shared_primitives[i])
            ]

        return cell_shared_primitives

    def profile(self, gpu):
        criterion = nn.CrossEntropyLoss()

        for ss in self.search_spaces:
            print("PROFILING SEARCH SPACE", ss)
            primitives = self.search_spaces[ss]
            default_block = self.default_blocks[ss]

            # make SSBN_0
            ssbn_0_id = self.get_ssbn_identifier(
                ss, [str(default_block)] * self.num_cell_groups
            )
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
                ssbn_0 = ssbn_0.to(gpu)

            optimizer = self.make_optimizer_from_cfg(
                self.first_train_optimizer_cfg, ssbn_0.parameters()
            )
            scheduler = self.make_scheduler_from_cfg(
                self.first_train_scheduler_cfg, optimizer
            )

            print(
                f"SSBN_0 (id: {ssbn_0_id}) ready for first traing.",
                f"Total parameters: {total_parameters(ssbn_0)}",
            )

            # first training
            mean_val_accuracy = self.profile_accuracy(
                model=ssbn_0,
                name="SSBN_0",
                epochs=self.first_train_epochs,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                gpu=gpu,
            )

            self.accuracy_table[ssbn_0_id] = mean_val_accuracy
            self.save_checkpoint(ssbn_0, ssbn_0_id)

            # send model to profile latency
            latency = self.profile_latency(ssbn_0, ssbn_0_id, gpu)
            self.latency_table[ssbn_0_id] = latency

            # save to CPU to generate other SSBNs
            ssbn_0 = ssbn_0.requires_grad_(False).cpu()

            print()

            # make other SSBNs
            count_ssbn = 1
            for cell_group in range(self.num_cell_groups):
                for i_block in range(len(primitives)):
                    if i_block == default_block:
                        continue

                    block_indices = [str(default_block)] * self.num_cell_groups
                    block_indices[cell_group] = str(i_block)
                    ssbn_id = self.get_ssbn_identifier(ss, block_indices)
                    ssbn_name = f"SSBN_{count_ssbn}"

                    ssbn = ssbn_0.swap_block(cell_group, i_block)

                    if torch.cuda.is_available():
                        ssbn = ssbn.to(gpu)

                    optimizer = self.make_optimizer_from_cfg(
                        self.finetune_optimizer_cfg, ssbn.parameters()
                    )
                    scheduler = self.make_scheduler_from_cfg(
                        self.finetune_scheduler_cfg, optimizer
                    )

                    print(
                        f"{ssbn_name} (id: {ssbn_id}) ready for finetuning.",
                        f"Total parameters: {total_parameters(ssbn)}",
                    )

                    # finetuning
                    mean_val_accuracy = self.profile_accuracy(
                        model=ssbn,
                        name=ssbn_name,
                        epochs=self.finetune_epochs,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        gpu=gpu,
                    )

                    self.accuracy_table[ssbn_id] = mean_val_accuracy
                    self.save_checkpoint(ssbn, ssbn_id)

                    # send model to profile latency
                    latency = self.profile_latency(ssbn, ssbn_id, gpu)
                    self.latency_table[ssbn_id] = latency

                    count_ssbn += 1
                    print()

            print("\n")  # for ss in self.search_spaces:

    def profile_accuracy(
        self, model, name, epochs, criterion, optimizer, scheduler, gpu
    ):
        if os.environ["DEBUG"]:
            print("Random value returned during DEBUG mode for `profile_accuracy`")
            return torch.rand(1).item()

        num_train_minibatch = len(self.dataloaders["train"])
        num_val_minibatch = len(self.dataloaders["val"])

        for epoch in range(epochs):
            print(f"{name} Epoch {epoch:2d} - learning rate: {get_lr(scheduler):.6f}")

            model.train()
            for i_minibatch, (inputs, targets) in enumerate(self.dataloaders["train"]):
                if torch.cuda.is_available():
                    inputs = inputs.to(gpu)
                    targets = targets.to(gpu)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if i_minibatch % 10 == 0:
                    accuracy = calc_accuracy(outputs, targets)
                    print(
                        f"{name} Epoch {epoch:2d} -",
                        f"train ({i_minibatch:3d}/{num_train_minibatch:3d})",
                        f"loss: {loss.item():.4f};",
                        f"acc: {accuracy:.2f}",
                    )

            scheduler.step()

            model.eval()
            mean_val_accuracy = 0.0
            for i_minibatch, (inputs, targets) in enumerate(self.dataloaders["val"]):
                if torch.cuda.is_available():
                    inputs = inputs.to(gpu)
                    targets = targets.to(gpu)

                outputs = model(inputs)
                accuracy = calc_accuracy(outputs, targets)
                mean_val_accuracy += accuracy

                if i_minibatch % 10 == 0:
                    print(
                        f"{name} Epoch {epoch:2d} -",
                        f"val ({i_minibatch:3d}/{num_val_minibatch:3d})",
                    )

            mean_val_accuracy /= num_val_minibatch
            print(
                f"{name} Epoch {epoch:2d} - val mean acc: \033[1m{mean_val_accuracy:.2f}\033[0m",
            )

        return mean_val_accuracy

    def profile_latency(self, model: nn.Module, model_id: str, gpu: int):
        if os.environ["DEBUG"]:
            print("Random value returned during DEBUG mode for `profile_latency`")
            return torch.rand(1).item() * 100

        print(f"{model_id} ready for latency profiling")
        is_training = model.training
        model.eval()

        model_prototxt = path.join(self.caffemodels_dir, f"{model_id}.prototxt")
        model_caffemodel = path.join(self.caffemodels_dir, f"{model_id}.caffemodel")

        quantize_dir = path.join(self.vitis_dir, f"{model_id}_quantize")
        compile_dir = path.join(self.vitis_dir, f"{model_id}_compile")
        os.makedirs(quantize_dir)
        os.makedirs(compile_dir)

        convert_log_file = path.join(self.log_dir, f"{model_id}.convert.log")
        quantize_log_file = path.join(self.log_dir, f"{model_id}.quantize.log")
        compile_log_file = path.join(self.log_dir, f"{model_id}.compile.log")

        from external.pytorch2caffe import pytorch_to_caffe

        with open(convert_log_file, "w") as fo:
            with contextlib.redirect_stdout(fo):
                with contextlib.redirect_stderr(fo):
                    pytorch_to_caffe.trans_net(
                        model, torch.randn(1, 3, 32, 32).to(gpu), model_id
                    )
                    pytorch_to_caffe.save_prototxt(model_prototxt)
                    pytorch_to_caffe.save_caffemodel(model_caffemodel)

        print("Saved model prototxt and caffemodel to:")
        print("   ", model_prototxt, pretty_size(path.getsize(model_prototxt)))
        print("   ", model_caffemodel, pretty_size(path.getsize(model_caffemodel)))

        with open(model_prototxt, "r") as fi:
            new_input_layer = "\n".join(
                [
                    "layer {",
                    '  name: "data"',
                    '  type: "Input"',
                    '  top: "data"',
                    "  input_param { shape: { dim: 1 dim: 3 dim: 32 dim: 32 } }",
                    "}",
                    "",
                ]
            )
            content = RE_INPUT_LAYER.sub(new_input_layer, fi.read())
            content = RE_FIRST_BOTTOM.sub('  bottom: "data"', content)

        with open(model_prototxt, "w") as fo:
            fo.write(content)

        print("Edited model prototxt:")
        print("   ", model_prototxt)

        self.vai_q_caffe(
            model_prototxt, model_caffemodel, gpu, quantize_dir, quantize_log_file
        )

        print("Saved quantized model to:")
        print("   ", quantize_dir)

        quantize_deploy_prototxt = path.join(quantize_dir, "deploy.prototxt")
        quantize_deploy_caffemodel = path.join(quantize_dir, "deploy.caffemodel")

        # TODO: Edit quantized deploy.prototxt for compile

        self.vai_c_caffe(
            quantize_deploy_prototxt,
            quantize_deploy_caffemodel,
            self.arch_file,
            model_id,
            compile_dir,
            compile_log_file,
        )

        elf_file = path.join(compile_dir, f"dpu_{model_id}.elf")
        latency_file = path.join(self.vitis_dir, f"{model_id}.latency.txt")

        print("Saved compiled ELF file to:")
        print("   ", elf_file, pretty_size(path.getsize(elf_file)))

        # TODO: send elf file

        with open(elf_file, "rb") as fb:
            resp = requests.post(
                url=self.dpu_url, files={"file": fb}, data={"kernel_name": model_id},
            )

            if resp.ok:
                with open(latency_file, "w") as fo:
                    fo.write(resp.content)
            else:
                raise RuntimeError("Cannot get responce from DPU")

        # TODO: parse latency file and get one float number

        model.train(is_training)

        return latency

    def save_checkpoint(self, model: nn.Module, model_id: str):
        if os.environ["DEBUG"]:
            return  # DEBUG

        if self.profile_dir is None:
            warnings.warn("Unable to save checkpoints. No profile_dir specified")
            return

        repr_txt = path.join(self.checkpoints_dir, f"{model_id}.repr.txt")
        state_dict_pth = path.join(self.checkpoints_dir, f"{model_id}.state_dict.pth")

        with open(repr_txt, "a") as fo:
            fo.write(repr(model))
            fo.write("\n")

        torch.save(model.state_dict(), state_dict_pth)

        print("Saved state dict and repr text to:")
        print("   ", state_dict_pth, pretty_size(path.getsize(state_dict_pth)))
        print("   ", repr_txt, pretty_size(path.getsize(repr_txt)))

    @staticmethod
    def vai_q_caffe(model, weights, gpu, output_dir, log_file):
        with open(log_file, "a") as fo:
            res = subprocess.run(
                "vai_q_caffe quantize"
                f" -model {model}"
                f" -weights {weights}"
                f" -gpu {gpu}"
                f" -output_dir {output_dir}",
                shell=True,
                stdout=fo,
                stderr=fo,
            )

            if res.returncode != 0:
                raise subprocess.SubprocessError(
                    "Quantize failed when running `vai_q_caffe`\n"
                    f"Check output and error by `cat {log_file}`"
                )

    @staticmethod
    def vai_c_caffe(p, c, arch, net_name, output_dir, log_file):
        with open(log_file, "w") as fo:
            res = subprocess.run(
                "vai_c_caffe"
                f" --prototxt {p}"
                f" --caffemodel {c}"
                f" --arch {arch}"
                f" --net_name {net_name}"
                f" --output_dir {output_dir}",
                shell=True,
                stdout=fo,
                stderr=fo,
            )

            res.check_returncode()

            if res.returncode != 0:
                raise subprocess.SubprocessError(
                    "Compile failed when running `vai_c_caffe`\n"
                    f"Check output and error by `cat {log_file}`"
                )

    @staticmethod
    def get_ssbn_identifier(search_space_name, block_indices_for_each_cell_group):
        return f"{search_space_name}_{'_'.join(block_indices_for_each_cell_group)}"

    @staticmethod
    def make_optimizer_from_cfg(cfg: Dict, parameters) -> optim.Optimizer:
        cfg = deepcopy(cfg)
        assert cfg["type"] in dir(optim), f"Unknown optimizer type {cfg['type']}"

        optimizer_cls = getattr(optim, cfg.pop("type"))
        optimizer = optimizer_cls(parameters, **cfg)

        return optimizer

    @staticmethod
    def make_scheduler_from_cfg(cfg: Dict, optimizer) -> lr_scheduler._LRScheduler:
        cfg = deepcopy(cfg)
        assert cfg["type"] in dir(lr_scheduler), f"Unknown scheduler type {cfg['type']}"

        scheduler_cls = getattr(lr_scheduler, cfg.pop("type"))
        scheduler = scheduler_cls(optimizer, **cfg)

        return scheduler


if __name__ == "__main__":
    pass
