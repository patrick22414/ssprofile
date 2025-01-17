# ssprofile

> Work in progress. Proceed with caution!

This repo is an implementation of the ASP-DAC20 paper *[Black Box Search Space Profiling for Accelerator-Aware Neural Architecture Search](https://nicsefc.ee.tsinghua.edu.cn/media/publications/2020/ASP-DAC20_292.pdf)*.

This repo works closely with the **[aw_nas](https://github.com/walkerning/aw_nas)** NAS framework. It takes the **aw_nas** YAML configuration files as inputs and outputs the optimized search space in `cell_shared_primitives`. **aw_nas** is required to make use of the YAML configuration files, but is not required for running the program in this repo.

Requires:
- Python, >= 3.6, for f-strings
- PyTorch
- PyYAML, for loading and dumping YAML files
- requests, for sending DPU ELF files to DPU

## Usage

### Demo

You can use `DEBUG=1` to see a demo of the program running. Or you can use `DEBUG_ACC=1` and `DEBUG_LAT=1` to partially skip accuracy or latency profiling.
```sh
$ DEBUG=1 python ssprofile/main.py test.yaml
```

### Not Demo

First, start the `auto_deploy` backend on your DPU server.
```sh
$ git clone http://192.168.3.224:8081/toolchain/auto_deploy.git # only works at Novauto
$ cd auto_deploy && python3 manage.py runserver 0.0.0.0:8055
```

You will need the [Xilinx Vitis AI](https://github.com/Xilinx/Vitis-AI) GPU docker image to quantize and compile models for DPU latency profiling.

On you working machine, inside the *root folder of this repo*, run
```sh
$ nvidia-docker run -ti -v `pwd`:`pwd` -w `pwd` \
        -p 127.0.0.1:80:8080/tcp \
        xilinx/vitis-ai:latest bash
```

Insider the docker container
```sh
$ conda activate vitis-ai-caffe
$ conda install requests pyyaml # (add pytorch if it's not installed)
```

Check the command line options by
```sh
$ python ssprofile/main.py --help
```

Run the profiler
```sh
$ python ssprofile/main.py <path to your YAML> --gpu 0 --profile-dir <path to profile dir>
```

## Workflow

1. Generate all SSBN models. Train or finetune them to get `accuracy_table`. Trained/finetuned PyTorch module state dicts and text representations are saved to `<profile dir>/checkpoints`.

1. Convert PyTorch models to caffemodels using `pytorch2caffe`, and save `prototxt`s and `caffemodel`s to `<profile dir>/caffemodels`.

1. Quantize (`vai_q_caffe`) and compile (`vai_c_caffe`) caffemodels. Quaitzed models are saved to `<profile dir>/vitis/XXX_quantize`. Compiled ELF files are saved to `<profile dir>/vitis/XXX_compile`.

1. Post ELF files to DPU server via HTTP and save the responses to `<profile dir>/vitis/XXX.latency.txt`.

1. Read latency results and get `latency_table`.

1. Get `cell_shared_primitives` from `accuracy_table` and `latency_table`.

## Known Issues:

1. `pytorch2caffe` -> `vai_q_caffe` -> `vai_c_caffe` toolchain still fails sometimes, especially for VGG-style networks.

1. Calibration images only have `32x32` resolution right now (inside `calib_data`, [Vitis docs](https://www.xilinx.com/html_docs/vitis_ai/1_2/modelquantization.html#tlm1570695754169)).

1. Other issues that I haven't noticed.
