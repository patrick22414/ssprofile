# ssprofile
Search space profiling by accuracy and latency.

Requires:
- Python >= 3.6, for f-strings
- PyTorch
- PyYAML, for loading and dumping YAML files
- requests, for sending .elf files to DPU

## Command Line

```sh
$ python ssprofile/main.py <path to your YAML> --gpu 0 --profile-dir <path to profile dir>
```

**!!! BUT THIS WILL NOT WORK RIGHT NOW** because of bugs in `pytorch2caffe`, so please use:
```sh
$ DEBUG=1 python ssprofile/main.py test.yaml --gpu 0 --profile-dir tmp
```
to see a demo of the program running.

You can check more command line options by
```sh
$ python ssprofile/main.py --help
```

## Workflow

1. Generate all SSBN models. Train or finetune them to get `accuracy_table`. Trained/finetuned PyTorch module state dicts and text representations are saved to `<profile dir>/checkpoints`.

1. Convert to caffemodels and save `prototxt`s, `caffemodel`s, and logs during conversion to `<profile dir>/caffemodels`.

1. **WIP** Compile caffemodels and send them to test latency on hardware. Quaitzed models are saved to `<profile dir>/vitis/quantize`. Compiled `.elf` files are saved to `<profile dir>/vitis/compile`.

1. **WIP** Read latency results and get `latency_table`.

1. Get `cell_shared_primitives` from `accuracy_table` and `latency_table`.
