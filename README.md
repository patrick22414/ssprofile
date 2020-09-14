# ssprofile
Search space profiling by accuracy and latency.

Requires:
- Python >= 3.6

## Workflow

```sh
$ python ssprofile/main.py <path to your YAML> \
        --gpu 0 \
        --arch-file <DPU compile target arch .json> \
        --dpu-url <DPU network location> \
        --profile-dir <path to profile dir>
```

1. Generate all SSBN models. Train or finetune them to get `accuracy_table`. Trained/finetuned PyTorch module state dicts and text representations are saved to `<profile dir>/checkpoints`.

1. Convert to caffemodels and save `prototxt`s, `caffemodel`s, and logs during conversion to `<profile dir>/caffemodels`.

1. **WIP** Compile caffemodels and send them to test latency on hardware. Quaitzed models are saved to `<profile dir>/vitis/quantize`. Compiled `.elf` files are saved to `<profile dir>/vitis/compile`.

1. **WIP** Read latency results and get `latency_table`.

1. **WIP** Get `cell_shared_primitives` from `accuracy_table` and `latency_table`.
