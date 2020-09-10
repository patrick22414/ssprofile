# ssprofile
Search space profiling by accuracy and latency.

Requires:
- Python >= 3.6

## Workflow
1. Generate all SSBN models. Train or finetune them to get `accuracy_table`. Convert them caffemodels and save.

```sh
$ python ssprofile/main.py <path to your YAML> --gpu 0 --profile-dir <path to profile dir>
```

`prototxt`s, `caffemodel`s, and logs during conversion are saved to `<path to profile dir>/caffemodels`.

Trained/finetuned PyTorch module state dicts/text representations are saved to `<path to profile dir>/checkpoints`.

2. **WIP** Compile caffemodels and test latency on hardware.

3. **WIP** Read latency results and get `latency_table`.
