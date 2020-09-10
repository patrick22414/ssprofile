import torch
from torch import nn
from torch.nn import functional as F
import ssprofile.external.nn_tools.Caffe.caffe_net

if __name__ == "__main__":
    module = nn.Linear(2, 3)
    # module.requires_grad_(False)
    module.train()
    for n, p in module.named_parameters():
        print(n, p.requires_grad)

    state_dict = module.state_dict()

    for k, v in state_dict.items():
        print(k, v.requires_grad)

    new_module = nn.Linear(2, 3)
    new_module.load_state_dict(module.state_dict())

    for n, p in new_module.named_parameters():
        print(n, p.requires_grad)
