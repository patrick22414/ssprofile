import os
import torch

AWNAS_HOME = os.environ.get("AWNAS_HOME", os.path.expanduser("~/awnas"))


def get_awnas_dataset_dir(dataset_name):
    return os.path.join(AWNAS_HOME, "data", dataset_name)


def calc_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Args:
        outputs: [N x num_classes]
        targets: [N]
    """

    predicts = torch.argmax(outputs, dim=1)
    accuracy = (predicts == targets).sum().float() / targets.nelement()

    return accuracy.item()


def total_parameters(model: torch.nn.Module):
    n = 0
    for param in model.parameters():
        n += param.nelement()

    n = n * 1e-6
    return f"{n:.2f}M"


def get_lr(scheduler):
    """Solve PyTorch compatibility"""
    if "get_last_lr" in dir(scheduler):
        return scheduler.get_last_lr()[0]
    else:
        return scheduler.get_lr()[0]


def pretty_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, "Yi", suffix)

