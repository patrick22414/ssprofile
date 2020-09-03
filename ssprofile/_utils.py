import os

AWNAS_HOME = os.environ.get("AWNAS_HOME", os.path.expanduser("~/awnas"))


def get_awnas_dataset_dir(dataset_name):
    return os.path.join(AWNAS_HOME, "data", dataset_name)
