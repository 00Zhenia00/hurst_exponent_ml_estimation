import os
import yaml
from types import SimpleNamespace

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config():
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
    return SimpleNamespace(**config)
