import json
from pathlib import Path
import pandas as pd
SUPPORTED = ["adult"]
DATA_PATH = Path.cwd() / "src/tabular_synthesis/data"
CONFIG_PATH = DATA_PATH / "config"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_dataset_config(name:str):
    try:
        # check if dataset is in config path

        config = load_json(CONFIG_PATH/f"{name.lower()}.json")
        return config
    except FileNotFoundError:
        print(f"Dataset {name} not found in config path ({CONFIG_PATH}).\nSupported datasets: {SUPPORTED}")
        return None 

def get_dataset(name:str):
    config = get_dataset_config(name)
    if config:
        return pd.read_csv(DATA_PATH/config["path"]), config
    else:
        return None