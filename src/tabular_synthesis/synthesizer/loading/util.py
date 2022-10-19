import json
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SUPPORTED = ["adult"]
DATA_PATH = Path.cwd() / "src/tabular_synthesis/data"
CONFIG_PATH = DATA_PATH / "config"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_dataset_config(name:str):
    try:
        # check if dataset is in config path

        config = load_json(CONFIG_PATH/f"{name}.json")
        return config
    except FileNotFoundError:
        print(f"Dataset {name} not found in config path ({CONFIG_PATH}).\nSupported datasets: {SUPPORTED}")
        return None 

def get_dataset(name:str, azure=False):
    name = name.lower()
    config = get_dataset_config(name)
    if config:
        if name == "adult":
            features=["age","workclass","fnlwgt", "education", "education-num",	"marital-status", "occupation", "relationship", 
                        "race", "gender","capital-gain", "capital-loss", "hours-per-week","native-country", "income"]
            df=pd.read_csv(DATA_PATH/config["path"], names=features, sep=r'\s*,\s*', 
                             engine='python', na_values="?")
            if azure:
                from azureml.core.dataset import Dataset
                import azureml.core
                from azureml.core import Workspace
                import tempfile
                import os
                # Load the workspace from the saved config file
                ws = Workspace.from_config()
                dataset = Dataset.get_by_name(ws,"adult_train")

                path = tempfile.mkdtemp()
                with dataset.mount(path):
                    df = pd.read_csv(path+"/"+os.listdir(path)[0], na_values="?")
            return df, config

        return pd.read_csv(DATA_PATH/config["path"]), config
    else:
        return None

class MultiColumnLabelEncoder:
    """
    from https://python.tutorialink.com/how-to-reverse-label-encoder-from-sklearn-for-multiple-columns/
    """

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output
