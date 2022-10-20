import json
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder



def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_dataset_config(path:str):
    try:
        # check if dataset is in config path
        config = load_json(path)
        return config
    except FileNotFoundError:
        print(f"Dataset config not found ({path})")
        return None 

def get_dataset(path:str, config_path:str):
    config = get_dataset_config(config_path)
    if config:
        if config["dataset_name"] == "adult" :
            features=["age","workclass","fnlwgt", "education", "education-num",	"marital-status", "occupation", "relationship", 
                        "race", "gender","capital-gain", "capital-loss", "hours-per-week","native-country", "income"]

            df=pd.read_csv(path, names=features, sep=r'\s*,\s*', 
                engine='python', na_values="?")
            return df, config

        return pd.read_csv(path), config
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
