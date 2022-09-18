import unittest

import pandas as pd
import os
from tabular_synthesis.synthesizer.loading.tabular_loader import TabularLoader
import pytest


data_location = "data/real_datasets/adult/mini_adult.csv"

test = pd.read_csv(data_location)
print(test)
init_arguments = {"test_ratio": 0.20,
                  "categorical_columns": ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                          'race', 'gender', 'native-country', 'income'],
                  "log_columns": [],
                  "mixed_columns": {'capital-loss': [0.0],
                                    'capital-gain': [0.0]},
                  "general_columns": ["age"],
                  "non_categorical_columns": [],
                  "integer_columns": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']}


class TestTabularLoader(unittest.TestCase):

    def test_init(self):
        data = pd.read_csv(data_location, sep=",")
        tabular_loader = TabularLoader(data=data, **init_arguments)
        assert all(k is not None for k in (vars(tabular_loader).values()))
