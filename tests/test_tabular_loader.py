import unittest

import pandas as pd
import os

import torch

from tabular_synthesis.synthesizer.loading.tabular_loader import TabularLoader
import pytest
import os

cwd = os.getcwd()
if "tests" in cwd:
    data_location = "data/real_datasets/adult/mini_adult.csv"
else:
    data_location = "tests/data/real_datasets/adult/mini_adult.csv"

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
                  "integer_columns": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
                  "batch_size": 32}


class TestTabularLoader(unittest.TestCase):



    # @pytest.mark.skip()
    def test_init(self):
        data = pd.read_csv(data_location, sep=",")
        tabular_loader = TabularLoader(data=data, **init_arguments)
        assert all(k is not None for k in (vars(tabular_loader).values()))

    def test_get_batch(self):
        data = pd.read_csv(data_location, sep=",")
        tabular_loader = TabularLoader(data=data, **init_arguments)
        batch, c, col, opt = tabular_loader.get_batch()
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (tabular_loader.batch_size,
                               tabular_loader.data_transformer.output_dim +
                               tabular_loader.cond_generator.n_opt)
        assert all(k is not None for k in (c, col, opt))
        batch, c, col, opt = tabular_loader.get_batch(image_shape=True)
        assert batch.shape == (tabular_loader.batch_size, 1, tabular_loader.side, tabular_loader.side)

    def test_inverse_batch(self):
        data = pd.read_csv(data_location, sep=",")
        tabular_loader = TabularLoader(data=data, **init_arguments)
        batch, c, col, opt = tabular_loader.get_batch()
        inverse = tabular_loader.inverse_batch(batch)
        assert isinstance(inverse, pd.DataFrame)
        assert inverse.shape == (tabular_loader.batch_size, data.shape[1])


    # def test_noise_batch(self):
    #     data = pd.read_csv(data_location, sep=",")
    #     init_arguments["batch_size"] = 3
    #     tabular_loader = TabularLoader(data=data, **init_arguments)
    #     noise = tabular_loader.get_noise_batch()
    #     assert isinstance(noise, torch.Tensor)
    #     assert noise.shape == (init_arguments["batch_size"], tabular_loader.data_transformer.output_dim)