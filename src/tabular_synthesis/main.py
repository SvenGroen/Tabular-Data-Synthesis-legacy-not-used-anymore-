from tabular_synthesis.synthesizer.loading.tabular_loader import TabularLoader
from tabular_synthesis.synthesizer.loading.transform.data_preparation import DataPrep
from tabular_synthesis.synthesizer.loading.transform.transformer import ImageTransformer, DataTransformer
import pandas as pd

dataset = "Adult"
# Specifying the path of the dataset used
real_path = "data/real_datasets/adult/mini_adult.csv"

test_ratio = 0.20
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'gender', 'native-country', 'income']
log_columns = []
mixed_columns = {'capital-loss': [0.0], 'capital-gain': [0.0]}
integer_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
problem_type = {"Classification": 'income'}
epochs = 1

if __name__ == '__main__':
    data = pd.read_csv(real_path)


    init_arguments = {"test_ratio": 0.20,
                      "categorical_columns": ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                              'race', 'gender', 'native-country', 'income'],
                      "log_columns": [],
                      "mixed_columns": {'capital-loss': [0.0],
                                        'capital-gain': [0.0]},
                      "general_columns": ["age"],
                      "non_categorical_columns": [],
                      "integer_columns": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']}


    tabular_loader = TabularLoader(data=data, **init_arguments)
