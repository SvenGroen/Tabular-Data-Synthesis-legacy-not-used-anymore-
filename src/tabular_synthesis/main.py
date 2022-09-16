from src.tabular_synthesis.synthesizer.loading.transform import DataPrep
from src.tabular_synthesis.synthesizer.loading.transform import ImageTransformer, DataTransformer
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
    print("before prep:", data.shape)
    data_prep = DataPrep(data, categorical_columns, log_columns,
                         mixed_columns, integer_columns, problem_type, test_ratio)
    print("after prep:", data_prep.df.shape)
    data_transformer = DataTransformer(train_data=data_prep.df,
                                       categorical_list=data_prep.column_types["categorical"],
                                       mixed_dict=data_prep.column_types["mixed"], n_clusters=5)
    data_transformer.fit()
    train_data = data_transformer.transform(data_prep.df.values)
    print("after transform:", train_data.shape)

    sides = [4, 8, 16, 24, 32]
    col_size_g = data_transformer.output_dim
    for i in sides:
        if i * i >= col_size_g:
            gside = i
            break

    Gtransformer = ImageTransformer(gside)
    data = Gtransformer.inverse_transform(train_data)
    print("after image transform:", data.shape)