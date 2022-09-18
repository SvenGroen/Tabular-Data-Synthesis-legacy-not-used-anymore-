import pandas as pd
from .transform.transformer import DataTransformer, ImageTransformer
from .transform.data_preparation import DataPrep
from .sampling.sampler import Sampler
from .sampling.conditional_vector import Cond


class TabularLoader(object):
    def __init__(self,
                 data: pd.DataFrame,
                 test_ratio: float = 0.2,
                 categorical_columns: list = None,
                 log_columns: list = None,
                 mixed_columns: dict = None,
                 general_columns: list = None,
                 non_categorical_columns: list = None,
                 integer_columns: list = None):
        print("Initializing Tabular Loader...")
        self.data = data
        self.test_ratio = test_ratio
        self.categorical_columns = [] or categorical_columns
        self.log_columns = [] or log_columns
        self.mixed_columns = {} or mixed_columns
        self.general_columns = [] or general_columns
        self.non_categorical_columns = [] or non_categorical_columns
        self.integer_columns = [] or integer_columns
        print("Preparing raw data...")
        self.data_prep = DataPrep(raw_df=self.data,
                                  categorical=categorical_columns,
                                  log=log_columns,
                                  mixed=mixed_columns,
                                  general=general_columns,
                                  non_categorical=non_categorical_columns,
                                  integer=integer_columns)

        self.data_transformer = DataTransformer(train_data=self.data_prep.df,
                                                categorical_list=self.data_prep.column_types["categorical"],
                                                mixed_dict=self.data_prep.column_types["mixed"],
                                                general_list=self.data_prep.column_types["general"],
                                                non_categorical_list=self.data_prep.column_types["non_categorical"],
                                                n_clusters=10, eps=0.005)
        print("Fitting data transformer...")
        self.data_transformer.fit()
        self.transformed_data = self.data_transformer.transform(self.data_prep.df.values)
        print("Setting up Sampler, Cond and ImageTransformer...")
        self.sampler = Sampler(data=self.transformed_data,output_info=self.data_transformer.output_info)
        self.cond_generator = Cond(data=self.transformed_data,output_info=self.data_transformer.output_info)
        self.Image_transformer = ImageTransformer(side=32)
        print("Tabular Loader initialized successfully.")

