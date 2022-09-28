import pandas as pd
import numpy as np
import torch

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
                 integer_columns: list = None,
                 batch_size: int = 32,
                 noise_dim: int = 100):
        print("Initializing Tabular Loader...")
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.noise_dim = noise_dim

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
        self.sampler = Sampler(data=self.transformed_data, output_info=self.data_transformer.output_info)
        self.cond_generator = Cond(data=self.transformed_data, output_info=self.data_transformer.output_info)
        self.cond_vector = self.cond_generator.sample_train(self.batch_size)
        self.side = self.determine_image_side()
        self.Image_transformer = ImageTransformer(side=self.side)
        print("Tabular Loader initialized successfully.")

    def transform_to_image(self, data):
        pass

    def calculate_new_cond_vector(self):
        self.cond_vector = self.cond_generator.sample_train(self.batch_size)
        return self.cond_vector

    def get_batch(self, image_shape=False):
        c, mask, col, opt = self.calculate_new_cond_vector()
        perm = np.arange(self.batch_size)
        np.random.shuffle(perm)
        c = torch.from_numpy(c).to(self.device)
        data_batch = self.sampler.sample(n=self.batch_size, col=col[perm], opt=opt[perm])
        data_batch = torch.from_numpy(data_batch).to(self.device)
        data_batch = torch.cat([data_batch, c[perm]], dim=1)
        if image_shape:
            data_batch = self.Image_transformer.transform(data_batch)

        return data_batch, c[perm], col[perm], opt[perm]

    def determine_image_side(self):
        sides = [4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]
        col_size = self.data_transformer.output_dim + self.cond_generator.n_opt
        side = None
        for i in sides:
            if i * i >= col_size:
                side = i
                break
        return side

    # def get_noise_batch(self,refresh_cond_vector=False, image_shape=False):
    #     noise = torch.randn(self.batch_size, self.noise_dim, device=self.device)
    #     if refresh_cond_vector:
    #         self.calculate_new_cond_vector()
    #     c, mask, col, opt = self.cond_vector
    #     c = torch.from_numpy(c).to(self.device)
    #     noise = torch.cat([noise, c], dim=1)
    #     noise = noise.view(self.batch_size, self.noise_dim + self.cond_generator.n_opt, 1, 1)
    #
    #     if image_shape:
    #         noise = self.Image_transformer.transform(noise)
    #     return noise

    def inverse_batch(self, batch):
        result, num_invalid_ids = self.data_transformer.inverse_transform(batch)
        result_df = self.data_prep.inverse_prep(result)
        return result_df