import numpy as np
import pandas as pd
from typing import Tuple, Dict#This will tell you what is the type of the things in the tuple
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

class AvocadoPreprocessing:
    def __init__(self, prices: pd.DataFrame, test_ratio: float,
                 price_column_name: str = 'AveragePrice'):
        self.prices = prices
        self.test_ratio = test_ratio
        self.price_column_name = price_column_name

    def train_test_split(self) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """
        Performs train-test split on self.prices. Returns (train, test),
        with the number of test rows determined by self.test_ratio.
        """
        total_rows = len(self.prices)
        start_of_test = int(total_rows*(1-self.test_ratio))
        train = self.prices.iloc[:start_of_test].copy()
        test = self.prices.iloc[start_of_test:].copy()
        return train, test



    def min_max_scale(self, train: pd.DataFrame, test:pd.DataFrame ):
        """
        Applies sklearn.proprocessing.MinMaxScaler to train and test.
        Returns the transformed DataFrames.
        """
        scaler= MinMaxScaler()
        scaler = scaler.fit(train[[self.price_column_name]])
        train[self.price_column_name] = scaler.transform(train[[self.price_column_name]])
        test[self.price_column_name] = scaler.transform(test[[self.price_column_name]])
        self.scaler = scaler##
        return train, test

    @property#cached_property
    def scaled_split_data(self) -> Tuple[pd.DataFrame,pd.DataFrame]:
        return self.min_max_scale(*self.train_test_split())


    def run(self) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """
        Return the scaled train-test split obtained from self.prices.
        """
        return self.scaled_split_data

    def __str__(self):
        return(f"""
        price_column_name: {self.price_column_name} \n
        data start index: {self.prices.index[0]} \n
        data end index: {self.prices.index[-1]}
        test start index: {self.scaled_split_data[1].index[0]} \n
        """)
