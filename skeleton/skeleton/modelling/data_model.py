import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *
from utils import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
                 # This method will create the model for data
                 #This will be performed in second activity
                 self.X = X
                 self.y = df['y'].to_numpy()
                 self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                     self.X, self.y, test_size=0.2, random_state=0)

    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df

