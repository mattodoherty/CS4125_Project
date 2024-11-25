from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import utils


class BaseModel:
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y



    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self) -> int:
        pass

    @abstractmethod
    def data_transform(self) -> None:
        pass

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
