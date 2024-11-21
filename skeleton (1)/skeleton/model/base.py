from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import utils


class BaseModel(ABC):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.predictions = None


    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        pass

    @abstractmethod
    def predict(self) -> int:
        """

        """
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
