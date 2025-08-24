from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    
    return np.mean(y_hat == y)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    true_positives = np.sum((y_hat == cls) & (y == cls))
    predicted_positives = np.sum(y_hat == cls)
    return true_positives / predicted_positives if predicted_positives > 0 else 0


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    true_positives = np.sum((y_hat == cls) & (y == cls))
    actual_positives = np.sum(y == cls)
    return true_positives / actual_positives if actual_positives > 0 else 0



def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    return np.sqrt(np.mean((y_hat - y) ** 2))



def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    return np.mean(np.abs(y_hat - y))
