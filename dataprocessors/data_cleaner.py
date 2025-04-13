# data_cleaner.py
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.feature_selection import VarianceThreshold

class DataCleaner:
    def __init__(self):
        pass

    def detect_outliers_zscore(self, data, threshold=3):
        z_scores = np.abs(zscore(data))
        return z_scores > threshold

    def detect_outliers_iqr(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        return (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))

    def remove_outliers(self, data, method="zscore", threshold=3):
        if method == "zscore":
            outlier_mask = self.detect_outliers_zscore(data, threshold)
        elif method == "iqr":
            outlier_mask = self.detect_outliers_iqr(data)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        return data[~outlier_mask.any(axis=1)]

    def winsorize(self, data, lower_limit=0.05, upper_limit=0.95):
        lower_bound = data.quantile(lower_limit)
        upper_bound = data.quantile(upper_limit)
        return data.clip(lower_bound, upper_bound)

    def standardize_units(self, data, column, target_unit):
        if target_unit == "meters":
            data[column] = data[column].apply(lambda x: x * 0.3048 if "feet" in str(x) else x)
        return data

    def remove_duplicates(self, data):
        return data.drop_duplicates()

    def remove_redundant_features(self, data, threshold=0.9):
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return data.drop(to_drop, axis=1)
