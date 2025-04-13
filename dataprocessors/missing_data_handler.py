# missing_data_handler.py
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class MissingDataHandler:
    def __init__(self):
        pass

    def listwise_deletion(self, data):
        return data.dropna()

    def pairwise_deletion(self, data):
        return data

    def mean_imputation(self, data):
        imputer = SimpleImputer(strategy="mean")
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    def median_imputation(self, data):
        imputer = SimpleImputer(strategy="median")
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    def mode_imputation(self, data):
        imputer = SimpleImputer(strategy="most_frequent")
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    def knn_imputation(self, data, n_neighbors=5):
        imputer = KNNImputer(n_neighbors=n_neighbors)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    def mice_imputation(self, data):
        imputer = IterativeImputer(random_state=42)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    def forward_fill(self, data):
        return data.ffill()

    def backward_fill(self, data):
        return data.bfill()

    def interpolate(self, data, method="linear"):
        return data.interpolate(method=method)
