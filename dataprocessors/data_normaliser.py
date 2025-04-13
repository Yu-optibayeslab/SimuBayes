import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

class DataNormaliser:
    def __init__(self):
        """
        Initialise the DataNormaliser.
        """
        self.scalers = {}  # Store scalers or transformation methods for each column

    def normalise_data(self, data, normalisation_methods):
        """
        Normalise the data using specified methods.
        :param data: Input data as a pandas DataFrame.
        :param normalisation_methods: Dictionary specifying normalisation methods for each column.
                                      Example: {"D": "minmax", "L": "standard", "P": "robust"}
        :return: Normalised data as a pandas DataFrame.
        """
        normalised_data = data.copy()
        for column, method in normalisation_methods.items():
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in data.")

            if method == "minmax":
                scaler = MinMaxScaler()
                normalised_data[column] = scaler.fit_transform(data[[column]]).flatten()
                self.scalers[column] = scaler
            elif method == "standard":
                scaler = StandardScaler()
                normalised_data[column] = scaler.fit_transform(data[[column]]).flatten()
                self.scalers[column] = scaler
            elif method == "robust":
                scaler = RobustScaler()
                normalised_data[column] = scaler.fit_transform(data[[column]]).flatten()
                self.scalers[column] = scaler
            elif method == "l2":
                # Row-wise L2 normalisation
                l2_norms = np.linalg.norm(data.values, axis=1, keepdims=True)
                normalised_data = data.div(l2_norms.flatten(), axis=0)
                self.scalers["l2_row_norms"] = l2_norms  # Store L2 norms for inverse transform
            elif method == "log":
                if (data[column] < 0).any():
                    raise ValueError(f"Column '{column}' contains negative values. Log transformation requires non-negative values.")
                normalised_data[column] = np.log1p(data[column])
                self.scalers[column] = {"method": "log"}
            elif method == "power":
                scaler = PowerTransformer(method="yeo-johnson")
                normalised_data[column] = scaler.fit_transform(data[[column]]).flatten()
                self.scalers[column] = scaler
            else:
                raise ValueError(f"Unknown normalisation method: {method}")
        return normalised_data, self.scalers

    def apply_scalers(self, data):
        """
        Apply the stored scalers to a new dataset.
        :param data: Input data as a pandas DataFrame.
        :return: Normalised data as a pandas DataFrame.
        """
        normalised_data = data.copy()
        for column, scaler in self.scalers.items():

            if column not in data.columns:
                continue  # Skip if column doesn't exist in the input data

            if column == "l2_row_norms":
                # Apply row-wise L2 normalisation
                l2_norms = np.linalg.norm(data.values, axis=1, keepdims=True)
                normalised_data = data.div(l2_norms.flatten(), axis=0)
            elif isinstance(scaler, (MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer)):
                normalised_data[column] = scaler.transform(data[[column]]).flatten()
            elif isinstance(scaler, dict) and scaler.get("method") == "log":
                normalised_data[column] = np.log1p(data[column])
        return normalised_data

    def inverse_transform(self, data, output_columns, normalisation_methods):
        """
        Inverse transform the normalised data back to its original scale.
        :param data: Normalised data as a pandas DataFrame.
        :param normalisation_methods: Dictionary specifying normalisation methods for each column.
        :return: Data in the original scale.
        """
        # Convert inputs to numpy arrays if they aren't already
        data = np.asarray(data)

        # Ensure 2D arrays even for single output cases
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples = data.shape[0]
        n_outputs = len(output_columns)

        # Initialise DataFrame to store results
        original_data = pd.DataFrame(index=range(n_samples))

        # Initialise DataFrame to store results
        results = pd.DataFrame(index=range(n_samples))
        for i, column in enumerate(normalisation_methods):
            method = normalisation_methods[column]

            if method in ["minmax", "standard", "robust", "power"]:
                scaler = self.scalers.get(column)
                if scaler:
                    original= scaler.inverse_transform(data).flatten()
            elif method == "l2":
                # Inverse of row-wise L2 normalisation
                l2_norms = self.scalers.get("l2_row_norms")
                if l2_norms is not None:
                    original = data.mul(l2_norms.flatten(), axis=0)
            elif method == "log":
                original = np.expm1(data)
            else:
                raise ValueError(f"Unknown normalisation method: {method}")

            # Add to results DataFrame with proper column names
            original_data[f"{column}_org"] = original

        return original_data

    def inverse_transform_gps_outputs(self, mean, variance, output_columns, normalisation_methods):
        """
        Inverse transform the outputs (mean and variance) of a DeepGP model for multiple outputs.
        Returns a DataFrame with columns for each output's inverse-transformed mean and variance.
    
        Args:
            mean: Predicted mean values (normalised) of shape (n_samples, n_outputs) or (n_samples,).
            variance: Predicted variance values (normalised) of shape (n_samples, n_outputs) or (n_samples,).
            output_columns: List of column names for each output being predicted.
            normalisation_methods: List of normalisation methods used for each output column.
        
        Returns:
            DataFrame with columns named as "{output_col}_model_mean_org" and "{output_col}_model_var_org"
            for each output column.
        """
        # Convert inputs to numpy arrays if they aren't already
        mean = np.asarray(mean)
        variance = np.asarray(variance)
    
        # Ensure 2D arrays even for single output cases
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        if variance.ndim == 1:
            variance = variance.reshape(-1, 1)
    
        n_samples = mean.shape[0]
        n_outputs = len(output_columns)
    
        # Initialise DataFrame to store results
        results = pd.DataFrame(index=range(n_samples))
    
        # Process each output dimension separately
        for i, output_col in enumerate(normalisation_methods):
            current_mean = mean[:, i]
            current_var = variance[:, i]
            norm_method = normalisation_methods[output_col]
            if norm_method == "minmax":
                scaler = self.scalers.get(output_col)
                if scaler:
                    mean_original = scaler.inverse_transform(current_mean.reshape(-1, 1)).flatten()
                    scale_factor = scaler.scale_[0]
                    var_original = current_var * (scale_factor ** 2)
                
            elif norm_method == "standard":
                scaler = self.scalers.get(output_col)
                if scaler:
                    mean_original = scaler.inverse_transform(current_mean.reshape(-1, 1)).flatten()
                    std_dev = scaler.scale_[0]
                    var_original = current_var * (std_dev ** 2)
                
            elif norm_method == "robust":
                scaler = self.scalers.get(output_col)
                if scaler:
                    mean_original = scaler.inverse_transform(current_mean.reshape(-1, 1)).flatten()
                    iqr = scaler.scale_[0]
                    var_original = current_var * (iqr ** 2)
                
            elif norm_method == "log":
                mean_original = np.expm1(current_mean)
                var_original = current_var * (np.exp(current_mean) ** 2)
            
            elif norm_method == "power":
                scaler = self.scalers.get(output_col)
                if scaler:
                    mean_original = scaler.inverse_transform(current_mean.reshape(-1, 1)).flatten()
                    var_original = current_var  # Placeholder
            
            elif norm_method == "l2":
                mean_original = current_mean
                var_original = current_var
            
            else:
                raise ValueError(f"Unknown normalisation method for {output_col}: {norm_method}")
        
            # Add to results DataFrame with proper column names
            results[f"{output_col}_model_mean_org"] = mean_original
            results[f"{output_col}_model_var_org"] = var_original
    
        return results