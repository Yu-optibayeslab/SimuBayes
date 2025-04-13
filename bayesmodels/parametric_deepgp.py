import torch
import gpytorch
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.likelihoods import GaussianLikelihood
from itertools import zip_longest

import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st

class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, outputs_dims, num_inducing=128, mean_type='constant'):
        
        # 1. Construct or shape your inducing points
        if outputs_dims is None:
            # single-outputs (scalar) layer
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            # multi-outputs layer
            inducing_points = torch.randn(outputs_dims, num_inducing, input_dims)
            batch_shape = torch.Size([outputs_dims])

        # 2. Variational distribution & strategy
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        # 3. Call parent constructor with required arguments
        super(DeepGPHiddenLayer, self).__init__(
            variational_strategy,
            input_dims=input_dims,
            output_dims=outputs_dims,
        )

        # 4. Define mean/covar modules
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'zero':
            self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        else:
            # e.g. 'linear'
            self.mean_module = LinearMean(input_dims)

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        # Optional skip-connection logic
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()
            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]
            x = torch.cat([x] + processed_inputs, dim=-1)
        return super().__call__(x, are_samples=bool(len(other_inputs)))

class ParametrisedDeepGP(DeepGP):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: list,
        num_inducing_per_layer: list,
        mean_type: str = 'zero',
        final_outputs_dim=None
    ):
        """
        Builds a deep GP with the specified hidden dimensions.
        Example: 
          input_dims=10, hidden_dims=[4,2], final_outputs_dim=None => 2 hidden layers, last layer is single-outputs.
          input_dims=10, hidden_dims=[4,2,2], num_inducing_per_layer=[32, 64, 96, 128] => [32, 64, 96] inducingpoint for 1-3 hidden layers, [128] for the laster layer!
        """
        super().__init__()

        # Create the first hidden layer
        self.hidden_layers = torch.nn.ModuleList()
        current_input_dims = input_dims
        for hd, ni in zip (hidden_dims, num_inducing_per_layer[0:-1]):
            layer = DeepGPHiddenLayer(
                input_dims=current_input_dims,
                outputs_dims=hd,
                num_inducing=ni,
                mean_type=mean_type,
            )
            self.hidden_layers.append(layer)
            current_input_dims = hd

        # Final layer
        self.last_layer = DeepGPHiddenLayer(
            input_dims=current_input_dims,
            outputs_dims=final_outputs_dim,
            num_inducing=num_inducing_per_layer[-1],
            mean_type=mean_type
        )

        # Likelihood
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.last_layer(x)
        return x

    def predict(self, test_x):
        with torch.no_grad():
            mus = []
            variances = []
            preds = self.likelihood(self(test_x))
            mus.append(preds.mean.mean(0))
            variances.append(preds.variance.mean(0))

            return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)

    @property
    def device(self):
        """
        Convenience property: returns the current device of the model (assuming 
        all parameters are on the same device).
        """
        return next(self.parameters()).device

class DeviceAware:
    def __init__(self):
        self.device = self._get_device()  # Determine the device (GPU or CPU)

    def _get_device(self):
        """
        Check if CUDA is available and return the appropriate device.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataHandler(DeviceAware):
    def __init__(self, data_df, input_columns, outputs_columns, split_ratio=[0.7, 0.15, 0.15]):
        """
        Initialize the DataHandler.
        :param data_path: dataframe contains data to be modelled
        :param input_columns: List of input column names.
        :param outputs_columns: Name of the outputs column.
        :param split_ratio: List of ratios for train, validation, and test splits (e.g., [0.7, 0.15, 0.15]).
        """
        super().__init__()  # Initialize DeviceAwareBase to set up self.device
        self.data_df = data_df 
        self.input_columns = input_columns
        self.outputs_columns = outputs_columns
        self.split_ratio = split_ratio

    def load_data(self):
        """
        Load data from the CSV file and split it into train, validation, and test sets.
        """
        # Extract inputs and outputs
        X = self.data_df[self.input_columns].values
        y = self.data_df[self.outputs_columns].values

        # Split data into train, validation, and test sets
        train_ratio, val_ratio, test_ratio = self.split_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def preprocess_data(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        """
        Convert data to PyTorch tensors and move them to the appropriate device.
        """
        # Convert data to PyTorch tensors
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        val_x = torch.tensor(X_val, dtype=torch.float32)
        val_y = torch.tensor(y_val, dtype=torch.float32)
        test_x = torch.tensor(X_test, dtype=torch.float32)
        test_y = torch.tensor(y_test, dtype=torch.float32)

        _, num_cols = train_y.size()
        # Ensure y tensors have the correct shape
        if num_cols == 1:
            train_y, test_y, val_y = train_y.squeeze(-1) , test_y.squeeze(-1), val_y.squeeze(-1)  # remove additional dim if it is a colum vector

        # Move tensors to the appropriate device (GPU or CPU)
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        val_x = val_x.to(self.device)
        val_y = val_y.to(self.device)
        test_x = test_x.to(self.device)
        test_y = test_y.to(self.device)

        return train_x, train_y, val_x, val_y, test_x, test_y

    def create_dataloaders(self, train_x, train_y, val_x, val_y, test_x, test_y, batch_size=1024):
        """
        Create DataLoader objects for training, validation, and testing datasets.
        """
        # Create TensorDataset objects
        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)
        test_dataset = TensorDataset(test_x, test_y)

        # Create DataLoader objects
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, shuffle=False)

        return train_loader, val_loader, test_loader

class Customised_Loss:
    def __init__(self, multipliers, denominators, v, m, pred_weight=0.1):
        """
        Example code!!!!
        Initialize the custom loss function.
        :param multipliers: Multipliers for the physics-based penalty.
        :param denominators: Denominators for the physics-based penalty.
        :param v: Scaling factor for the predicted CHF mean.
        :param m: Scaling factor for the predicted CHF variance.
        :param pred_weight: Weight for the physics-based penalty.
        """
        self.multipliers = multipliers
        self.denominators = denominators
        self.v = v
        self.m = m
        self.pred_weight = pred_weight

    def compute_loss(self, outputs, target, mll):
        """
        Example code!!!!
        Compute the custom loss.
        :param outputs: Model outputs (mean and variance).
        :param target: Target values.
        :param mll: Marginal log-likelihood (for the ELBO loss).
        :return: Total loss (ELBO + physics-based penalty).
        """
        # Physics-based penalty
        pred_chf_mean = torch.mean(outputs.mean * self.v + self.m, axis=0)
        pred_chf_var = torch.mean(outputs.variance * (self.v ** 2), axis=0)
        pred_Hout_mean = self.multipliers * pred_chf_mean
        pred_Hout_var = pred_chf_var * (self.multipliers ** 2)
        pred_X_mean = (pred_Hout_mean - self.denominators) * self.denominators
        pred_X_var = pred_Hout_var * (self.denominators ** 2)
        pred_LL = 0.5 * torch.sum(torch.log(2 * torch.pi * pred_X_var) + ((target - pred_X_mean) ** 2) / (2 * pred_X_var))

        # Total loss
        elbo_loss = -mll(outputs, target)  # ELBO loss
        total_loss = (1 - self.pred_weight) * elbo_loss + self.pred_weight * pred_LL
        return total_loss

class DeepGPModeler(DeviceAware):
    def __init__(self, input_dims, hidden_dims, num_inducing_per_layer, final_outputs_dim, mean_type='zero',  custom_loss=None):
        """
        Initialize the DeepGPModeler.
        :param input_dims: Input dimensions.
        :param hidden_dims: Hidden layer dimensions.
        :param num_inducing_per_layer: Number of inducing points for each hidden layer.
        :param final_outputs_dim: Dimension of the final outputs layer.
        :param mean_type: Type of mean function (default: 'zero').
        :param self.custom_loss: Instance of Customised_Loss (optional).
        """
        super().__init__()  # Initialize DeviceAwareBase to set up self.device
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_inducing_per_layer = num_inducing_per_layer
        self.final_outputs_dim = final_outputs_dim
        self.mean_type = mean_type
        self.custom_loss = custom_loss
        self.model = self.build_model()

    def build_model(self):
        # Define the DeepGP model
        model = ParametrisedDeepGP(
            input_dims=self.input_dims,
            hidden_dims=self.hidden_dims,
            num_inducing_per_layer=self.num_inducing_per_layer,
            mean_type=self.mean_type,
            final_outputs_dim=self.final_outputs_dim
        )
        return model.to(self.device)

    def reset_state(self):
        st.session_state.loss_values = []
        st.session_state.test_rmse_values = []
        st.session_state.test_msll_values = []
        st.session_state.loss_values_df = pd.DataFrame()  # Persist losses_df across reruns
        st.session_state.test_metrics_df = pd.DataFrame()  # Persist test_metric_df across reruns

    def train(self, train_loader, test_loader, num_epochs, learning_rate, eval_every=10):
        # Training logic
        # Create a placeholder for the loss plot
        loss_placeholder = st.empty() # Placeholder for dynamic chart
        test_metrics_placeholder = st.empty()  # For test metrics
        training_log = st.expander("📜 Training Log", expanded=True)  # Collapsible log

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mll = DeepApproximateMLL(VariationalELBO(self.model.likelihood, self.model, num_data=len(train_loader.dataset)))

        for epoch in range(num_epochs):
            if st.session_state.get("stop_training", False):
                st.error("The training process was manually stopped. As a result, the model could not be saved!")
                break  # Exit loop but retain existing outputs

            running_loss = 0.0
            for x_batch, y_batch in train_loader:
                with gpytorch.settings.num_likelihood_samples(5):
                    optimizer.zero_grad()
                    outputs = self.model(x_batch)
                    # Compute the loss
                    if self.custom_loss:
                        loss = self.custom_loss.compute_loss(outputs, y_batch, mll)  # Use custom loss (just an example here)
                        loss.backward(retain_graph=True)
                    else:
                        loss = -mll(outputs, y_batch)  # Use standard ELBO loss
                        loss.backward(retain_graph=True)
                    optimizer.step()
                    #minibatch_iter.set_postfix(loss=loss.item())
                    running_loss += loss.item()
            running_loss /= len(train_loader)
            st.session_state.loss_values.append(running_loss)

            # Update loss plot
            loss_df = pd.DataFrame({"Epoch": range(epoch + 1), "Loss": st.session_state.loss_values})
            loss_placeholder.line_chart(loss_df.set_index("Epoch"))
            st.session_state.loss_values_df = loss_df

            if st.session_state.get("enable_test_losses_check", False):
                # Evaluate on test data
                if (epoch + 1) % eval_every == 0:
                    test_rmse_loss = 0.0
                    test_msll_loss = 0.0
                    for test_x_batch, test_y_batch in test_loader:
                        #test_rmse, test_msll = self.evaluate(test_x_batch, test_y_batch)
                        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):  # or 2:
                            pred_y_mean, pred_y_var = self.model.predict(test_x_batch)
                            test_rmse = torch.mean((pred_y_mean - test_y_batch) ** 2).sqrt()
            
                            pi = 3.141592653589793
                            test_msll = torch.mean(0.5*torch.log(2*pi*pred_y_var) + ((pred_y_mean - test_y_batch) ** 2)/(2*pred_y_var))

                        test_rmse_loss += test_rmse.item()
                        test_msll_loss += test_msll.item()
                    test_rmse_loss=test_rmse_loss/len(test_loader)
                    test_msll_loss=test_msll_loss/len(test_loader)
                    st.session_state.test_rmse_values.append(test_rmse_loss)
                    st.session_state.test_msll_values.append(test_msll_loss)

                    # Update test metrics plot (if evaluated at least once)
                    st.session_state.test_metrics_df = pd.DataFrame({
                            "Epoch": [i * eval_every for i in range(len(st.session_state.test_rmse_values))],
                            "RMSE": st.session_state.test_rmse_values,
                            "MSLL": st.session_state.test_msll_values
                        })
                    test_metrics_placeholder.line_chart(st.session_state.test_metrics_df.set_index("Epoch"))
                    training_log.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, \
                                            Test RMSE: {test_rmse_loss:.4f}, Test MSLL: {test_msll_loss:.4f}")
            else:
                if (epoch + 1) % eval_every == 0:
                    training_log.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}")
        # Final results
        st.success("Training completed." if not st.session_state.get("stop_training", False) else "Training stopped.")
        st.write("## Final Training State")
        st.line_chart(st.session_state.loss_values_df.set_index("Epoch"))
        if st.session_state.test_rmse_values:
            st.line_chart(st.session_state.test_metrics_df.set_index("Epoch"))

    def evaluate(self, test_x, test_y):
        # Evaluation logic
        self.model.eval()
        '''
        with torch.no_grad():
            predictive_means, predictive_variances, _ = self.model.predict(test_loader)
            test_rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
            print(f"negative RMSE: {-1.0*test_rmse.item()}")
        '''
        with torch.no_grad():
            pred_y_mean, pred_y_var = self.model.predict(test_x)
            test_rmse = torch.mean((pred_y_mean - test_y) ** 2).sqrt().item()
            
            pi = 3.141592653589793
            test_msll = torch.mean(torch.log(2*pi*pred_y_var) + ((pred_y_mean - test_y) ** 2)/(2*pred_y_var)).item()
        return test_rmse, test_msll