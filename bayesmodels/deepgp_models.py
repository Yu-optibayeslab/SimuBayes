import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

'Defined the basic deepGP hidden layer, which can be used to construct the deep GP model'

class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=256, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

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

        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'zero':
            self.mean_module =  gpytorch.means.ZeroMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

class DeepGP_2HLs(DeepGP):
    def __init__(self, train_x_shape, num_hidden_dims=[4,2], num_inducing_points=256):
        hidden_layer_1 = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            #input_dims = 4,
            output_dims=num_hidden_dims[0],
            num_inducing=num_inducing_points,
            mean_type='zero',
        )
        
        hidden_layer_2 = DeepGPHiddenLayer(
            input_dims=hidden_layer_1.output_dims,
            #input_dims = 4,
            output_dims=num_hidden_dims[1],
            num_inducing=num_inducing_points,
            mean_type='zero',
        )
        
        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer_2.output_dims,
            output_dims=None,
            num_inducing=num_inducing_points,
            mean_type='zero',
        )

        super().__init__()

        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer_1(inputs)
        hidden_rep2 = self.hidden_layer_2(hidden_rep1)
        output = self.last_layer(hidden_rep2)
        return output

    def predict(self, test_loader, model):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

class DeepGP_3HLs(DeepGP):
    def __init__(self, train_x_shape, num_hidden_dims=[6,4,2], num_inducing_points=256):
        #input_dims = train_x_shape[-1]
        
        hidden_layer_0 = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            num_inducing=NXu,
            output_dims=num_hidden_dims[0],
            mean_type='zero',
        )
        
        hidden_layer_1 = DeepGPHiddenLayer(
            input_dims=hidden_layer_0.output_dims,
            num_inducing=NXu,
            output_dims=num_hidden_dims[1],
            mean_type='zero',
        )
        
        hidden_layer_2 = DeepGPHiddenLayer(
            input_dims=hidden_layer_1.output_dims,
            num_inducing=NXu,
            output_dims=num_hidden_dims[2],
            mean_type='zero',
        )
        
        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer_2.output_dims,
            num_inducing=NXu,
            output_dims=None,
            mean_type='zero',
        )

        super().__init__()
        
        self.hidden_layer_0 = hidden_layer_0
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep0 = self.hidden_layer_0(inputs)
        hidden_rep1 = self.hidden_layer_1(hidden_rep0)
        hidden_rep2 = self.hidden_layer_2(hidden_rep1)
        output = self.last_layer(hidden_rep2)
        return output

    def predict(self, test_loader, model):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
