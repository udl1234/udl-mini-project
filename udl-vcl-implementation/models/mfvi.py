import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal

from utils.misc import weighted_kl_divergence


class Layer(nn.Module):
    """Layer for mean-field variational inference."""

    def __init__(self, input_dim, output_dim, prior_mean, prior_var):
        super(Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.prior_mean = prior_mean
        self.prior_var = prior_var

        # Prior
        self.register_buffer("prior_w_mu", torch.empty(output_dim, input_dim))
        self.register_buffer("prior_w_var", torch.empty(output_dim, input_dim))
        self.register_buffer("prior_b_mu", torch.empty(output_dim))
        self.register_buffer("prior_b_var", torch.empty(output_dim))

        # Variational posterior q
        self.q_w_mu = nn.Parameter(torch.empty(output_dim, input_dim))
        self.q_w_logvar = nn.Parameter(torch.empty(output_dim, input_dim))

        self.q_b_mu = nn.Parameter(torch.empty(output_dim))
        self.q_b_logvar = nn.Parameter(torch.empty(output_dim))

    def reset_parameters(self):
        # self.reset_posterior()
        nn.init.trunc_normal_(self.q_w_mu, std=0.1)
        nn.init.constant_(self.q_w_logvar, -6)

        nn.init.trunc_normal_(self.q_b_mu, std=0.1)
        nn.init.constant_(self.q_b_logvar, -6)

        self.prior_w_mu = torch.full_like(self.prior_w_mu, self.prior_mean)
        self.prior_w_var = torch.full_like(self.prior_w_var, self.prior_var)

        self.prior_b_mu = torch.full_like(self.prior_b_mu, self.prior_mean)
        self.prior_b_var = torch.full_like(self.prior_b_var, self.prior_var)

    def reset_posterior(self):
        # nn.init.trunc_normal_(self.prior_w_mu, std=0.1)
        # nn.init.trunc_normal_(self.q_w_logvar, std=0.1)
        # self.q_w_logvar.data = self.prior_w_var.mul(2).log().clone().detach()

        # nn.init.trunc_normal_(self.prior_b_mu, std=0.1)
        # nn.init.trunc_normal_(self.q_b_logvar, std=0.1)
        # self.q_b_logvar.data = self.prior_b_var.mul(2).log().clone().detach()
        # pass

        nn.init.trunc_normal_(self.q_w_mu, std=0.1)
        nn.init.constant_(self.q_w_logvar, -6)

        nn.init.trunc_normal_(self.q_b_mu, std=0.1)
        nn.init.constant_(self.q_b_logvar, -6)

    def update_prior(self):
        self.prior_w_mu = self.q_w_mu.data.clone().detach()
        self.prior_w_var = self.q_w_logvar.data.exp().clone().detach()
        self.prior_b_mu = self.q_b_mu.data.clone().detach()
        self.prior_b_var = self.q_b_logvar.data.exp().clone().detach()

    @property
    def prior_w(self):
        """prior weight distribution"""
        return Normal(self.prior_w_mu, self.prior_w_var.sqrt())

    @property
    def prior_b(self):
        """prior bias distribution"""
        return Normal(self.prior_b_mu, self.prior_b_var.sqrt())

    @property
    def q_w(self):
        """variational weight posterior"""
        return Normal(self.q_w_mu, (0.5 * self.q_w_logvar).exp())

    @property
    def q_b(self):
        """variational bias posterior"""
        return Normal(self.q_b_mu, (0.5 * self.q_b_logvar).exp())

    def kl(self):

        # x_w = Normal(torch.zeros_like(self.q_w_mu), torch.ones_like(self.q_w_logvar))
        # x_b = Normal(torch.zeros_like(self.q_b_mu), torch.ones_like(self.q_b_logvar))

        weight_kl = kl_divergence(self.q_w, self.prior_w)
        bias_kl = kl_divergence(self.q_b, self.prior_b)

        # alpha = self.prior_w.mean.abs().sum(dim=1) + self.prior_b.mean.abs()

        # weight_kl *= torch.softmax(alpha, dim=0).unsqueeze(1)
        # bias_kl *= torch.softmax(alpha, dim=0)
        # weight_kl *= torch.softmax(self.prior_w.mean.abs(), dim=0)
        # bias_kl *= torch.softmax(self.prior_b.mean.abs(), dim=0)

        # torch.norm

        # min_val =

        # weight_kl *= (self.prior_w.mean.abs() - min_val) / (max_val - min_val)
        # bias_kl *= (self.prior_b.mean.abs() - min_val) / (max_val - min_val)

        # weight_kl_init = kl_divergence(self.prior_w, x_w)
        # bias_kl_init = kl_divergence(self.prior_b, x_b)

        # weight_kl /= weight_kl_init +1
        # bias_kl /= bias_kl_init +1

        # weight_kl = weight_kl - weight_kl_init
        # bias_kl = bias_kl - bias_kl_init

        # weight_kl *= torch.softmax(-weight_kl_init, dim=-1)
        # bias_kl *= torch.softmax(-bias_kl_init, dim=-1)

        # weight_kl += kl_divergence(self.q_w, x_w)
        # bias_kl += kl_divergence(self.q_b, x_b)

        # return weight_kl.sum() + bias_kl.sum()

        # print(weight_kl.shape, bias_kl.shape)
        # print(weight_kl.sum(dim=-1).shape)

        # comb_kl = weight_kl.sum(dim=-1).add(bias_kl).add(1)

        # return comb_kl.log().pow(MFVI.kl_alpha).mul(comb_kl).sub(1).sum()
        # return comb_kl.pow(MFVI.kl_alpha).sum() - bias_kl.shape[-1]
        # return comb_kl.pow(MFVI.kl_alpha).sum() - 1

        # return weight_kl.sum() + bias_kl.sum()

        return weight_kl.sum() + bias_kl.sum()

        factor = torch.tensor(
            MFVI.kl_alpha, device=weight_kl.device, dtype=weight_kl.dtype
        )
        # factor = factor.div(2)
        # factor = 1
        weight_kl = weight_kl.sum().add(1)
        # weight_kl = weight_kl.log1p().mul(weight_kl)
        weight_kl = weight_kl.pow(factor)
        # weight_kl = weight_kl.mul(weight_kl.log().pow(factor))

        bias_kl = bias_kl.sum().add(1)
        # bias_kl = bias_kl.log1p().mul(bias_kl)
        bias_kl = bias_kl.pow(factor)
        # bias_kl = bias_kl.mul(bias_kl.log().pow(factor))

        return weight_kl + bias_kl - 2

        # return weight_kl.sum() + bias_kl.sum()

        # print(weight_kl.shape, bias_kl.shape)
        # print(weight_kl.sum().shape)
        # print(self.prior_w.variance.shape)

        # weight_kl_new = weighted_kl_divergence(self.q_w, self.prior_w)
        # bias_kl_new = weighted_kl_divergence(self.q_b, self.prior_b)
        # print(weight_kl_new.shape)
        # print(weight_kl_new.sum().shape)
        # print(weight_kl_new.sum(), weight_kl.sum())
        # print("!!!!!")

        # return weight_kl.sum() + bias_kl.sum()
        # return weight_kl_new.sum() + bias_kl_new.sum()

    def forward(self, x, deterministic):

        if deterministic:
            return torch.einsum("nbi,oi->nbo", x, self.q_w.mean) + self.q_b.mean

        num_samples = x.shape[0]
        weights = self.q_w.rsample(
            (num_samples,)
        )  # (num_samples, input_dim, output_dim)
        biases = self.q_b.rsample((num_samples,))  # (num_samples, output_dim)

        return torch.einsum("nbi,noi->nbo", x, weights) + biases.unsqueeze(1)

    def __repr__(self):
        return f"MeanFieldLayer({self.input_dim}, {self.output_dim})"


class MFVI(nn.Module):
    """Mean-field variational inference neural network."""

    kl_alpha = 1

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation=nn.ReLU(),
        prior_mean=0,
        prior_var=1,
        num_heads=1,
    ):
        super(MFVI, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.activation = activation

        self.hidden_layers = nn.ModuleList()
        last_size = input_dim
        for hs in hidden_dims:
            layer = Layer(last_size, hs, prior_mean, prior_var)
            self.hidden_layers.append(layer)
            last_size = hs

        self.output_layers = nn.ModuleList()
        for _ in range(num_heads):
            self.output_layers.append(
                Layer(last_size, output_dim, prior_mean, prior_var)
            )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hidden_layers:
            layer.reset_parameters()
        for layer in self.output_layers:
            layer.reset_parameters()

    def reset_posterior(self):
        for layer in self.hidden_layers:
            layer.reset_posterior()
        for layer in self.output_layers:
            layer.reset_posterior()

    def reset_head(self, head):
        assert 0 <= head < self.num_heads, "Invalid head index."
        self.output_layers[head].reset_parameters()

    def update_prior(self):
        for layer in self.hidden_layers:
            layer.update_prior()
        for layer in self.output_layers:
            layer.update_prior()

    def forward(self, x, num_samples=10, deterministic=False, head=0):
        assert 0 <= head < self.num_heads, "Invalid head index."

        # x = torch.flatten(x, 1)  # (batch_size, input_dim)

        x = x.expand(num_samples, -1, -1)  # (num_samples, batch_size, input_dim).

        for layer in self.hidden_layers:
            x = layer(x, deterministic)
            x = self.activation(x)

        x = self.output_layers[head](
            x, deterministic
        )  # (num_samples, batch_size, output_dim)

        return x

    def kl(self):
        kl = 0.0
        factor = MFVI.kl_alpha

        for layer in self.hidden_layers:
            kl += layer.kl() #* 30  # * factor
            # factor /= 2
        # for layer in self.output_layers:
        #     kl += layer.kl()
        if self.num_heads == 1:
            kl += self.output_layers[0].kl() #* 10  # * factor  # * factor #* 10

        # factor = torch.tensor(MFVI.kl_alpha, device=kl.device, dtype=kl.dtype)
        # if factor > 2:
        # kl = torch.square(kl + 1) - 1
        # kl = kl * factor * torch.log1p(kl)
        # kl = kl * torch.log1p(kl).pow(factor)
    
        # kl = kl.add(1)
        # kl = kl.pow(factor).mul(kl.log()).sub(1)

        # return kl.mul(kl.log())
        # return kl
        # return kl.pow(factor)
        # kl = kl.add(1).pow(factor).sub(1)
        return kl * factor

    def _prepare_logits(self, x, num_samples, deterministic, head):
        if deterministic:
            num_samples = 1

        y_pred = self.forward(
            x, num_samples, deterministic, head
        )  # (num_samples, batch_size, output_dim)
        y_pred_softmax = F.log_softmax(y_pred, dim=-1)
        y_pred_log = (
            torch.logsumexp(y_pred_softmax, dim=0)
            - torch.tensor(num_samples, dtype=torch.float32, device=x.device).log()
        )  # (batch_size, output_dim)

        return y_pred_log

    def nll(self, x, y, num_samples=10, deterministic=False, head=0):
        logits = self._prepare_logits(x, num_samples, deterministic, head)
        nll = F.nll_loss(logits, y, reduction="mean")
        return nll

    def predict(self, x, num_samples=100, deterministic=False, head=0):
        logits = self._prepare_logits(x, num_samples, deterministic, head)
        predictions = torch.argmax(logits, dim=-1)  # (batch_size)

        probabilities = logits.gather(1, predictions.unsqueeze(1)).squeeze(1).exp()

        return predictions, probabilities

    def dump_parameters(self):
        params = {}
        for i, layer in enumerate(self.hidden_layers):
            params[f"hidden_{i}_w_mu"] = layer.q_w_mu.data
            params[f"hidden_{i}_w_var"] = layer.q_w_logvar.data.exp()
            params[f"hidden_{i}_b_mu"] = layer.q_b_mu.data
            params[f"hidden_{i}_b_var"] = layer.q_b_logvar.data.exp()

        for i, layer in enumerate(self.output_layers):
            params[f"output_{i}_w_mu"] = layer.q_w_mu.data
            params[f"output_{i}_w_var"] = layer.q_w_logvar.data.exp()
            params[f"output_{i}_b_mu"] = layer.q_b_mu.data
            params[f"output_{i}_b_var"] = layer.q_b_logvar.data.exp()

        return params

    def print_params(self):
        for i, layer in enumerate(self.hidden_layers):
            print(f"Hidden Layer {i}:")
            print(f"Weight mu: {layer.q_w_mu.data}")
            print(f"Weight var: {layer.q_w_logvar.data}")
            print(f"Bias mu: {layer.q_b_mu.data}")
            print(f"Bias var: {layer.q_b_logvar.data}")

        for i, layer in enumerate(self.output_layers):
            print(f"Output Layer {i}:")
            print(f"Weight mu: {layer.q_w_mu.data}")
            print(f"Weight var: {layer.q_w_logvar.data}")
            print(f"Bias mu: {layer.q_b_mu.data}")
            print(f"Bias var: {layer.q_b_logvar.data}")
