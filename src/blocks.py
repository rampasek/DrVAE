"""
Ladislav Rampasek (rampasek@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import layers as lyr
import numpy as np
import math
import inspect
from collections import OrderedDict

'''
Nonlinear functions
'''
nonlinearities = {'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1), 
                  'softplus': nn.Softplus(), 'softsign': nn.Softsign(),
                  'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(0.1),
                  'elu': nn.ELU(), 'selu': nn.SELU()}

'''
Kernel functions
'''
def rbf(x1, x2, gamma=1.):
    new_size = [x2.size()[0]] + list(x1.size())
    d = x1[np.newaxis, :, :].expand(*new_size) - x2[:, np.newaxis, :].expand(*new_size)
    return (-d.pow_(2).sum(2).squeeze_(2) * gamma).exp_().t()

def poly(x1, x2, degree=2, gamma=1., bias=1.):
    return torch.pow(gamma * x1.mm(x2.t()) + bias, degree)

def identity(x1, x2):
    return (x1.mean(0) - x2.mean(0)).pow_(2).sum()

def mmd_fourier(x1, x2, bandwidth=2., dim_r=500):
    """
    Approximate RBF kernel by random features
    """
    if x1.is_cuda:
        rnd_a = Variable(torch.cuda.FloatTensor(x1.size()[1], dim_r).normal_())
        rnd_b = Variable(torch.cuda.FloatTensor(dim_r).uniform_())
    else:
        rnd_a = Variable(torch.FloatTensor(x1.size()[1], dim_r).normal_())
        rnd_b = Variable(torch.FloatTensor(dim_r).uniform_())

    rW_n = math.sqrt(2. / bandwidth) * rnd_a / math.sqrt(x1.size()[1])
    rb_u = 2 * math.pi * rnd_b
    rf0 = math.sqrt(2. / rW_n.size()[1]) * torch.cos(x1.mm(rW_n) + rb_u.expand(x1.size()[0], dim_r))
    rf1 = math.sqrt(2. / rW_n.size()[1]) * torch.cos(x2.mm(rW_n) + rb_u.expand(x2.size()[0], dim_r))
    return ((rf0.mean(0) - rf1.mean(0))**2).sum()

kernels = {'rbf': rbf, 'poly': poly, 'identity': identity, 'rbf_fourier': mmd_fourier}

def mmd_objective(x1, x2, kernel='rbf', bandwidths=1. / (2 * (np.array([1., 2., 5., 8., 10])**2))):
    """
    Return the mmd score between a pair of observations
    """
    K = kernels[kernel]
    if kernel == 'identity':
        return torch.sqrt(K(x1, x2))
    elif kernel == 'rbf_fourier':
        return torch.sqrt(K(x1, x2, bandwidth=2.))

    # possibly mixture of kernels
    x1x1, x1x2, x2x2 = 0, 0, 0
    for bandwidth in bandwidths:
        x1x1 += K(x1, x1, gamma=math.sqrt(x1.size()[1]) * bandwidth) / len(bandwidths)
        x2x2 += K(x2, x2, gamma=math.sqrt(x2.size()[1]) * bandwidth) / len(bandwidths)
        x1x2 += K(x1, x2, gamma=math.sqrt(x1.size()[1]) * bandwidth) / len(bandwidths)

    return torch.sqrt(x1x1.mean() - 2 * x1x2.mean() + x2x2.mean())

def one_hot(y, max_dim):
    """
    One hot encoding of torch tensor @y
    """
    if y is not None and len(y) > 0:
        if y.ndimension() == 1:
            y.data.unsqueeze_(1)
        if y.is_cuda:
            y1inK = Variable(torch.cuda.FloatTensor(y.size()[0], max_dim).zero_())
        else:
            y1inK = Variable(torch.FloatTensor(y.size()[0], max_dim).zero_())
        y1inK.data.scatter_(1, y.data.long(), 1)
    else:
        y1inK = None
    return y1inK


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, nonlin='softplus', weight_norm=False, batch_norm=False, dropout_rate=0., input_dropout_rates=None):
        """
        Deterministic MLP that takes a number of inputs and returns the last hidden layer

        input_dims: List of input sizes
            [x0, x1, ...]
        hidden_dims : List of widths of the hidden layers
            [s0, s1, ...]
        nonlin : Nonlinearity to use
        weight_norm: Use weight normalization https://arxiv.org/abs/1602.07868
        batch_norm: Use batch normalization https://arxiv.org/abs/1502.03167
        dropout_rate: Dropout for hidden layers
        input_dropout_rates: List of dropout rates for inputs, if None then "dropout_rate-0.3" is used
            e.g. [0.2, 0.2, ...]
        """
        super(MLP, self).__init__()

        # set parameters as instance variables
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop('self')
        for arg, val in values.items():
            setattr(self, arg, val)

        # use standard linear layer or weight-normed linear layer
        if weight_norm:
            nnlayer = lyr.WeightNormLinear
        else:
            nnlayer = nn.Linear

        # Dropout on the inputs
        if input_dropout_rates is None:
            input_dropout_rates = [max(0., dropout_rate - 0.3)] * len(input_dims)
        if len(input_dropout_rates) != len(input_dims):
            raise ValueError("MLP: input_dropout_rates is not the same length as input_dims "
                             + str(input_dropout_rates) + " " + str(input_dims))
        self.input_dropouts = nn.ModuleList([nn.Dropout(p=input_dropout_rates[i]) for i in range(len(input_dims))])
        self.input_dropout_rates = input_dropout_rates

        ## MLP model
        modules = OrderedDict()
        prev_layer_size = int(np.asarray(input_dims).sum())
        if batch_norm:
            modules['bn_input'] = nn.BatchNorm1d(prev_layer_size, affine=True)
        for i, layer_size in enumerate(hidden_dims):
            # Dropout
            if i > 0 and dropout_rate > 0.:
                modules['dropout'+str(i+1)] = nn.Dropout(p=dropout_rate)
            # Linear
            modules['linear'+str(i+1)] = nnlayer(prev_layer_size, layer_size)
            # Activation
            modules['activ'+str(i+1)] = nonlinearities[nonlin]
            # BatchNorm without scale&bias (no affine transformation after normalization)
            if batch_norm:
                modules['bn'+str(i+1)] = nn.BatchNorm1d(layer_size, affine=True)
            prev_layer_size = layer_size
        self.model = nn.Sequential(modules)

    def forward(self, inputs):
        """
        Feedforward the inputs
        """
        ## apply input dropout and concatenate to one tensor
        assert(len(inputs) == len(self.input_dims))
        if np.any(self.input_dropout_rates):
            x = [self.input_dropouts[i](inp) for i, inp in enumerate(inputs)]
        x = torch.cat(inputs, 1)
        ## feedforward through the model
        output = self.model(x)
        return output

class GaussianLogVarMixin:
    """
    For Gaussian distributions parametrized by mu and log(sigma^2) (log variance)
    """
    def sample(self, mu, logvar):
        '''sample from the gaussian by reparametrzation'''
        eps = Variable(mu.data.new(mu.size()).normal_())
        std = logvar.mul(0.5).exp_()
        return tuple([eps * std + mu])

    def kldivergence(self, mu_q, logvar_q, mu_p, logvar_p):
        '''KL( q || p ) summed over the batch of qs and ps'''
        return self.kldivergence_perx(mu_q, logvar_q, mu_p, logvar_p).sum()

    def kldivergence_perx(self, mu_q, logvar_q, mu_p, logvar_p):
        '''KL( q || p ) computed for each q and p'''
        return -.5 * torch.sum(1 - logvar_p + logvar_q - ((mu_q - mu_p)**2 + logvar_q.exp()) / logvar_p.exp(), dim=1)

    def kldivergence_from_prior(self, mu, logvar):
        '''KL( q || prior p ) summed over the batch of qs'''
        return self.kldivergence_from_prior_perx(mu, logvar).sum()

    def kldivergence_from_prior_perx(self, mu, logvar):
        '''KL( q || prior p ) computed for each q'''
        return self.kldivergence_perx(mu, logvar, self.prior_mu.expand_as(mu), self.prior_lv.expand_as(logvar))

    def logp(self, sample, mu, logvar):
        return self.logp_perx(sample, mu, logvar).sum()

    def logp_perx(self, sample, mu, logvar):
        return -.5 * torch.sum(float(np.log(2 * np.pi)) + logvar + ((sample - mu)**2) / logvar.exp(), dim=1)

    def logp_prior(self, sample):
        return self.logp_prior_perx(sample).sum()

    def logp_prior_perx(self, sample):
        return self.logp_perx(sample, self.prior_mu.expand_as(sample), self.prior_lv.expand_as(sample))

class GaussianSigmaMixin:
    """
    For Gaussian distributions parametrized by mu and sigma (standard deviation)
    """
    def sample(self, mu, std):
        '''sample from the gaussian by reparametrzation'''
        eps = Variable(mu.data.new(mu.size()).normal_())
        return tuple([eps * std + mu])

    def kldivergence(self, mu_q, std_q, mu_p, std_p):
        '''KL( q || p ) summed over the batch of qs and ps'''
        return self.kldivergence_perx(mu_q, std_q, mu_p, std_p).sum()

    def kldivergence_perx(self, mu_q, std_q, mu_p, std_p):
        '''KL( q || p ) computed for each q and p'''
        return -.5 * torch.sum(1. - torch.log(std_p**2) + torch.log(std_q**2) -
                           ((mu_q - mu_p)**2 + std_q**2) / std_p**2, dim=1)

    def kldivergence_from_prior(self, mu, std):
        '''KL( q || prior p ) summed over the batch of qs'''
        return self.kldivergence_from_prior_perx(mu, std).sum()

    def kldivergence_from_prior_perx(self, mu, std):
        '''KL( q || prior p ) computed for each q'''
        return self.kldivergence_perx(mu, std, self.prior_mu.expand_as(mu), self.prior_sg.expand_as(std))

    def logp(self, sample, mu, std):
        return self.logp_perx(sample, mu, std).sum()

    def logp_perx(self, sample, mu, std):
        return -.5 * torch.sum(float(np.log(2 * np.pi)) + torch.log(std**2) + ((sample - mu)**2) / (std**2), dim=1)

    def logp_prior(self, sample):
        return self.logp_prior_perx(sample).sum()

    def logp_prior_perx(self, sample):
        return self.logp_perx(sample, self.prior_mu.expand_as(sample), self.prior_sg.expand_as(sample))


class DiagGaussianModule(GaussianLogVarMixin, nn.Module):
    """
    Encoder that maps inputs to a latent Gaussian distribution
    """
    def __init__(self, input_dims, hidden_dims, output_dim, nonlin='softplus', weight_norm=False, batch_norm=False,
                dropout_rate=0., input_dropout_rates=None, prior_mu=0., prior_sg=1., constrain_means=False, fixed_variance=None):
        super(DiagGaussianModule, self).__init__()

        ## Neural Net parametrizing the latent Gaussian distribution
        self.nnet = MLP(input_dims=input_dims, hidden_dims=hidden_dims, nonlin=nonlin,
                        weight_norm=weight_norm, batch_norm=batch_norm,
                        dropout_rate=dropout_rate, input_dropout_rates=input_dropout_rates)
        self.output_dim = output_dim
        self.constrain_means = constrain_means
        self.fixed_variance = fixed_variance
        if fixed_variance is not None:
            self.fixed_variance = Variable(torch.zeros(1) + fixed_variance).log()

        # use standard linear layer or weight-normed linear layer
        if weight_norm:
            nnlayer = lyr.WeightNormLinear
        else:
            nnlayer = nn.Linear

        ## Mean and Log Variance modules
        if len(hidden_dims) > 0:
            prev_layer_dim = hidden_dims[-1]
        else:
            prev_layer_dim = int(np.asarray(input_dims).sum())
        modules_mu, modules_lv = OrderedDict(), OrderedDict()
        # Dropout
        if dropout_rate > 0.:
            modules_mu['dropout_mu'] = nn.Dropout(p=dropout_rate)
            modules_lv['dropout_lv'] = nn.Dropout(p=dropout_rate)
        # Linear
        modules_mu['linear_mu'] = nnlayer(prev_layer_dim, output_dim)
        modules_lv['linear_lv'] = nnlayer(prev_layer_dim, output_dim)
        # Activation
        if constrain_means: # constrain means to be (0,1)
            modules_mu['activ_mu'] = nonlinearities['sigmoid']
        
        ## Mean and Log Variance functions
        self.encoder_mu = nn.Sequential(modules_mu)
        self.encoder_lv = nn.Sequential(modules_lv)

        self.prior_mu = Variable(torch.zeros(1) + prior_mu)
        self.prior_lv = Variable(torch.zeros(1) + prior_sg**2).log()

    def forward(self, inputs):
        # feedforward the deterministic MLP
        h = self.nnet(inputs)
        # Mean and LogVar encoders
        mu = self.encoder_mu(h)
        logvar = self.encoder_lv(h) - 2.
        if self.fixed_variance is not None:
            logvar = self.fixed_variance.expand_as(mu)
        # logvar = torch.log(self.encoder_lv(h).exp() + 1e-7)
        # logvar = torch.clamp(self.encoder_lv(h), min=-12)
        return mu, logvar


class DiagGaussianModuleLinear(GaussianLogVarMixin, nn.Module):
    """
    Encoder that maps inputs to a latent Gaussian distribution
    """
    def __init__(self, input_dims, hidden_dims, latent_dim, nonlin='softplus', weight_norm=False, batch_norm=False,
                dropout_rate=0., input_dropout_rates=None, prior_mu=0., prior_sg=1., constrain_means=False, bias_only=False):
        super(DiagGaussianModuleLinear, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.nonlin = nonlin
        self.weight_norm = weight_norm
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.input_dropout_rates = input_dropout_rates

        ## Neural Net parametrizing the latent Gaussian distribution
        self.constrain_means = constrain_means
        self.bias_only = bias_only

        ## must not change dimensionality
        assert(len(input_dims) == 1 and input_dims[0] == latent_dim)
        prev_layer_dim = int(np.asarray(input_dims).sum())

        modules_lv = OrderedDict()
        # Dropout
        if dropout_rate > 0.:
            modules_lv['dropout_lv'] = nn.Dropout(p=dropout_rate)
        # Linear
        modules_lv['linear_lv'] = nn.Linear(prev_layer_dim, latent_dim)
        # Activation
        if constrain_means: # constrain means to be (0,1)
            modules_mu['activ_mu'] = nonlinearities['sigmoid']
        ## parameters for mu function
        fan_in_fan_out = prev_layer_dim + latent_dim
        self.W_mu = nn.Parameter(torch.Tensor(latent_dim, latent_dim).uniform_(-0.0001, 0.0001)) #.normal_(0, math.sqrt(2. / fan_in_fan_out))) # Glorot initialisation
        # self.W_mu2 = nn.Parameter(torch.Tensor(latent_dim, latent_dim).uniform_(-0.0001, 0.0001)) #.normal_(0, math.sqrt(2. / fan_in_fan_out))) # Glorot initialisation
        self.bias_mu = nn.Parameter(torch.Tensor(latent_dim).uniform_(-0.0001, 0.0001))
        # self.bias_mu2 = nn.Parameter(torch.Tensor(latent_dim).uniform_(-0.0001, 0.0001))

        ## Log Variance function
        self.encoder_lv = nn.Sequential(modules_lv)

        self.prior_mu = Variable(torch.zeros(1) + prior_mu)
        self.prior_lv = Variable(torch.zeros(1) + prior_sg**2).log()

    def forward(self, inputs):
        assert(len(inputs) == len(self.input_dims))
        x = torch.cat(inputs, 1)

        # Mean and LogVar functions
        if self.bias_only:
            mu = x + self.bias_mu.expand_as(x)
        else:
            mu = x + F.linear(x, self.W_mu) + self.bias_mu.expand_as(x)
            # h = F.elu(F.linear(x, self.W_mu) + self.bias_mu.expand_as(x))
            # mu = x + F.elu(F.linear(h, self.W_mu2) + self.bias_mu2.expand_as(x))
        logvar = self.encoder_lv(x) - 2.
        return mu, logvar


class DiagGaussianSigmaModule(GaussianSigmaMixin, nn.Module):
    """
    Encoder that maps inputs to a latent Gaussian distribution
    """
    def __init__(self, input_dims, hidden_dims, latent_dim, nonlin='softplus', weight_norm=False, batch_norm=False,
                dropout_rate=0., input_dropout_rates=None, prior_mu=0., prior_sg=1., constrain_means=False):
        super(DiagGaussianSigmaModule, self).__init__()

        ## Neural Net parametrizing the latent Gaussian distribution
        self.nnet = MLP(input_dims=input_dims, hidden_dims=hidden_dims, nonlin=nonlin,
                        weight_norm=weight_norm, batch_norm=batch_norm,
                        dropout_rate=dropout_rate, input_dropout_rates=input_dropout_rates)
        self.latent_dim = latent_dim
        self.constrain_means = constrain_means

        # use standard linear layer or weight-normed linear layer
        if weight_norm:
            nnlayer = lyr.WeightNormLinear
        else:
            nnlayer = nn.Linear

        ## Mean and Sigma modules
        if len(hidden_dims) > 0:
            prev_layer_dim = hidden_dims[-1]
        else:
            prev_layer_dim = int(np.asarray(input_dims).sum())
        modules_mu, modules_sg = OrderedDict(), OrderedDict()
        # Dropout
        if dropout_rate > 0.:
            modules_mu['dropout_mu'] = nn.Dropout(p=dropout_rate)
            modules_sg['dropout_sg'] = nn.Dropout(p=dropout_rate)
        # Linear
        modules_mu['linear_mu'] = nnlayer(prev_layer_dim, latent_dim)
        modules_sg['linear_sg'] = nnlayer(prev_layer_dim, latent_dim)
        # Activation
        if constrain_means: # constrain means to be (0,1)
            modules_mu['activ_mu'] = nonlinearities['sigmoid']
        modules_sg['activ_sg'] = nonlinearities['softplus']
        
        ## Mean and Sigma functions
        self.encoder_mu = nn.Sequential(modules_mu)
        self.encoder_sg = nn.Sequential(modules_sg)

        self.prior_mu = Variable(torch.zeros(1) + prior_mu)
        self.prior_sg = Variable(torch.zeros(1) + prior_sg)

    def forward(self, inputs):
        # feedforward the deterministic MLP
        h = self.nnet(inputs)
        # Mean and Sigma encoders
        mu = self.encoder_mu(h)
        std = self.encoder_sg(h) + 1e-3
        return mu, std


class CategoricalDecoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, reconstruction_dim, nonlin='softplus', weight_norm=False, batch_norm=False,
                dropout_rate=0., input_dropout_rates=None):
        super(CategoricalDecoder, self).__init__()

        ## Neural Net parametrizing the latent Gaussian distribution
        self.nnet = MLP(input_dims=input_dims, hidden_dims=hidden_dims, nonlin=nonlin,
                        weight_norm=weight_norm, batch_norm=batch_norm,
                        dropout_rate=dropout_rate, input_dropout_rates=input_dropout_rates)
        self.reconstruction_dim = reconstruction_dim

        # use standard linear layer or weight-normed linear layer
        if weight_norm:
            nnlayer = lyr.WeightNormLinear
        else:
            nnlayer = nn.Linear

        ## P decoder modules
        if len(hidden_dims) > 0: 
            prev_layer_dim = hidden_dims[-1]
        else:
            prev_layer_dim = int(np.asarray(input_dims).sum())
        modules = OrderedDict()
        # Dropout
        if dropout_rate > 0.:
            modules['dropout_p'] = nn.Dropout(p=dropout_rate)
        # Linear
        modules['linear_p'] = nnlayer(prev_layer_dim, reconstruction_dim)
        # Activation
        if self.reconstruction_dim > 1:
            modules['activ_p'] = nn.Softmax(dim=-1)
        else:
            modules['activ_p'] = nn.Sigmoid()
        
        ## Class P decoder
        self.decoder_p = nn.Sequential(modules)
    
    def forward(self, inputs):
        # feedforward the deterministic MLP
        h = self.nnet(inputs)
        # Class P decoder
        ps = self.decoder_p(h)
        if self.reconstruction_dim == 1:
            ps = torch.cat((1.-ps, ps), 1)
        return [torch.clamp(ps, min=1e-10, max=1.-1e-10)]

    def sample(self, ps):
        s = torch.multinomial(ps, 1)
        return s

    def logp(self, x, ps):
        # return -F.nll_loss(ps.log(), x.view(-1).long(), size_average=False)
        return torch.sum(self.logp_perx(x, ps))

    def logp_perx(self, x, ps):
        return -F.nll_loss(ps.log(), x.view(-1).long(), size_average=False, reduce=False)

    def entropy(self, ps):
        return - (ps * torch.log(ps)).sum()

    def kldivergence_perx(self, ps, prior):
        return - ps * (torch.log(prior).expand_as(ps) - torch.log(ps))

    def kldivergence(self, ps, prior):
        return self.kldivergence_perx(ps, prior).sum()

    def most_probable(self, ps):
        return torch.max(ps, dim=1)[1]
