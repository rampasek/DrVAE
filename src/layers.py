import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

class WeightNormLinear(nn.Linear):
    """
    Linear Layer with Weight Normalization
    (Salimans & Kingma, 2016) https://arxiv.org/abs/1602.07868

    code modified from
    https://github.com/openai/weightnorm
    and https://github.com/ruotianluo/weightnorm-pytorch
    """
    def __init__(self, in_features, out_features, data_init=False, init_scale=1.):
        super(WeightNormLinear, self).__init__(in_features, out_features, bias=True)
        if data_init: 
            self.g = Parameter(torch.Tensor(out_features).fill_(float('nan')))
        else:
            self.g = Parameter(torch.Tensor(out_features).zero_() + 1.)
        self.init_scale = init_scale

    def forward(self, x):
        # if g is all NaN then initialize
        if self.g.data.ne(self.g.data).all():
            # data based initialization of parameters
            self.weight.data = torch.randn(self.weight.size()).type_as(self.weight.data) * 0.05 # out_features * in_features
            W_norm = self.weight / torch.norm(self.weight, 2, 1).expand_as(self.weight) # norm is out_features * 1
            out = F.linear(x, W_norm).data # batch_size * out_features
            mu, var = out.mean(0).squeeze(0), out.var(0).squeeze(0) # out_features
            scale = self.init_scale / torch.sqrt(var + 1e-10) # out_features
            self.g.data = scale
            self.bias.data = -mu*scale

        # use weight normalization
        out = F.linear(x, self.weight) # batch_size * out_features
        scalar = self.g / torch.norm(self.weight, 2, 1)
        out = scalar.expand_as(out)*out + self.bias.expand_as(out)
        return out


class MaskedLinear(nn.Linear):
    """
    Masked Linear layer for Masked autoencoder (MADE)

    https://github.com/mgermain/MADE
    https://gist.github.com/taku-y/43550e688f4020ac7da15a181d262f2fs
    """
    def __init__(self, in_features, out_features, m_pre, output_layer, rev_order=False):
        super(MaskedLinear, self).__init__(np.asarray(in_features).sum(), out_features, bias=True)
        self.output_layer = output_layer
        self.rev_order = rev_order
        self.m_pre = m_pre
        if isinstance(in_features, tuple) or isinstance(in_features, list):                
            x_dim = in_features[0]
            h_dim = np.asarray(in_features[1:]).sum()
        else:
            x_dim = in_features
            h_dim = 0
        if self.m_pre is None:
            # Mask indices for the input layer
            self.m_pre = np.arange(1, x_dim + 1).astype(int)
            if rev_order:
                self.m_pre = self.m_pre[::-1]
            if h_dim > 0:
                self.m_pre = np.concatenate((self.m_pre, np.asarray([1] * h_dim, dtype=int)))

        m, mask = self._create_made_mask(x_dim + h_dim, out_features, self.m_pre, self.output_layer, self.rev_order)
        ## max number of inputs to each unit at this layer, to be used to construct next layer
        self.m = m
        ## weight matrix mask
        self.mask = Variable(torch.FloatTensor(mask))

    def get_m(self):
        """
        Return max number of inputs to each unit at this layer,
        to be used to correctly construct next layer of MADE such that
        it respects the conditional connectivity
        """
        return self.m

    def _create_made_mask(self, d_pre, d, m_pre, output_layer, rev_order=False):
        """Create a mask for MADE.
        
        Parameters
        ----------
        d_pre : int
            The number of rows in the weight matrix. 
        d : int
            The number of columns in the weight matrix. 
        m_pre : numpy.ndarray, shape=(d_pre,)
            The number of inputs to the units in the previous layer.
        output_layer : bool
            True for the output layer. 
        rev_order : bool
            If true, the order of connectivity constraints is reversed. 
            It is used only for the output layer. 
        
        Returns
        -------
        m, mask : (numpy.ndarray, numpy.ndarray)
            Mask indices and Mask.

        Code by "taku-y":
        https://github.com/pymc-devs/pymc3/issues/1438
        https://gist.github.com/taku-y/43550e688f4020ac7da15a181d262f2f
        """
        d_input = np.max(m_pre)
        mask = np.zeros((d_pre, d)).astype('float32')

        if not output_layer:
            m = np.arange(1, d_input).astype(int)
            # if len(m) < d:
            #     m = np.hstack((m, (d_input - 1) * np.ones(d - len(m))))
            while len(m) < d:
                m = np.hstack((m, m))
            m = m[:d]
                
            for ix_col in range(d):
                ixs_row = np.where(m_pre <= m[ix_col])[0]
                (mask[:, ix_col])[ixs_row] = 1

        else:
            m = np.arange(1, d + 1)
            if rev_order:
                m = m[::-1]
            for ix_col in range(d):
                ixs_row = np.where(m_pre < m[ix_col])[0]
                mask[ixs_row, ix_col] = 1
                
        return m, mask

        def forward(self, x):
            # use weight masking
            out = F.linear(x, self.mask * self.weight)
            out = out + self.bias.expand_as(out)
            return out


