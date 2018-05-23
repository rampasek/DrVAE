"""
Ladislav Rampasek (rampasek@gmail.com)
Acknowledgements:
Christos Louizos's VFAE theano implementation https://arxiv.org/abs/1511.00830
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import inspect
import time
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import blocks as blk
import utils as utl
from DGMMixin import DeepGenerativeModelMixin


class DrVAE(DeepGenerativeModelMixin, nn.Module):

    """
    Drug Response Variational Autoencoder

    Implements a generative model with two layers of stochastic variables + 1 sequence step:
        p(x1, x2, z1, z2, z3, y | s) = p(z3) * p(y) * p(z1|z3,y) * p(z2|z1) * p(x1|z1,s) * p(x2|z2,s)
    with q(z1|x1,s) q(z2|x2,s) q(y|z1,z2) q(z3|z1,y) being the variational posteriors,
    where q(z2|x2,s) is shared with q(z1|x1,s) and if x2 is not observed than q(z2) = p(z2|z1) * q(z1|x1,s)

    Dr.VAE: Drug Response Variational Autoencoder
    Rampasek, Hidru, Smirnov, Haibe-Kains, Goldenberg (arxiv preprint)
    https://arxiv.org/abs/1706.08203

    Furthermore there can be an extra MMD penalty on z1 to further enforce independence between z1 and s, following VFAE
    Louizos, Swersky, Li, Welling, Zemel. The Variational Fair Autoencoder. ICLR 2016
    https://arxiv.org/abs/1511.00830
    """

    def __init__(self, 
                 dim_x,   # data space dim
                 dim_s,   # nuisance variable dim (number of classes)
                 dim_y,   # number of classes
                 dim_c=1, # concentration (usually one scalar per data pair)
                 dim_m=1, # dim of molecular features
                 dim_h_en_z1=(50, 50),  # q(z1|x1,s) also used as q(z2|x2,s)
                 dim_h_de_z1=(50, 50),  # p(z1|z3,y)
                 dim_h_en_z2Fz1=(50),   # p(z2|z1,c,m); we'll train q(z2|x2,s) ~ p(z2|z1,c,m)
                 dim_h_en_z3=(50, 50),  # q(z3|z1,y)
                 dim_h_de_x=(50, 50),   # p(x1|z1,s) also used as p(x2|z2,s)
                 dim_h_clf=(50, 50),    # q(y|z1,z2) [or q(y|z2)]
                 dim_z1=50,
                 dim_z3=50,
                 type_rec='binary',
                 clf_z1z2=True, type_y='discrete', prior_y='uniform', clf_1sig=False,
                 epochs=500, batch_size=100, nonlinearity='softplus',
                 learning_rate=0.001, optim_alg='adam', L=1, weight_decay=None,
                 dropout_rate=0., input_x_dropout=0., add_noise_var=0.,
                 yloss_rate=1., anneal_yloss_offset=0,
                 use_MMD=True, kernel_MMD='rbf_fourier', mmd_rate=1.,
                 kl_qz2pz2_rate=1., pertloss_rate=0.1, anneal_perturb_rate_itermax=1, anneal_perturb_rate_offset=0,
                 use_s=False, use_c=False, use_m=False,
                 random_seed=12345, log_txt=None):
        super(DrVAE, self).__init__()
        ## set parameters as instance variables
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop('self')
        for arg, val in values.items():
            setattr(self, arg, val)

        ## Model architecture
        self.dim_z2 = dim_z1 # shared q(z|x,s) for z1 and z2; we'll train q(z2|x2,s) ~ p(z2|z1,c,m)
        ## Other hyperparameters
        self.wn = False # weight norm
        self.bn = False # batch norm
        self.prior_mu = 0.
        self.prior_sg = 1.
        if self.prior_y is not None and not isinstance(self.prior_y, str):
            # if prior_y is set (and not a string), then it has to be a np.array of length self.dim_y
            assert(isinstance(self.prior_y, np.ndarray) and len(self.prior_y) == self.dim_y)
        ## Fitting settings & options
        if self.weight_decay is None:
            self.weight_decay = 0.
        # loss annealing
        self.kl_min = 2. # Number of "free bits/nats per dim_z gaussian"
        self.kl_min_tt = Variable(torch.FloatTensor([self.kl_min]))
        self.anneal_learning_rate = False
        self.anneal_kl = False
        self.anneal_kl_itermax = 100
        self.anneal_yloss = False
        self.anneal_yloss_itermax = 1
        self.finished_training_iters = 0

        ## Log file
        self.w2log('DrVAE:')
        self.w2log(vars(self))

        self.nprng = np.random.RandomState(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        ## Build the model
        self._build_blocks()
        self._create_optimizer()

    def _build_blocks(self):
        """
        Instantiate encoder/decoder blocks of the model
        """
        common_hyper_params = {'nonlin': self.nonlinearity,
                               'weight_norm': self.wn,
                               'batch_norm': self.bn,
                               'dropout_rate': self.dropout_rate
                              }
        ## Create the decoder for p(x|z1,s)
        # type of x
        data_decoder_hyper_params = copy.deepcopy(common_hyper_params)
        if self.type_rec == 'binary':
            data_decoder = blk.BernoulliDecoder
        elif self.type_rec == 'diag_gaussian':
            data_decoder = blk.DiagGaussianSigmaModule
        elif self.type_rec == 'poisson':
            data_decoder = blk.PoissonDecoder
        else:
            raise ValueError()

        ## Create the encoder for z1 q(z1|x1,s) also used as q(z2|x2,s)
        if self.use_s:
            in_qz1 = [self.dim_x, self.dim_s]
            in_dropouts = [self.input_x_dropout, 0]
        else:
            in_qz1 = [self.dim_x]
            in_dropouts = [self.input_x_dropout]
        self.encoder_z1 = blk.DiagGaussianModule(in_qz1, self.dim_h_en_z1, self.dim_z1, input_dropout_rates=in_dropouts,
                                prior_mu=self.prior_mu, prior_sg=self.prior_sg, **common_hyper_params)

        ## Create the decoder for z2 from z1: p(z2|z1,c,m)
        in_pz2Fz1 = [self.dim_z1]
        if self.use_c:
            in_pz2Fz1.append(self.dim_c)
        if self.use_m:
            in_pz2Fz1.append(self.dim_m)
        # self.decoder_z2Fz1 = blk.DiagGaussianModule(in_pz2Fz1, self.dim_h_en_z2Fz1, self.dim_z2,
        #                         prior_mu=self.prior_mu, prior_sg=self.prior_sg, **common_hyper_params)
        self.decoder_z2Fz1 = blk.DiagGaussianModuleLinear(in_pz2Fz1, [], self.dim_z2, bias_only=False,
                                prior_mu=self.prior_mu, prior_sg=self.prior_sg, **common_hyper_params)

        ## Create the encoder for y q(y|z1,z2) [or q(y|z2)]
        if self.clf_z1z2:
            in_clf = [self.dim_z1, self.dim_z2]
        else:
            in_clf = [self.dim_z2]
        if self.type_y == 'discrete':
            if self.clf_1sig:
                if self.dim_y != 2: raise ValueError('Invalid combination of clf_1sig and dim_y')
                out_dim = 1
            else:
                out_dim = self.dim_y
            self.encoder_y = blk.CategoricalDecoder(in_clf, self.dim_h_clf, out_dim, **common_hyper_params)
        else:  # regression case
            self.encoder_y = blk.DiagGaussianModule(in_clf, self.dim_h_clf, self.dim_y,
                                    fixed_variance=0.05**2, constrain_means=True, **common_hyper_params) 

        ## Create the encoder for z3 q(z3|z1,y)
        self.encoder_z3 = blk.DiagGaussianModule([self.dim_z1, self.dim_y], self.dim_h_en_z3, self.dim_z3,
                                prior_mu=self.prior_mu, prior_sg=self.prior_sg, **common_hyper_params)

        ## Create the decoder for p(z1|z3,y)
        self.decoder_z1 = blk.DiagGaussianModule([self.dim_z3, self.dim_y], self.dim_h_de_z1, self.dim_z1,
                                **common_hyper_params)

        ## Create the decoder for p(x1|z1,s) also used as p(x2|z2,s)
        if self.use_s:
            in_px = [self.dim_z1, self.dim_s]
        else:
            in_px = [self.dim_z1]
        self.decoder_x = data_decoder(in_px, self.dim_h_de_x, self.dim_x, **data_decoder_hyper_params)

    def forward_w_pert_identity(self, x1, x2, s=[]):
        """
        Run inference in the model **assuming perturbation func is identity**
        - predict y
        - embedding qz1
        - prediction px2 (assuming perturbation is identity)
        - embedding qz2 (from x2)
        - reconstruction px2 (from qz2)
        """
        self.eval()

        ## posterior q(z1|x1, s)
        if self.use_s:
            s1inK = blk.one_hot(s, self.dim_s)
            in_qz1 = [x1, s1inK]
        else:
            in_qz1 = [x1]
        qz1 = self.encoder_z1(in_qz1)
        z1_mu = qz1[0]

        ## prediction: q(y|z1,z2) [or q(y|z2)] assuming perturbation is identity => p(z2|z1) == q(z1|x1)
        if self.clf_z1z2:
            # in_clf = [z1_mu, z1_mu]
            in_clf = [z1_mu, z1_mu - z1_mu]
        else:
            in_clf = [z1_mu]
        qypred = self.encoder_y(in_clf)
        if self.type_y == 'discrete':
            classifier_pred = self.encoder_y.most_probable(*qypred)
            proba = qypred[0]
        else:
            if len(qypred) > 1:
                classifier_pred, proba = qypred[0], qypred[1]
            else: # for Bernoulli decoder
                classifier_pred, proba = qypred[0], qypred[0]

        #### perturbation prediction assuming perturbation is identity => p(z2|z1) == q(z1|x1)
        ## p(x2|z2,s) where z2 ~ q(z1|x1)
        if self.use_s:
            in_px2 = [z1_mu, s1inK]
        else:
            in_px2 = [z1_mu]
        px2 = self.decoder_x(in_px2)
        x2_mu = px2[0]

        #### post-treatment x2 embedding and reconstruction by encoder_z1 and decoder_x, respectively
        ## posterior q(z2|x2,s)
        if self.use_s:
            s1inK = blk.one_hot(s, self.dim_s)
            in_qz2 = [x2, s1inK]
        else:
            in_qz2 = [x2]
        qz2 = self.encoder_z1(in_qz2)
        z2_mu = qz2[0]
        ## p(x2|z2,s) where z2 ~ q(z2|x2)
        if self.use_s:
            in_px2 = [z2_mu, s1inK]
        else:
            in_px2 = [z2_mu]
        px2_rec = self.decoder_x(in_px2)
        x2_rec_mu = px2_rec[0]

        return {'pred':classifier_pred, 'proba':proba, 'z1':z1_mu, 'qz1':qz1,
                'px2':px2, 'x2_pert':x2_mu,           # p(x2|z2,s) where z2 ~ q(z1|x1) (perturbation is identity)
                'z2':z2_mu, 'qz2':qz2,                # q(z2|x2,s)
                'px2_rec':px2_rec, 'x2_rec':x2_rec_mu # p(x2|z2,s) where z2 ~ q(z2|x2)
                } 

    def forward(self, x1, s=[]):
        """
        Run inference in the model to:
        - predict y
        - embedding qz1
        - embedding pz2 (from z1)
        - reconstruction px1
        - prediction px2
        """
        self.eval()

        ## posterior q(z1|x1, s)
        if self.use_s:
            s1inK = blk.one_hot(s, self.dim_s)
            in_qz1 = [x1, s1inK]
        else:
            in_qz1 = [x1]
        qz1 = self.encoder_z1(in_qz1)
        z1_mu = qz1[0]
        
        ## posterior p(z2|z1)
        pz2Fz1 = self.decoder_z2Fz1([z1_mu])
        z2Fz1_mu = pz2Fz1[0]

        ## prediction: q(y|z1,z2) [or q(y|z2)]
        if self.clf_z1z2:
            # in_clf = [z1_mu, z2Fz1_mu]
            in_clf = [z1_mu, z2Fz1_mu - z1_mu]
        else:
            in_clf = [z2Fz1_mu]
            # in_clf = [z2Fz1_mu - z1_mu]
        qypred = self.encoder_y(in_clf)
        if self.type_y == 'discrete':
            classifier_pred = self.encoder_y.most_probable(*qypred)
            proba = qypred[0]
        else:
            if len(qypred) > 1:
                classifier_pred, proba = qypred[0], qypred[1]
            else: # for Bernoulli decoder
                classifier_pred, proba = qypred[0], qypred[0]

        #### reconstructions
        ## p(x1|z1,s)
        if self.use_s:
            in_px1 = [z1_mu, s1inK]
        else:
            in_px1 = [z1_mu]
        px1 = self.decoder_x(in_px1)
        x1_mu = px1[0]
        ## p(x2|z2,s) where z2 ~ p(z2|z1)
        if self.use_s:
            in_px2 = [z2Fz1_mu, s1inK]
        else:
            in_px2 = [z2Fz1_mu]
        px2 = self.decoder_x(in_px2)
        x2_mu = px2[0]

        return {'pred':classifier_pred, 'proba':proba, 'z1':z1_mu, 'qz1':qz1, 'px1':px1, 'x1_rec':x1_mu,
                'z2':z2Fz1_mu, 'pz2':pz2Fz1, 'px2':px2, 'x2_pert':x2_mu}

    def predict(self, **kwargs):
        res = self.forward(**kwargs)
        pred = res['pred'].data.squeeze().numpy()
        proba = res['proba'].data.numpy()
        return pred, proba

    def reconstruct(self, **kwargs):
        res = self.forward(**kwargs)
        x1_rec = res['x1_rec'].data.numpy()
        px1 = [_x.data.numpy() for _x in res['px1']]
        x2_pert = res['x2_pert'].data.numpy()
        px2 = [_x.data.numpy() for _x in res['px2']]
        return x1_rec, px1, x2_pert, px2

    def transform(self, **kwargs):
        res = self.forward(**kwargs)
        z1 = res['z1'].data.numpy()
        z2 = res['z2'].data.numpy()
        return z1, z2

    def _fprop(self, z1, qz1, y):
        """
        Propagate through the generative model for a given class Y
        """
        KLD_perx = 0.
        z1_sample = z1[0]
        
        ## q(z3|z1,y)
        qz3 = self.encoder_z3([z1_sample, y])
        z3 = self.encoder_z3.sample(*qz3)
        z3_sample = z3[0] # for compatibility with IAF encoders return a tuple

        # KL-divergence q(z3|z1,y) || p(z3) ; p(z3)=N(0,I)
        try:
            KLD_perx += self._use_free_bits(self.encoder_z3.kldivergence_from_prior_perx(*qz3)) # add KL from prior
        except:
            # no KL-divergence, use logq(z) - logp(z) Monte Carlo estimation
            logq_perx = self.encoder_z3.logp_perx(*(z3 + qz3))
            logp_perx = self.encoder_z3.logp_prior_perx(z3_sample)
            KLD_perx += self._use_free_bits(logq_perx - logp_perx)

        ## p(z1|z3,y)
        pz1 = self.decoder_z1([z3_sample, y])
        # KL-divergence q(z1|x1,s) || p(z1|z3,y)
        try:
            KLD_perx += self._use_free_bits(self.encoder_z1.kldivergence_perx(*(qz1 + pz1)) )
        except:
            # no KL-divergence, use logq(z) - logp(z) Monte Carlo estimation
            logq_perx = self.encoder_z1.logp_perx(*(z1 + qz1)) 
            logp_perx = self.decoder_z1.logp_perx(z1_sample, *pz1)
            KLD_perx += self._use_free_bits(logq_perx - logp_perx)

        return KLD_perx

    def _compute_losses(self, x1, x2, s, y, L):
        """
        Compute all losses of the model. For unlabeled data marginalize y.
        RECL - reconstruction loss E_{q(z1|x1)}[ p(x1|z1) ]
        KLD  - kl-divergences of all the other matching q and p distributions
        PERT - perturbation prediction loss E_{p(z2|z1)q(z1|x1)}[ p(x2|z2) ]
        YL   - prediction loss on y (for labeled data)
        MMD  - maximum mean discrepancy of z1 embedding w.r.t. grouping s
        """
        RECL, KLD, PERT, YL, MMD = 0., 0., 0., 0., 0.
        N = x1.size(0)
        isPertPair = len(x2) > 0

        if self.type_y == 'discrete':
            y1inK = blk.one_hot(y, self.dim_y)
        else:
            if len(y) > 0 :
                y = y.float()
        # instantiate prior of y for discrete y
        if isinstance(self.prior_y, str) and self.prior_y == 'uniform':
            pr_y = Variable(torch.from_numpy(np.ones((N, self.dim_y)) / (1. * self.dim_y))).float()
        else:
            pr_y = Variable(torch.from_numpy(np.ones((N, self.dim_y)) * self.prior_y)).float()

        s1inK = None
        if self.use_s:
            s1inK = blk.one_hot(s, self.dim_s)
        if self.use_s and self.use_MMD:
            sind = [] # get the indices for the nuisance variable groups
            for si in range(self.dim_s):
                sind.append(torch.eq(s, si).squeeze())

        # get q(z1|x1, s) and q(z2|x2, s)
        if self.use_s:
            in_qz1 = [x1, s1inK]
        else:
            in_qz1 = [x1]
        if self.training and self.add_noise:
            eps = x1.data.new(x1.size()).normal_()
            eps = Variable(eps.mul_(self.add_noise_var))
            in_qz1[0] += eps
        qz1 = self.encoder_z1(in_qz1)
        if isPertPair:
            if self.use_s:
                in_qz2 = [x2, s1inK]
            else:
                in_qz2 = [x2]
            if self.training and self.add_noise:
                eps = x2.data.new(x2.size()).normal_()
                eps = Variable(eps.mul_(self.add_noise_var))
                in_qz2[0] += eps
            qz2 = self.encoder_z1(in_qz2) # !! use the same encoder as z1 !!

        Lf = 1. * L
        for _ in range(L):
            # sample from q(z1|x1, s)
            z1 = self.encoder_z1.sample(*qz1) 
            z1_sample = z1[0] # for compatibility with IAF encoders return a tuple
            # sample from q(z2|x2, s)
            if isPertPair:
                z2 = self.encoder_z1.sample(*qz1) # !! use the same encoder as z1 !!
                z2_sample = z2[0] # for compatibility with IAF encoders return a tuple

            # encode and sample from p(z2|z1) ## TODO: extend to p(z2|z1,c,m)
            pz2Fz1 = self.decoder_z2Fz1([z1_sample])
            z2Fz1 = self.decoder_z2Fz1.sample(*pz2Fz1)
            z2Fz1_sample = z2Fz1[0]

            ## get the reconstruction loss
            # p(x1|z1,s) where z1 ~ q(z1|x1,s)
            if self.use_s:
                in_px1 = [z1_sample, s]
            else:
                in_px1 = [z1_sample]
            px1 = self.decoder_x(in_px1)
            RECL += self.decoder_x.logp(x1, *px1) / Lf

            ## get reconstruction loss & perturbation prediction loss for x2
            if isPertPair:
                # p(x2|z2,s) where z2 ~ q(z2|x2,s)
                if self.use_s:
                    in_px2 = [z2_sample, s]
                else:
                    in_px2 = [z2_sample]
                px2 = self.decoder_x(in_px2)
                RECL += self.decoder_x.logp(x2, *px2) / Lf

                # p(x2|z2,s) where z2 ~ p(z2|z1)
                if self.use_s:
                    in_px2pert = [z2Fz1_sample, s]
                else:
                    in_px2pert = [z2Fz1_sample]
                px2pert = self.decoder_x(in_px2pert)
                PERT += self.decoder_x.logp(x2, *px2pert) / Lf

            ## match distributions over z2: KL( p(z2|z1) || q(z2|x2,s) )
            if isPertPair:
                try:
                    #### try analytic KL-divergence
                    KLz2Fz1_perx = self._use_free_bits(self.encoder_z1.kldivergence_perx(*(qz2 + pz2Fz1)) )  # !! use the same encoder as z1 !!
                    # print('{}\t{:.4f}'.format(self.finished_training_iters, KLz2Fz1_perx.sum().data.numpy()[0]))
                except:
                    #### no KL-divergence, use Monte Carlo estimation: ( logp(z2|z1) - logq(z2|x2,s) )
                    KLz2Fz1_perx = 0.
                    ## for MC use a sample from q(z2|x2,s) !!! works only if p(z2|z1) is not IAF !!!
                    try:
                        # no KL-divergence, use logq(z) - logp(z) Monte Carlo estimation
                        logq_perx = self.encoder_z1.logp_perx(*(z2 + qz2))  # !! use the same encoder as z1 !!
                        logp_perx = self.decoder_z2Fz1.logp_perx(*(z2[:1] + pz2Fz1)).clamp(min=-1e10) # log probability of a sample from q(z2|x2,s) in p(z2|z1)
                        KLz2Fz1_perx += self._use_free_bits(logq_perx - logp_perx)
                    except Exception as e:
                        print(e)
                ## apply free bits/nats
                # KLz2Fz1_perx = self._use_free_bits(KLz2Fz1_perx)
                ## sum KL
                beta_pert = 1.
                if self.anneal_perturb_rate_itermax > 0:
                    beta_pert = self._compute_anneal_coef(self.finished_training_iters,
                                                iter_max = self.anneal_perturb_rate_itermax,
                                                iter_offset = self.anneal_perturb_rate_offset)
                KLD += beta_pert * ( self.kl_qz2pz2_rate * torch.sum(KLz2Fz1_perx) / Lf )
                ## prevent qz1 and qz2 from collapsing : -( KL( q(z1|x1,s) || q(z2|x2,s) ) + KL( q(z2|x2,s) || q(z1|x1,s) ) )/2.
                # KLqz1qz2 = -(self.encoder_z1.kldivergence_perx(*(qz1 + qz2)) + self.encoder_z1.kldivergence_perx(*(qz2 + qz1))) / 2.
                # KLD += beta_pert * ( self.kl_qz2pz2_rate * 0.05 * torch.sum(KLz2Fz1_perx.clamp(max=500)) / Lf )

            ## prediction: q(y|z1,z2) [or q(y|z2)]
            if self.clf_z1z2:
                # in_clf = [z1_sample, z2Fz1_sample]
                in_clf = [z1_sample, z2Fz1_sample - z1_sample]
            else:
                in_clf = [z2Fz1_sample]
                # in_clf = [z2Fz1_sample - z1_sample]
            qy = self.encoder_y(in_clf)


            _KLD = 0.
            if len(y) > 0 :
                ## if y is given then
                # (i) compute prediction loss
                YL += self.encoder_y.logp(y, *qy) / Lf
                # (ii) condition on true y
                if self.type_y == 'discrete':
                    _y = y1inK
                else:
                    _y = y
                    if _y.ndimension() == 1:
                        _y.data.unsqueeze_(1)
                # the logprior of y is omitted as it is constant wrt the optimization
                _KLD = self._fprop(z1, qz1, _y)
            else:
                ## otherwise use predicted qy to marginalize out y
                if self.type_y == 'discrete':
                    # sum out y
                    for _j in range(self.dim_y):
                        _y_j = blk.one_hot(Variable(x1.data.new(N).float().fill_(_j)), self.dim_y)
                        _KLD_j = self._fprop(z1, qz1, _y_j)
                        assert qy[0][:, _j].size() == _KLD_j.size()
                        _KLD += qy[0][:, _j] * _KLD_j
                    # add logprior of y
                    _KLD += self.encoder_y.kldivergence_perx(qy[0], pr_y).sum(1)
                else:
                    # if continous then just use SGVB and sample y
                    _y = self.encoder_y.sample(*qy)[0]
                    _KLD = self._fprop(z1, qz1, _y)
                    # add logprior of y
                    if self.prior_y != 'uniform':
                        _KLD += self.encoder_y.kldivergence_perx(*(qy + self.prior_y)).sum(1)
            KLD += torch.sum(_KLD) / Lf

            ## maximum mean discrepancy regularization
            if self.use_s and self.use_MMD:
                MMD += self._get_mmd_criterion(z1_sample, sind) / Lf
                if isPertPair:
                    MMD += self._get_mmd_criterion(z2_sample, sind) / Lf
        
        ## loss per batch
        return OrderedDict([('RECL',RECL), ('KLD',KLD), ('PERT',PERT), ('YL',YL), ('MMD',MMD)])

    def loss_function(self, x1, x2, s, y, has_x2, has_y):
        """
        Compile total loss for the data
        """
        beta_kl = 1.
        if self.anneal_kl:
            beta_kl = self._compute_anneal_coef(self.finished_training_iters,
                                                iter_max = self.anneal_kl_itermax)
        beta_yr = 1.
        if self.anneal_yloss:
            beta_yr = self._compute_anneal_coef(self.finished_training_iters,
                                                iter_max = self.anneal_yloss_itermax,
                                                iter_offset = self.anneal_yloss_offset)
        beta_pert = 1.
        if self.anneal_perturb_rate_itermax > 0:
            beta_pert = self._compute_anneal_coef(self.finished_training_iters,
                                                iter_max = self.anneal_perturb_rate_itermax,
                                                iter_offset = self.anneal_perturb_rate_offset)
        # print("annealing KLD, YR: ", beta_kl, beta_yr, end='; ')

        hasnt_y = (has_y + 1) % 2   # negate has_y binary mask
        hasnt_x2 = (has_x2 + 1) % 2 # negate has_x2 binary mask
        # indices of Labeled Singleton data
        idx_ls = torch.nonzero((has_y + hasnt_x2 == 2).data).view(-1) 
        Nls = len(idx_ls)
        # indices of Unlabeled Singleton data
        idx_us = torch.nonzero((hasnt_y + hasnt_x2 == 2).data).view(-1)
        Nus = len(idx_us)
        # indices of Labeled Paired perturbation data
        idx_lp = torch.nonzero((has_y + has_x2 == 2).data).view(-1)
        Nlp = len(idx_lp)
        # indices of Unlabeled Paired perturbation data
        idx_up = torch.nonzero((hasnt_y + has_x2 == 2).data).view(-1)
        Nup = len(idx_up)
        assert Nls + Nus + Nlp + Nup == x1.size(0)

        #### compute model loss
        zero = Variable(torch.zeros(1))
        dummy_loss = OrderedDict([('RECL',zero), ('KLD',zero), ('PERT',zero), ('YL',zero), ('MMD',zero)])
        # Labeled Singleton data
        if Nls != 0:
            losses_ls = self._compute_losses(x1=x1[idx_ls], x2=[], s=s[idx_ls], y=y[idx_ls], L=self.L)
        else:
            warnings.warn("No Labeled Singleton data in the minibatch")
            losses_ls = dummy_loss
        # Unlabeled Singleton data
        if Nus != 0:
            losses_us = self._compute_losses(x1=x1[idx_us], x2=[], s=s[idx_us], y=[], L=self.L)
        else:
            warnings.warn("No Unlabeled Singleton data in the minibatch")
            losses_us = dummy_loss

        # Labeled Paired perturbation data
        if Nlp != 0:
            losses_lp = self._compute_losses(x1=x1[idx_lp], x2=x2[idx_lp], s=s[idx_lp], y=y[idx_lp], L=self.L)
        else:
            warnings.warn("No Labeled Paired perturbation data in the minibatch")
            losses_lp = dummy_loss
        # Unlabeled Paired perturbation data
        if Nup != 0:
            losses_up = self._compute_losses(x1=x1[idx_up], x2=x2[idx_up], s=s[idx_up], y=[], L=self.L)
        else:
            warnings.warn("No Unlabeled Paired perturbation data in the minibatch")
            losses_up = dummy_loss

        #### sum and normalize the losses per example
        losses = OrderedDict()
        losses['RECL'] = (losses_ls['RECL'] + losses_us['RECL'] + losses_lp['RECL'] + losses_up['RECL']) / (Nls + Nus + Nlp + Nup)
        losses['KLD']  = (losses_ls['KLD'] + losses_us['KLD'] + losses_lp['KLD'] + losses_up['KLD']) / (Nls + Nus + Nlp + Nup)
        losses['PERT'] = (losses_lp['PERT'] + losses_up['PERT']) / max(1., (Nlp + Nup))
        losses['YL']   = (losses_ls['YL'] + losses_lp['YL']) / max(1., (Nls + Nlp))
        losses['MMD']  = (losses_ls['MMD'] + losses_us['MMD'] + losses_lp['MMD'] + losses_up['MMD']) / (Nls + Nus + Nlp + Nup)

        ## ELBO
        losses['ELBO'] = losses['RECL'] + (beta_pert * self.pertloss_rate  * losses['PERT']) - (beta_kl * losses['KLD'])

        ## complete compound loss
        losses['CMPL'] = -losses['ELBO'] -(beta_yr * self.yloss_rate * losses['YL'])
        if self.use_MMD:
            losses['CMPL'] += -(self.mmd_rate * losses['MMD'])

        return losses

    def evaluate_performance_on_dataset(self, ds, return_full_data=False):
        """
        Evaluate the model performance on Dataset
        """
        x1 = Variable(ds.x1, volatile=True)
        x2 = Variable(ds.x2, volatile=True)
        s = Variable(ds.s, volatile=True)
        y = Variable(ds.y, volatile=True)
        has_x2 = Variable(ds.has_x2, volatile=True)
        has_y = Variable(ds.has_y, volatile=True)
        return self.evaluate_performance(x1, x2, s, y, has_x2, has_y, return_full_data)

    def evaluate_performance(self, x1, x2, s, y, has_x2, has_y, return_full_data=False):
        """
        Evaluate the model performance
        """
        perf = OrderedDict()

        ## eval losses
        try:
            losses = self.run_on_batch(train_mode=False, x1=x1, x2=x2, s=s, y=y, has_x2=has_x2, has_y=has_y)
            perf['losses'] = losses
        except Exception as e:
            print("Warning, computation of losses failed in evaluation!")
            print(e)
            perf['losses'] = None

        ## run inference in the model on all data
        res = self.forward(x1, s)
        
        #### eval prediction performance on "y" for labeled data
        yidx = torch.nonzero(has_y.data).view(-1)
        ylab = y[yidx]
        pred = res['pred'][yidx]
        proba = res['proba'][yidx]
        res_pd = self.eval_y_prediction(pred, proba, ylab)
        perf = utl.concat_dicts(perf, dict([('y_' + k, v) for k, v in res_pd.items()]))
        if self.type_y == 'discrete':
            # y_str = 'F1: {:.3f} AUROC: {:.3f} AUPR: {:.3f}'.format(perf['y_f1'], perf['y_auroc'], perf['y_aupr'])
            y_str = 'Accuracy: {:.3f}% AUROC: {:.3f} AUPR: {:.3f}'.format(perf['y_acc'] * 100., perf['y_auroc'],
                                                                          perf['y_aupr'])
        else:
            y_str = 'RMSE: {:.3f} R2: {:.3f} Pearson: {:.3f}'.format(perf['y_rmse'], perf['y_r2'], perf['y_pearr'])
            # print("mean prediction:{:.4f}\tprediction std:{:.4f}\tmean std:{:.4f}\tmean ytrue:{:.4f}\tytrue std:{:.4f}".format(
            #         pred_np.mean(), pred_np.std(), np.exp(proba_np*0.5).mean(), ylab_np.mean(), ylab_np.std()))
        
        #### eval reconstruction of "x1" for all data
        assert torch.equal(res['x1_rec'], res['px1'][0])
        res_pd = self.eval_x_reconstruction(x1, *res['px1'])
        perf = utl.concat_dicts(perf, dict([('x1_' + k, v) for k, v in res_pd.items()]))
        rec_str = 'RMSE: {:.3f} R2: {:.3f} Pearson: {:.3f}'.format(perf['x1_rmse'], perf['x1_r2'], perf['x1_pearr'])

        #### eval perturbation prediction of "x2" on paired data
        x2idx = torch.nonzero(has_x2.data).view(-1)
        if len(x2idx) > 0:
            assert torch.equal(res['x2_pert'][x2idx], res['px2'][0][x2idx])
            res_pd = self.eval_x_reconstruction(x2[x2idx], res['px2'][0][x2idx], res['px2'][1][x2idx])
            perf = utl.concat_dicts(perf, dict([('x2_' + k, v) for k, v in res_pd.items()]))
            pert_str = 'RMSE: {:.3f} R2: {:.3f} Pearson: {:.3f}'.format(perf['x2_rmse'], perf['x2_r2'], perf['x2_pearr'])
        else:
            perf = utl.concat_dicts(perf, dict([('x2_' + k, np.nan) for k in ('rmse', 'r2', 'pearr', 'll')]))
            pert_str = 'no x2 data'
        
        ## return also full data
        if return_full_data:
            perf['z1'] = res['z1'].data.numpy()
            perf['z2'] = res['z2'].data.numpy()
            perf['x2_pert'] = res['x2_pert'].data.numpy()
            perf['pred'] = res['pred'].data.numpy()
            perf['proba'] = res['proba'].data.numpy()

            ## run inference in the model with **perturbation function qual to identity**
            res2 = self.forward_w_pert_identity(x1, x2, s)

            # prediction of y
            pred = res2['pred'][yidx]
            proba = res2['proba'][yidx]
            res_pd = self.eval_y_prediction(pred, proba, ylab)
            perf = utl.concat_dicts(perf, dict([('y_wI_' + k, v) for k, v in res_pd.items()]))
            # subset to paired data that have "x2" to compute perturbation prediction loss
            x2idx = torch.nonzero(has_x2.data).view(-1)
            if len(x2idx) > 0:
                # perturbation prediction (of x2) with Identity as pert. function
                assert torch.equal(res2['x2_pert'][x2idx], res2['px2'][0][x2idx])
                res_pd = self.eval_x_reconstruction(x2[x2idx], res2['px2'][0][x2idx], res2['px2'][1][x2idx])
                perf = utl.concat_dicts(perf, dict([('x2_wI_' + k, v) for k, v in res_pd.items()]))
                # reconstruction of x2 from q(z2|x2)
                assert torch.equal(res2['x2_rec'][x2idx], res2['px2_rec'][0][x2idx])
                res_pd = self.eval_x_reconstruction(x2[x2idx], res2['px2_rec'][0][x2idx], res2['px2_rec'][1][x2idx])
                perf = utl.concat_dicts(perf, dict([('x2_rec_' + k, v) for k, v in res_pd.items()]))

                ## compute correlations between qz1_mu and qz2_mu, and between z2Fz1_mu and qz2_mu
                qz1mu_np = res2['z1'][x2idx].data.numpy().astype(float)
                qz2mu_np = res2['z2'][x2idx].data.numpy().astype(float)
                pz2Fz1_mu_np = res['z2'][x2idx].data.numpy().astype(float)
                perf['qz1mu_qz2mu_rmse'] = np.sqrt(((qz1mu_np - qz2mu_np) ** 2).mean())
                perf['pz2Fz1mu_qz2mu_rmse'] = np.sqrt(((pz2Fz1_mu_np - qz2mu_np) ** 2).mean())

                qz1 = [res2['qz1'][0][x2idx], res2['qz1'][1][x2idx]]
                qz2 = [res2['qz2'][0][x2idx], res2['qz2'][1][x2idx]]
                pz2Fz1 = [res['pz2'][0][x2idx], res['pz2'][1][x2idx]]
                perf['KL_qz2_qz1'] = self.encoder_z1.kldivergence_perx(*(qz2 + qz1)).mean().data.numpy().astype(float)[0]
                perf['KL_qz2_pz2Fz1'] = self.encoder_z1.kldivergence_perx(*(qz2 + pz2Fz1)).mean().data.numpy().astype(float)[0]
            else:
                perf = utl.concat_dicts(perf, dict([('x2_wI_' + k, np.nan) for k in ('rmse', 'r2', 'pearr', 'll')]))
                perf = utl.concat_dicts(perf, dict([('x2_rec_' + k, np.nan) for k in ('rmse', 'r2', 'pearr', 'll')]))
                perf['qz1mu_qz2mu_rmse'] = np.nan
                perf['pz2Fz1mu_qz2mu_rmse'] = np.nan
                perf['KL_qz2_qz1'] = np.nan
                perf['KL_qz2_pz2Fz1'] = np.nan

        perf['model_class'] = self.__class__.__name__
        combined_str = 'Y: {}\t X1: {}\t X2: {}'.format(y_str, rec_str, pert_str)
        return perf, combined_str

    def fit(self, train_loader, valid_loader, add_noise=False,
            verbose=False, early_stop=False, model_filename='best_model.pth'):
        """
        Main method to train the model

        @train_loader: training data as torch.utils.data.DataLoader
        @valid_loader: validation data as torch.utils.data.DataLoader
        @verbose: bool, print more stats
        @early_stop: bool
        @model_filename: string, file name where to snapshot the model parameters
        """
        np.set_printoptions(precision=4)
        ## early-stopping parameters
        patience = 50              # at least this many epochs
        patience_increase = 15     # wait this much longer when a new best is found
        improvement_threshold = 0.999
        best_valid_obj = -np.inf
        rolling_valid_obj = np.array([], dtype=float)
        memory_length = 3          # average over this many validation rounds
        useRollingMean = True      # use rolling average
        epochs_after_improvement = 0 # count how many epochs have it been since the best yet validation performance
        save_model_after = 0       # snapshot the model if there has not been an improvement after this many epoch
        snapshotted = False

        self.w2log('Starting training at: {}'.format(time.strftime("%c")))
        try:
            self.add_noise = add_noise
            for epoch in range(1, self.epochs + 1):
                #### training -------------------------------------------------
                t = time.time()
                train_loss = 0
                for batch_idx, (data_x1, data_x2, data_s, data_y, data_has_x2, data_has_y) in enumerate(train_loader):
                    data_x1 = Variable(data_x1)
                    data_x2 = Variable(data_x2)
                    data_s = Variable(data_s)
                    data_y = Variable(data_y)
                    data_has_x2 = Variable(data_has_x2)
                    data_has_y = Variable(data_has_y)
                    loss = self.run_on_batch(train_mode=True, x1=data_x1, x2=data_x2, s=data_s, y=data_y, has_x2=data_has_x2, has_y=data_has_y)
                    train_loss += self.yloss_rate * loss['YL'].data[0] + loss['RECL'].data[0]

                    if verbose and batch_idx % max(10, len(train_loader)/10) == 0:
                        # test prediction performance on labeled examples in the minibatch
                        train_perf, perf_str = self.evaluate_performance(x1=data_x1, x2=data_x2, s=data_s, y=data_y, has_x2=data_has_x2, has_y=data_has_y)

                        # str_losses = '\t'.join([(k+': {:.2f}').format(loss[k] if isinstance(loss[k], float) else loss[k].data[0]) for k in loss.keys()])
                        str_losses = '\t'.join([(k+': {:.3f}').format(loss[k].data[0]) for k in ['CMPL', 'ELBO', 'RECL', 'PERT', 'YL']])
                        self.w2log('training epoch: {} [{}/{} ({:.0f}%)]\t{}'.format(
                            epoch, batch_idx * len(data_x1), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            str_losses)) #, perf_str))
                train_loss /= len(train_loader)

                ## run prediction on the whole train set
                train_perf, train_str = self.evaluate_performance_on_dataset(train_loader.dataset)
                dt = time.time() - t
                self.w2log('====> Epoch: {}\tIter: {}'.format(epoch, self.finished_training_iters))
                self.w2log('Train: sec/epoch: {:.2f}\tAvg train loss: {:9.4f}\t{}'.format(dt, train_loss, train_str))

                #### validation -----------------------------------------------
                t = time.time()
                valid_loss = 0
                # for i, (data_x1, data_x2, data_s, data_y, data_has_x2, data_has_y) in enumerate(valid_loader):
                #     data_x1 = Variable(data_x1, volatile=True)
                #     data_x2 = Variable(data_x2, volatile=True)
                #     data_s = Variable(data_s, volatile=True)
                #     data_y = Variable(data_y, volatile=True)
                #     data_has_x2 = Variable(data_has_x2, volatile=True)
                #     data_has_y = Variable(data_has_y, volatile=True)
                #     loss = self.run_on_batch(train_mode=False, x1=data_x1, x2=data_x2, s=data_s, y=data_y, has_x2=data_has_x2, has_y=data_has_y)
                #     valid_loss += self.yloss_rate * loss['YL'].data[0] + loss['RECL'].data[0]
                # valid_loss /= len(valid_loader)
                ## print weights
                # print(self.decoder_z2Fz1.state_dict()["W_mu"].numpy())
                # print(self.decoder_z2Fz1.state_dict()["bias_mu"].numpy())
                # print(self.encoder_z1.state_dict()["nnet.model.linear1.weight"])

                ## run prediction on the whole validation set 
                valid_perf, perf_str = self.evaluate_performance_on_dataset(valid_loader.dataset)
                if self.type_y == 'discrete':
                    valid_loss = valid_perf['y_auroc'] + valid_perf['y_aupr'] + valid_perf['x1_pearr'] + valid_perf['x2_pearr']
                else:
                    valid_loss = valid_perf['y_r2'] + valid_perf['y_pearr'] + valid_perf['x1_pearr'] + valid_perf['x2_pearr']
                # valid_loss = self.yloss_rate * valid_perf['losses']['YL'].data[0] + valid_perf['losses']['RECL'].data[0]
                dt = time.time() - t
                self.w2log('Valid: sec/epoch: {:.2f}\tValid set loss: {:9.4f}\t{}'.format(dt, valid_loss, perf_str))

                #### early stopping -------------------------------------------
                epochs_after_improvement += 1
                this_valid_obj = valid_loss #TODO!!!! use just the classification loss

                if useRollingMean:
                    rolling_valid_obj = np.append(rolling_valid_obj, this_valid_obj)
                    rolling_valid_obj = rolling_valid_obj[max(0, len(rolling_valid_obj)-memory_length):]
                    self.w2log('Valid rolling mem: {}\tmean: {:.4f}\tbest: {:.4f}'.format(
                        rolling_valid_obj, rolling_valid_obj.mean(), best_valid_obj))
                    evaluate_valid_obj = rolling_valid_obj.mean()
                else:
                    evaluate_valid_obj = this_valid_obj

                # improve patience if objective improvement is good enough
                if evaluate_valid_obj * improvement_threshold > best_valid_obj:
                    patience = max(patience, epoch + patience_increase)
                    best_valid_obj = evaluate_valid_obj
                    epochs_after_improvement = 0

                # save a snapshot of the model
                if (early_stop and epochs_after_improvement == save_model_after) or (patience <= epoch and not snapshotted):
                    snapshotted = True
                    self.save_to_file(model_filename)
                    self.w2log('* Snapshotting at epoch {}'.format(epoch))

                if patience <= epoch:
                    self.w2log('Early stopping at: {} with train: {:.4f} valid: {:.4f} evaluate_valid_obj: {:.4f} best_valid_obj: {:.4f}'.format(
                                epoch, train_loss, valid_loss, evaluate_valid_obj, best_valid_obj))
                    if early_stop:
                        if not snapshotted:
                            snapshotted = True
                            self.save_to_file(model_filename)
                            self.w2log('* Snapshotting at epoch {}'.format(epoch))
                        break
                    else:
                        self.w2log('Continuing')
                        patience = self.epochs + 1
                        snapshotted = False
        except KeyboardInterrupt:
            self.w2log('KeyboardInterrupt')
            if not snapshotted:
                snapshotted = True
                self.save_to_file(model_filename)
                self.w2log('* Snapshotting at epoch {}'.format(epoch))
            pass

        self.w2log('Finished training at: {}'.format(time.strftime("%c")))
        return


class DrVAEDataset(torch.utils.data.Dataset):
    """
    pytorch wrapper for semi-supervised DrVAE dataset containing
    "x1", "x2", "s", "y", "has_x2", "has_y"
    """

    def __init__(self, x1, x2, s, y, has_x2, has_y):
        assert x1.size(0) == x2.size(0)
        assert x1.size(1) == x2.size(1)
        assert x1.size(0) == y.size(0)
        assert x1.size(0) == s.size(0)
        assert x1.size(0) == has_x2.size(0)
        assert x1.size(0) == has_y.size(0)
        self.x1 = x1
        self.x2 = x2
        self.s = s
        self.y = y
        self.has_x2 = has_x2
        self.has_y = has_y

    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.s[index], self.y[index], self.has_x2[index], self.has_y[index]

    def __len__(self):
        return self.x1.size(0)


def wrap_in_DrVAEDataset(sing, pair, y_key='y', concat='both', downlabel_to=None, remove_unlabeled=False):
    """
    Wrap given data in pytorch DrVAEDataset
    @sing - dict with singleton data
    @pair - dict with paired perturbation data
    @y_key - selects target variable key,
            i.e. 'y' for discrete or 'ycont' for continuous labels
    @concat - if 'both': concatenate @sing to @pair on x1 and impute fake x2
              if 'pair_only', ignore @sing
              if 'sing_only', ignore @pair
    return: DrVAEDataset instance, dictionary of the data in numpy arrays
    """
    if concat == 'both':
        ## concatenate the @sing and @pair on common keys
        common_keys = set(sing.keys()) & set(pair.keys())
        ddict = dict([(k, np.concatenate((sing[k], pair[k]))) for k in common_keys])
        ## fake x2 for singletons
        ddict['x2'] = np.concatenate((np.zeros(sing['x1'].shape), pair['x2']))
        ddict['has_x2'] = np.concatenate((np.zeros(sing['x1'].shape[0]),
                                          np.ones(pair['x2'].shape[0])))
    elif concat == 'pair_only':
        ddict = pair
        ddict['has_x2'] = np.ones(pair['x2'].shape[0])
    elif concat == 'sing_only':
        ddict = sing
        ## fake x2 for singletons
        ddict['x2'] = np.zeros(sing['x1'].shape)
        ddict['has_x2'] = np.zeros(sing['x1'].shape[0])
    else:
        raise ValueError("Invalid parameter for dataset concatenation type")

    if downlabel_to is not None:
        ## select random @downlabel_to labeled data and mark the rest as unlabeled
        selectcids = np.unique(ddict['cid'][ddict['has_y']])
        np.random.shuffle(selectcids)
        if len(selectcids) > downlabel_to:
            selectcids = selectcids[:downlabel_to]
        for i in range(ddict['has_y'].shape[0]):
            if ddict['cid'][i] not in selectcids:
                ddict['has_y'][i] = 0
                ddict['y'][i] = -66
                ddict['ycont'][i] = -66
    if remove_unlabeled:
        ## remove unlabeled singleton data but keep unlabeled perturbations
        # keep = np.logical_or(ddict['has_y'], ddict['has_x2'])
        keep = ddict['has_y'] != 0
        for k in ddict.keys():
            ddict[k] = ddict[k][keep]

    dataset = DrVAEDataset(x1=torch.from_numpy(ddict['x1']).float(),
                           x2=torch.from_numpy(ddict['x2']).float(),
                           s=torch.from_numpy(ddict['s'].astype(np.int32)),
                           y=torch.from_numpy(ddict[y_key]),
                           has_x2=torch.from_numpy(ddict['has_x2'].astype(np.int32)),
                           has_y=torch.from_numpy(ddict['has_y'].astype(np.int32))
                           )
    return dataset, ddict
