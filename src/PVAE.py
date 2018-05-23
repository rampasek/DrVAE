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


class PVAE(DeepGenerativeModelMixin, nn.Module):

    """
    Drug Perturbation Variational Autoencoder (PertVAE)

    Implements a generative model with 1 sequence step in latent space:
        p(x1, x2, z1, z2 | s) = p(z1) * p(z2|z1) * p(x1|z1,s) * p(x2|z2,s)
    additionally q(z2|x2,s) and q(z2|x2,s) are a shared variational posterior,
    if x2 is not observed then q(z2|x1) = p(z2|z1) * q(z1|x1,s)

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
                 dim_h_en_z2Fz1=(50),   # p(z2|z1,c,m); we'll train q(z2|x2,s) ~ p(z2|z1,c,m)
                 dim_h_de_x=(50, 50),   # p(x1|z1,s) also used as p(x2|z2,s)
                 dim_z1=50,
                 type_rec='binary',
                 epochs=500, batch_size=100, nonlinearity='softplus',
                 learning_rate=0.001, optim_alg='adam', L=1, weight_decay=None,
                 dropout_rate=0., input_x_dropout=0., add_noise_var=0.,
                 use_MMD=True, kernel_MMD='rbf_fourier', mmd_rate=1.,
                 kl_qz2pz2_rate=1., pertloss_rate=0.1, anneal_perturb_rate_itermax=1, anneal_perturb_rate_offset=0,
                 use_s=False, use_c=False, use_m=False,
                 random_seed=12345, log_txt=None):
        super(PVAE, self).__init__()
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
        # self.anneal_perturb_rate_itermax = anneal_perturb_rate_itermax
        # self.anneal_perturb_rate_offset = anneal_perturb_rate_offset
        self.finished_training_iters = 0

        ## Log file
        self.w2log('PVAE:')
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

        ## Create the decoder for p(x1|z1,s) also used as p(x2|z2,s)
        if self.use_s:
            in_px = [self.dim_z1, self.dim_s]
        else:
            in_px = [self.dim_z1]
        self.decoder_x = data_decoder(in_px, self.dim_h_de_x, self.dim_x, **data_decoder_hyper_params)

    def forward_w_pert_identity(self, x1, x2, s=[]):
        """
        Run inference in the model **assuming perturbation func is identity**
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

        return {'z1':z1_mu, 'qz1':qz1,
                'px2':px2, 'x2_pert':x2_mu,           # p(x2|z2,s) where z2 ~ q(z1|x1) (perturbation is identity)
                'z2':z2_mu, 'qz2':qz2,                # q(z2|x2,s)
                'px2_rec':px2_rec, 'x2_rec':x2_rec_mu # p(x2|z2,s) where z2 ~ q(z2|x2)
                }

    def forward(self, x1, s=[]):
        """
        Run inference in the model to:
        - predict y
        - embedding qz1
        - embedding qz2
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

        #### reconstructions
        ## p(x1|z1,s)
        if self.use_s:
            in_px1 = [z1_mu, s1inK]
        else:
            in_px1 = [z1_mu]
        px1 = self.decoder_x(in_px1)
        x1_mu = px1[0]
        ## p(x2|z2,s)
        if self.use_s:
            in_px2 = [z2Fz1_mu, s1inK]
        else:
            in_px2 = [z2Fz1_mu]
        px2 = self.decoder_x(in_px2)
        x2_mu = px2[0]

        return {'z1':z1_mu, 'qz1':qz1, 'px1':px1, 'x1_rec':x1_mu, 'z2':z2Fz1_mu, 'pz2':pz2Fz1, 'px2':px2, 'x2_pert':x2_mu}

    def predict(self, **kwargs):
        raise NotImplementedError("This is not a classification model")

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

    def _compute_losses(self, x1, x2, s, L):
        """
        Compute all losses of the model. For unlabeled data marginalize y.
        RECL - reconstruction loss E_{q(z1|x1)}[ p(x1|z1) ]
        KLD  - kl-divergences of all the other matching q and p distributions
        PERT - perturbation prediction loss E_{p(z2|z1)q(z1|x1)}[ p(x2|z2) ]
        MMD  - maximum mean discrepancy of z1 embedding w.r.t. grouping s
        """
        RECL, KLD, PERT, MMD = 0., 0., 0., 0.
        N = x1.size(0)
        isPertPair = len(x2) > 0

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

            ## KL-divergence q(z1|x1) || p(z1); p(z1)=N(0,I)
            KLD_perx = 0.
            try:
                KLD_perx += self._use_free_bits(self.encoder_z1.kldivergence_from_prior_perx(*qz1)) # add KL from prior
            except:
                # no KL-divergence, use logq(z) - logp(z) Monte Carlo estimation
                logq_perx = self.encoder_z1.logp_perx(*(z1 + qz1))
                logp_perx = self.encoder_z1.logp_prior_perx(z1_sample)
                KLD_perx += self._use_free_bits(logq_perx - logp_perx)
            KLD += torch.sum(KLD_perx) / Lf

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
                # PERT = Variable(torch.FloatTensor(1).zero_())
                try:
                    if self.use_s:
                        in_px2pert = [z2Fz1_sample, s]
                    else:
                        in_px2pert = [z2Fz1_sample]
                    px2pert = self.decoder_x(in_px2pert)
                    PERT += self.decoder_x.logp(x2, *px2pert) / Lf
                except Exception as e:
                    print(e)

                ## KL-divergence q(z2|x2) || p(z2); p(z2)=N(0,I)
                KLD_perx = 0.
                try: # add KL from prior
                    KLD_perx += self._use_free_bits(self.encoder_z1.kldivergence_from_prior_perx(*qz2)) # !! use the same encoder as z1 !!
                except:
                    # no KL-divergence, use logq(z) - logp(z) Monte Carlo estimation
                    logq_perx = self.encoder_z1.logp_perx(*(z2 + qz2))
                    logp_perx = self.encoder_z1.logp_prior_perx(z2_sample)
                    KLD_perx += self._use_free_bits(logq_perx - logp_perx)
                KLD += torch.sum(KLD_perx) / Lf

            ## match distributions over z2: KL( q(z2|x2,s) || p(z2|z1) )
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
                        logp_perx = self.decoder_z2Fz1.logp_perx(*(z2[:1] + pz2Fz1)).clamp(min=-10e10) # log probability of a sample from q(z2|x2,s) in p(z2|z1)
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
                # KLD += torch.norm(self.decoder_z2Fz1.W_mu) / Lf

            ## maximum mean discrepancy regularization
            if self.use_s and self.use_MMD:
                MMD += self._get_mmd_criterion(z1_sample, sind) / Lf
                if isPertPair:
                    MMD += self._get_mmd_criterion(z2_sample, sind) / Lf
        
        ## loss per batch
        return OrderedDict([('RECL',RECL), ('KLD',KLD), ('PERT',PERT), ('MMD',MMD)])

    def loss_function(self, x1, x2, s, has_x2):
        """
        Compile total loss for the data
        """
        beta_kl = 1.
        if self.anneal_kl:
            beta_kl = self._compute_anneal_coef(self.finished_training_iters,
                                                iter_max = self.anneal_kl_itermax)
        beta_pert = 1.
        if self.anneal_perturb_rate_itermax > 0:
            beta_pert = self._compute_anneal_coef(self.finished_training_iters,
                                                iter_max = self.anneal_perturb_rate_itermax,
                                                iter_offset = self.anneal_perturb_rate_offset)
        # print("annealing KLD, YR: ", beta_kl, beta_yr, end='; ')

        hasnt_x2 = (has_x2 + 1) % 2 # negate has_x2 binary mask
        # indices of Singleton data
        idx_s = torch.nonzero(hasnt_x2.data).view(-1) 
        Ns = len(idx_s)
        # indices of Paired perturbation data
        idx_p = torch.nonzero(has_x2.data).view(-1)
        Np = len(idx_p)
        assert Ns + Np == x1.size(0)

        #### compute model loss
        zero = Variable(torch.zeros(1))
        dummy_loss = OrderedDict([('RECL',zero), ('KLD',zero), ('PERT',zero), ('MMD',zero)])
        # Singleton data
        if Ns != 0:
            losses_s = self._compute_losses(x1=x1[idx_s], x2=[], s=s[idx_s], L=self.L)
        else:
            warnings.warn("No Labeled Singleton data in the minibatch")
            losses_s = dummy_loss

        # Paired perturbation data
        if Np != 0:
            losses_p = self._compute_losses(x1=x1[idx_p], x2=x2[idx_p], s=s[idx_p], L=self.L)
        else:
            warnings.warn("No Labeled Paired perturbation data in the minibatch")
            losses_p = dummy_loss

        #### sum and normalize the losses per example
        losses = OrderedDict()
        losses['RECL'] = (losses_s['RECL'] + losses_p['RECL']) / (Ns + Np)
        losses['KLD']  = (losses_s['KLD'] + losses_p['KLD']) / (Ns + Np)
        losses['PERT'] = (losses_p['PERT']) / max(1., Np)
        losses['MMD']  = (losses_s['MMD'] + losses_p['MMD']) / (Ns + Np)

        ## ELBO
        losses['ELBO'] = losses['RECL'] + (beta_pert * self.pertloss_rate * losses['PERT']) - (beta_kl * losses['KLD'])

        ## complete compound loss
        losses['CMPL'] = -losses['ELBO']
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
        has_x2 = Variable(ds.has_x2, volatile=True)
        return self.evaluate_performance(x1, x2, s, has_x2, return_full_data)

    def evaluate_performance(self, x1, x2, s, has_x2, return_full_data=False):
        """
        Evaluate the model performance
        """
        perf = OrderedDict()

        ## eval losses
        losses = self.run_on_batch(train_mode=False, x1=x1, x2=x2, s=s, has_x2=has_x2)
        perf['losses'] = losses

        ## run inference in the model on all data
        res = self.forward(x1, s)

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

            ## run inference in the model with **perturbation function qual to identity**
            res2 = self.forward_w_pert_identity(x1, x2, s)

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
        combined_str = 'X1: {}\t X2: {}'.format(rec_str, pert_str)
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
                    data_has_x2 = Variable(data_has_x2)
                    loss = self.run_on_batch(train_mode=True, x1=data_x1, x2=data_x2, s=data_s, has_x2=data_has_x2)
                    train_loss += loss['RECL'].data[0]

                    if verbose and batch_idx % max(10, len(train_loader)/10) == 0:
                        # test prediction performance on labeled examples in the minibatch
                        train_perf, perf_str = self.evaluate_performance(x1=data_x1, x2=data_x2, s=data_s, has_x2=data_has_x2)

                        # str_losses = '\t'.join([(k+': {:.2f}').format(loss[k] if isinstance(loss[k], float) else loss[k].data[0]) for k in loss.keys()])
                        str_losses = '\t'.join([(k+': {:.3f}').format(loss[k].data[0]) for k in ['CMPL', 'ELBO', 'RECL', 'PERT']])
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
                
                ## run prediction on the whole validation set 
                valid_perf, perf_str = self.evaluate_performance_on_dataset(valid_loader.dataset)
                valid_loss = valid_perf['x1_pearr'] + valid_perf['x2_pearr']
                # valid_loss = valid_perf['losses']['RECL'].data[0]
                dt = time.time() - t
                self.w2log('Valid: sec/epoch: {:.2f}\tValid set loss: {:9.4f}\t{}'.format(dt, valid_loss, perf_str))
                try: self.w2log("Pert W norm:{:9.4f}".format(torch.norm(self.decoder_z2Fz1.W_mu).data.numpy()[0]))
                except: pass

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

        