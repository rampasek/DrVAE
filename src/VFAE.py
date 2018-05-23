"""
Ladislav Rampasek (rampasek@gmail.com)
Acknowledgements:
Christos Louizos's original theano implementation https://arxiv.org/abs/1511.00830
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


class VFAE(DeepGenerativeModelMixin, nn.Module):

    """
    Variational Fair Autoencoder

    Implements a generative model with two layers of stochastic variables,
    where both are conditional, i.e.:
        p(x, z1, z2, y | s) = p(z2)p(y)p(z1|z2,y)p(x|z1, s)
    with q(z1|x,s)q(z2|z1,y)q(y|z1) being the variational posteriors.

    Furthermore there is an extra MMD penalty on z1 to further enforce independence between z1 and s.

    Louizos, Swersky, Li, Welling, Zemel. The Variational Fair Autoencoder. ICLR 2016
    https://arxiv.org/abs/1511.00830
    """

    def __init__(self,
                 dim_x,   # data space dim
                 dim_s,   # nuisance variable dim (number of classes)
                 dim_y,   # number of classes
                 dim_h_en_z1=(50, 50),  # q(z1|x1,s)
                 dim_h_de_z1=(50, 50),  # p(z1|z2,y)
                 dim_h_en_z2=(50, 50),  # q(z2|z1,y)
                 dim_h_de_x=(50, 50),   # p(x1|z1,s)
                 dim_h_clf=(50, 50),    # q(y|z1)
                 dim_z1=50,
                 dim_z2=50,
                 type_rec='binary',
                 type_y='discrete', prior_y='uniform', semi_supervised=False, clf_1sig=False,
                 epochs=500, batch_size=100, nonlinearity='softplus',
                 learning_rate=0.001, optim_alg='adam', L=1, weight_decay=None,
                 dropout_rate=0., input_x_dropout=0., add_noise_var=0.,
                 yloss_rate=1., anneal_yloss_offset=0,
                 use_MMD=True, kernel_MMD='rbf_fourier', mmd_rate=1., use_s=False,
                 random_seed=12345, log_txt=None):
        super(VFAE, self).__init__()
        ## set parameters as instance variables
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop('self')
        for arg, val in values.items():
            setattr(self, arg, val)

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
        # self.anneal_yloss_offset = anneal_yloss_offset
        self.finished_training_iters = 0

        ## Log file
        self.w2log('VFAE:')
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
        
        ## Create the encoder for z1 q(z1|x,s)
        if self.use_s:
            in_qz1 = [self.dim_x, self.dim_s]
            in_dropouts = [self.input_x_dropout, 0]
        else:
            in_qz1 = [self.dim_x]
            in_dropouts = [self.input_x_dropout]
        self.encoder_z1 = blk.DiagGaussianModule(in_qz1, self.dim_h_en_z1, self.dim_z1, input_dropout_rates=in_dropouts,
                                prior_mu=self.prior_mu, prior_sg=self.prior_sg, **common_hyper_params)
        # self.encoder_z1 = blk.GaussianEncoderMadeIAF(in_qz1, self.dim_h_en_z1, self.dim_z1,
        #                         prior_mu=self.prior_mu, prior_sg=self.prior_sg, iaf_length=2, iaf_style='LSTM-like', **common_hyper_params)

        ## Create the encoder for y q(y|z1)
        if self.type_y == 'discrete':
            if self.clf_1sig:
                if self.dim_y != 2: raise ValueError('Invalid combination of clf_1sig and dim_y')
                out_dim = 1
            else:
                out_dim = self.dim_y
            self.encoder_y = blk.CategoricalDecoder([self.dim_z1], self.dim_h_clf, out_dim, **common_hyper_params)
        else:  # regression case
            self.encoder_y = blk.DiagGaussianModule([self.dim_z1], self.dim_h_clf, self.dim_y,
                                    fixed_variance=0.01, constrain_means=True, **common_hyper_params)
            # self.encoder_y = blk.BernoulliDecoder([self.dim_z1], self.dim_h_clf, self.dim_y,
            #                         **common_hyper_params)

        ## Create the encoder for z2 q(z2|z1,y)
        self.encoder_z2 = blk.DiagGaussianModule([self.dim_z1, self.dim_y], self.dim_h_en_z2, self.dim_z2,
                                prior_mu=self.prior_mu, prior_sg=self.prior_sg, **common_hyper_params)
        # self.encoder_z2 = blk.GaussianEncoderMadeIAF([self.dim_z1, self.dim_y], self.dim_h_en_z2, self.dim_z2,
        #                         prior_mu=self.prior_mu, prior_sg=self.prior_sg, iaf_length=1, iaf_style='LSTM-like', **common_hyper_params)

        ## Create the decoder for p(z1|z2,y)
        self.decoder_z1 = blk.DiagGaussianModule([self.dim_z2, self.dim_y], self.dim_h_de_z1, self.dim_z1,
                                **common_hyper_params)

        ## Create the decoder for p(x|z1,s)
        if self.use_s:
            in_px = [self.dim_z1, self.dim_s]
        else:
            in_px = [self.dim_z1]
        self.decoder_x = data_decoder(in_px, self.dim_h_de_x, self.dim_x, **data_decoder_hyper_params)

        # if self.nonlinearity == 'selu':
        #     for m in self.modules():
        #         if isinstance(m, nn.Linear):
        #             size = m.weight.size()
        #             fan_out = size[0] # number of rows
        #             fan_in = size[1] # number of columns
        #             stddev=np.sqrt(1. / fan_in)
        #             # print(m, m.weight.data.mean(), m.weight.data.std(), stddev)
        #             m.weight.data.normal_(0.0, stddev)
        #             print(m, m.weight.data.mean(), m.weight.data.std())
    
    def forward(self, x1, s=[]):
        """
        Run inference in the model to:
        - predict y
        - embedding qz1
        - reconstruction px1
        """
        self.eval()

        # posterior q(z1|x1, s)
        if self.use_s:
            s1inK = blk.one_hot(s, self.dim_s)
            in_qz1 = [x1, s1inK]
        else:
            in_qz1 = [x1]
        qz1 = self.encoder_z1(in_qz1)
        z1_mu = qz1[0]
            
        ## prediction: q(y|z1)
        qypred = self.encoder_y([z1_mu])
        if self.type_y == 'discrete':
            classifier_pred = self.encoder_y.most_probable(*qypred)
            proba = qypred[0]
        else:
            if len(qypred) > 1:
                classifier_pred, proba = qypred[0], qypred[1]
            else: # for Bernoulli decoder
                classifier_pred, proba = qypred[0], qypred[0]

        ## p(x1|z1, s)
        if self.use_s:
            in_px1 = [z1_mu, s1inK]
        else:
            in_px1 = [z1_mu]
        px1 = self.decoder_x(in_px1)
        x1_mu = px1[0]

        return {'pred':classifier_pred, 'proba':proba, 'z1':z1_mu, 'qz1':qz1, 'px1':px1, 'x1_rec':x1_mu}

    def predict(self, **kwargs):
        res = self.forward(**kwargs)
        pred = res['pred'].data.squeeze().numpy()
        proba = res['proba'].data.numpy()
        return pred, proba

    def reconstruct(self, **kwargs):
        res = self.forward(**kwargs)
        x1_rec = res['x1_rec'].data.numpy()
        px1 = [_x.data.numpy() for _x in res['px1']]
        return x1_rec, px1

    def transform(self, **kwargs):
        res = self.forward(**kwargs)
        z1 = res['z1'].data.numpy()
        return z1

    def _fprop(self, z1, qz1, y):
        """
        Propagate through the generative model for a given class Y
        """
        KLD_perx = 0.
        z1_sample = z1[0]
        
        ## q(z2|z1,y)
        qz2 = self.encoder_z2([z1_sample, y])
        z2 = self.encoder_z2.sample(*qz2)
        z2_sample = z2[0] # for compatibility with IAF encoders return a tuple

        # KL-divergence q(z2|z1,y) || p(z2) ; p(z2)=N(0,I)
        try:
            KLD_perx += self._use_free_bits(self.encoder_z2.kldivergence_from_prior_perx(*qz2)) # add KL from prior
        except:
            # no KL-divergence, use logq(z) - logp(z) Monte Carlo estimation
            logq_perx = self.encoder_z2.logp_perx(*(z2 + qz2))
            logp_perx = self.encoder_z2.logp_prior_perx(z2_sample)
            KLD_perx += self._use_free_bits(logq_perx - logp_perx)

        ## p(z1|z2,y)
        pz1 = self.decoder_z1([z2_sample, y])
        # KL-divergence q(z1|x1,s) || p(z1|z2,y)
        try:
            KLD_perx += self._use_free_bits(self.encoder_z1.kldivergence_perx(*(qz1 + pz1)) )
        except:
            # no KL-divergence, use logq(z) - logp(z) Monte Carlo estimation
            logq_perx = self.encoder_z1.logp_perx(*(z1 + qz1)) 
            logp_perx = self.decoder_z1.logp_perx(z1_sample, *pz1)
            KLD_perx += self._use_free_bits(logq_perx - logp_perx)

        return KLD_perx
    
    def _compute_losses(self, x1, y, s, L):
        """
        Compute all losses of the model. For unlabeled data marginalize y.
        RECL - reconstruction loss E_{q(z1|x1)}[ p(x1|z1) ]
        KLD  - kl-divergences of all the other matching q and p distributions
        YL   - prediction loss on y (for labeled data)
        MMD  - maximum mean discrepancy of z1 embedding w.r.t. grouping s
        """
        RECL, KLD, YL, MMD = 0., 0., 0., 0.
        N = x1.size(0)

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

        # get q(z1|x1, s)
        if self.use_s:
            in_qz1 = [x1, s1inK]
        else:
            in_qz1 = [x1]
        if self.training and self.add_noise:
            eps = x1.data.new(x1.size()).normal_()
            eps = Variable(eps.mul_(self.add_noise_var))
            in_qz1[0] += eps
        qz1 = self.encoder_z1(in_qz1)

        Lf = 1. * L
        for _ in range(L):
            # sample from q(z1|x1, s)
            z1 = self.encoder_z1.sample(*qz1) 
            z1_sample = z1[0] # for compatibility with IAF encoders return a tuple

            ## get the reconstruction loss
            # p(x1|z1,s) where z1 ~ q(z1|x1,s)
            if self.use_s:
                in_px1 = [z1_sample, s]
            else:
                in_px1 = [z1_sample]
            px1 = self.decoder_x(in_px1)
            RECL += self.decoder_x.logp(x1, *px1) / Lf

            ## prediction: q(y|z1)
            qy = self.encoder_y([z1_sample])

            _KLD = 0.
            if len(y) > 0 :
                ## if y is given then
                # (i) compute prediction loss
                if self.type_y == 'discrete':
                    YL += self.encoder_y.logp(y, *qy) / Lf
                    # ## Margin Ranking Loss
                    # MRL = 0.
                    # for i in range(1, y.size(0)-1):
                    #     a = qy[0][:-i,1]
                    #     b = qy[0][i:,1]
                    #     ya = y.view(-1)[:-i]
                    #     yb = y.view(-1)[i:]
                    #     t = ((ya > yb).float() * 2 ) - 1
                    #     idx = torch.nonzero(ya.data != yb.data).view(-1)
                    #     if len(idx) == 0: continue
                    #     # print(a[:20], b[:20], t[:20])
                    #     MRL += -F.margin_ranking_loss(a[idx], b[idx], t[idx])
                    # YL += MRL / y.size(0) / Lf / 100.
                    # # print(MRL)
                else:
                    # print(y.size(), qy[0].size())
                    # print(y.view(-1) - qy[0].view(-1))
                    ## Squared Error Loss
                    YL += -((y.view(-1) - qy[0].view(-1)) ** 2).sum() / Lf
                    # ## Margin Ranking Loss
                    # MRL = 0.
                    # for i in range(1, y.size(0)-1):
                    #     a = qy[0].view(-1,1)[:-i]
                    #     b = qy[0].view(-1,1)[i:]
                    #     t = ((y.view(-1)[:-i] > y.view(-1)[i:]).float() * 2 ) - 1
                    #     # print(a[:20], b[:20], t[:20])
                    #     MRL += -F.margin_ranking_loss(a, b, t)
                    # YL += MRL / y.size(0) / Lf
                    # # print(MRL)
                    ## Log Likelihood
                    # YL += self.encoder_y.logp(y, *qy) / Lf
                # (ii) condition on true y
                if self.type_y == 'discrete':
                    _y = y1inK
                else:
                    _y = y
                    if _y.ndimension() == 1:
                        _y.data.unsqueeze_(1)
                # the logprior of y is ommited as it is constant wrt the optimization
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
                    _y = self.encoder_y.sample(*qy)                    
                    _KLD = self._fprop(z1, qz1, _y)
                    # add logprior of y
                    if self.prior_y != 'uniform':
                        _KLD += self.encoder_y.kldivergence_perx(*(qy + self.prior_y)).sum(1)
            KLD += torch.sum(_KLD) / Lf

            # maximum mean discrepancy regularization
            if self.use_s and self.use_MMD:
                MMD += self._get_mmd_criterion(z1_sample, sind) / Lf
        
        # yhat = self.encoder_y.most_probable(*qy)
        # print(yhat.eq(y).data.numpy().mean())
        
        ## loss per batch
        return OrderedDict([('RECL',RECL), ('KLD',KLD), ('YL',YL), ('MMD',MMD)])

    def loss_function(self, x1, s, y, has_y):
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
        # print("annealing KLD, YR: ", beta_kl, beta_yr, end='; ')


        ## compute model loss
        # mask for labeled data
        mask_l = torch.nonzero(has_y.data).view(-1)
        Nl = len(mask_l)
        # mask for unlabeled data
        mask_u = torch.nonzero((has_y.data + 1) % 2).view(-1)
        Nu = len(mask_u)
        if self.semi_supervised:
            # labeled data
            if Nl != 0:
                losses_l = self._compute_losses(x1=x1[mask_l], y=y[mask_l], s=s[mask_l], L=self.L)
            else:
                warnings.warn("No labeled data in the minibatch")
                losses_l = OrderedDict([('RECL',0.), ('KLD',0.), ('YL',0.), ('MMD',0.)])
            # unlabeled data
            if Nu != 0:
                losses_u = self._compute_losses(x1=x1[mask_u], y=[], s=s[mask_u], L=self.L)
            else:
                warnings.warn("No unlabeled data in the minibatch")
                losses_u = OrderedDict([('RECL',0.), ('KLD',0.), ('YL',0.), ('MMD',0.)])
            ## sum and normalize the losses per example
            losses = OrderedDict()
            losses['RECL'] = (losses_l['RECL'] + losses_u['RECL']) / (Nl + Nu)
            losses['KLD']  = (losses_l['KLD'] + losses_u['KLD']) / (Nl + Nu)
            losses['YL']   = losses_l['YL'] / Nl
            losses['MMD']  = (losses_l['MMD'] + losses_u['MMD']) / (Nl + Nu)
        else:
            if Nu != 0:
                warnings.warn("Sampled unlabeled data in supervised-only model, ignoring them")
            losses_l = self._compute_losses(x1=x1[mask_l], y=y[mask_l], s=s[mask_l], L=self.L)
            ## normalize the losses per example
            losses = OrderedDict([(k, losses_l[k]/Nl) for k in losses_l.keys()])

        ## ELBO
        losses['ELBO'] = losses['RECL'] -(beta_kl * losses['KLD'])

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
        s = Variable(ds.s, volatile=True)
        y = Variable(ds.y, volatile=True)
        has_y = Variable(ds.has_y, volatile=True)
        return self.evaluate_performance(x1, s, y, has_y, return_full_data)

    def evaluate_performance(self, x1, s, y, has_y, return_full_data=False):
        """
        Evaluate the model performance
        """
        perf = OrderedDict()

        ## eval losses
        try:
            losses = self.run_on_batch(train_mode=False, x1=x1, s=s, y=y, has_y=has_y)
            perf['losses'] = losses
        except Exception as e:
            print("Warning, computation of losses failed in evaluation!")
            print(e)
            perf['losses'] = None

        ## run inference in the model on all data
        res = self.forward(x1, s)
        
        #### eval prediction performance on "y" for labeled data
        mask = torch.nonzero(has_y.data).view(-1)
        ylab = y[mask]
        pred = res['pred'][mask]
        proba = res['proba'][mask]
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
        
        ## return also full data
        if return_full_data:
            perf['z1'] = res['z1'].data.numpy()
            perf['pred'] = res['pred'].data.numpy()
            perf['proba'] = res['proba'].data.numpy()

        perf['model_class'] = self.__class__.__name__
        combined_str = 'Y: {}\t X1: {}'.format(y_str, rec_str)
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
        patience = 40              # at least this many epochs
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
                for batch_idx, (data_x1, data_s, data_y, data_has_y) in enumerate(train_loader):
                    data_x1 = Variable(data_x1)
                    data_s = Variable(data_s)
                    data_y = Variable(data_y)
                    data_has_y = Variable(data_has_y)
                    loss = self.run_on_batch(train_mode=True, x1=data_x1, s=data_s, y=data_y, has_y=data_has_y)
                    train_loss += loss['RECL'].data[0] + self.yloss_rate * loss['YL'].data[0]

                    if verbose and batch_idx % max(10, len(train_loader)/10) == 0:
                        # test prediction performance on labeled examples in the minibatch
                        train_perf, perf_str = self.evaluate_performance(x1=data_x1, s=data_s, y=data_y, has_y=data_has_y)

                        # str_losses = '\t'.join([(k+': {:.2f}').format(loss[k] if isinstance(loss[k], float) else loss[k].data[0]) for k in loss.keys()])
                        str_losses = '\t'.join([(k+': {:.3f}').format(loss[k].data[0]) for k in ['CMPL', 'ELBO', 'YL']])
                        self.w2log('Train Epoch: {} [{}/{} ({:.0f}%)]\t{}\t{}'.format(
                            epoch, batch_idx * len(data_x1), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            str_losses,
                            perf_str))
                train_loss /= len(train_loader)

                ## run prediction on the whole train set
                train_perf, train_str = self.evaluate_performance_on_dataset(train_loader.dataset)
                dt = time.time() - t
                self.w2log('====> Epoch: {}\tIter: {}'.format(epoch, self.finished_training_iters))
                self.w2log('Train: sec/epoch: {:.2f}\tAvg train loss: {:9.4f}\t{}'.format(dt, train_loss, train_str))

                #### validation -----------------------------------------------
                beta_recl = 1.
                # if self.anneal_yloss:
                #     beta_recl = 1. - self._compute_anneal_coef(self.finished_training_iters,
                #                                         iter_max = self.anneal_yloss_itermax,
                #                                         iter_offset = self.anneal_yloss_offset)
                t = time.time()
                valid_loss = 0
                # for i, (data_x1, data_s, data_y, data_has_y) in enumerate(valid_loader):
                #     data_x1 = Variable(data_x1, volatile=True)
                #     data_s = Variable(data_s, volatile=True)
                #     data_y = Variable(data_y, volatile=True)
                #     data_has_y = Variable(data_has_y, volatile=True)
                #     loss = self.run_on_batch(train_mode=False, x1=data_x1, s=data_s, y=data_y, has_y=data_has_y)
                #     valid_loss += beta_recl * loss['RECL'].data[0] + self.yloss_rate * loss['YL'].data[0]
                # valid_loss /= len(valid_loader)
                
                ## run prediction on the whole validation set 
                valid_perf, perf_str = self.evaluate_performance_on_dataset(valid_loader.dataset)
                if self.type_y == 'discrete':
                    valid_loss = valid_perf['y_auroc'] + valid_perf['y_aupr'] + valid_perf['x1_pearr']
                else:
                    valid_loss = valid_perf['y_r2'] + valid_perf['y_pearr'] + valid_perf['x1_pearr']
                # valid_loss = valid_perf['losses']['RECL'].data[0] + self.yloss_rate * valid_perf['losses']['YL'].data[0]
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


class VFAEDataset(torch.utils.data.Dataset):
    """
    pytorch wrapper for semi-supervised VFAE dataset containing
    "x1", "s", "y", "has_y"
    """
    def __init__(self, x1, s, y, has_y):
        assert x1.size(0) == y.size(0)
        assert x1.size(0) == s.size(0)
        assert x1.size(0) == has_y.size(0)
        self.x1 = x1
        self.s = s
        self.y = y
        self.has_y = has_y

    def __getitem__(self, index):
        return self.x1[index], self.s[index], self.y[index], self.has_y[index]

    def __len__(self):
        return self.x1.size(0)
        

def wrap_in_VFAEDataset(sing, pair, y_key='y', concat=True, downlabel_to=None, remove_unlabeled=False):
    """
    Wrap given data in pytorch VFAEDataset
    @sing - dict with singleton data
    @pair - dict with paired perturbation data
    @y_key - selects target variable key,
            i.e. 'y' for discrete or 'ycont' for continuous labels
    @concat - concatenate @sing and @pair on x1, if False, ignore @pair
    return: VFAEDataset instance, dictionary of the data in numpy arrays
    """
    if concat:
        ## concatenate the @sing and @pair on common keys
        common_keys = set(sing.keys()) & set(pair.keys())
        ddict = dict([(k, np.concatenate((sing[k], pair[k]))) for k in common_keys])
    else:
        ddict = sing

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

    dataset = VFAEDataset(
        x1=torch.from_numpy(ddict['x1']).float(),
        s=torch.from_numpy(ddict['s'].astype(np.int32)),
        y=torch.from_numpy(ddict[y_key]),
        has_y=torch.from_numpy(ddict['has_y'].astype(np.int32)))
    # if y_key=='ycont':
    #     dataset.y = dataset.y.float()
    return dataset, ddict
