"""
Ladislav Rampasek (rampasek@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats
import sklearn.metrics
import torch


class DeepGenerativeModelMixin:
    """
    Mix-in class with common methods between deep generative models
    """

    def w2log(self, *args):
        """
        Print log massage and write to file if self.log_txt is not None
        """
        print(*args)
        if self.log_txt is not None:
            if not os.path.exists('logs'):
                os.makedirs('logs')
            with open('logs/' + self.log_txt, 'a') as f:
                f.write(' '.join([str(e) for e in args]) + '\n')

    def _create_optimizer(self):
        """
        Create model's optimizer
        """
        if self.optim_alg == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim_alg == 'adamax':
            self.optimizer = torch.optim.Adamax(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError('Selected unknown optimizer: ' + self.optim_alg)

    def _get_mmd_criterion(self, z, sind):
        mmd = 0.
        for ind in sind:
            ind0 = Variable(ind.data.nonzero().squeeze())
            ind1 = Variable((ind == 0).data.nonzero().squeeze())
            if len(ind0) == 0:
                # if an S category does not have any samples add a random row
                if z.is_cuda:
                    z0 = Variable(torch.cuda.FloatTensor(1, z.size()[1]).normal_())
                else:
                    z0 = Variable(torch.FloatTensor(1, z.size()[1]).normal_())
            else:
                z0 = z.index_select(0, ind0)
            if len(ind1) == 0:
                # if an S category does not have any samples add a random row
                if z.is_cuda:
                    z1 = Variable(torch.cuda.FloatTensor(1, z.size()[1]).normal_())
                else:
                    z1 = Variable(torch.FloatTensor(1, z.size()[1]).normal_())
            else:
                z1 = z.index_select(0, ind1)
            mmd += - blk.mmd_objective(z0, z1, kernel=self.kernel_MMD)
            if len(sind) == 2:
                return mmd
        return mmd / len(sind)

    def _use_free_bits(self, KL_perx, override_default_kl_min=None):
        # if np.any(KL_perx.data.numpy() < self.kl_min_tt.data.numpy()):
        #     print(" --------->> APPLYING FREE BITS") #, KL_perx.data.numpy())
        if override_default_kl_min is not None:
            kl_min = Variable(torch.FloatTensor([override_default_kl_min]))
            return torch.max(KL_perx, kl_min.expand_as(KL_perx))
        else:
            return torch.max(KL_perx, self.kl_min_tt.expand_as(KL_perx))

    def _compute_anneal_coef(self, iter_num, iter_max=1000, iter_offset=0, func_type='linear'):
        """
        Compute coefficent for annealing for @iter_num -th training iteration
        such that it starts at 0.01 and reaches 1. at @iter_max -th iteration
        """
        if func_type == 'linear':
            if iter_num - iter_offset > 0:
                beta = min(1., 0.01 + (iter_num - iter_offset) / (1. * iter_max))
            else:
                beta = 0.01
        else:
            raise ValueError("Unknown annealing function: " + func_type)
        return beta

    def run_on_batch(self, train_mode=False, **kwargs):
        """
        Train/Evaluate the model on a given mini batch **kwargs
        """
        ## switch to appropriate mode
        if train_mode:
            ## TRAIN MODE
            self.train()

            if self.anneal_learning_rate:
                # adjust the optimizer for adaptive learning rate over time
                beta_lr = max(0.01, 1. - self._compute_anneal_coef(self.finished_training_iters, 3000))
                lr = beta_lr * self.learning_rate
                self.optimizer.state_dict()['param_groups'][0]['lr'] = lr
                # print('annealing LR: ', lr)

            # zero the gradients
            self.optimizer.zero_grad()

        else:
            ## EVAL MODE
            self.eval()

        ## compute model loss
        losses = self.loss_function(**kwargs)
        total_loss = losses['CMPL']

        if train_mode:
            ## TRAIN MODE
            # make backward pass
            total_loss.backward()
            # make optimizer step to update weights
            self.optimizer.step()
            self.finished_training_iters += 1

        return losses

    def eval_x_reconstruction(self, x, x_rec, x_rec_logvar=None):
        """
        Compute reconstruction evaluation metrics for X_k
        """
        x_np = x.data.numpy().astype(float)
        x_rec_np = x_rec.data.numpy().astype(float)
        pd = dict()
        pd['rmse'] = np.sqrt(((x_np - x_rec_np) ** 2).mean())
        # pd['mae'] = (x_np - x_rec_np).abs().mean()
        pd['r2'] = sklearn.metrics.r2_score(x_np, x_rec_np, multioutput='variance_weighted')

        ## compute correlation measures and average over the batch
        num_examples = float(x_np.shape[0])
        pd['pearr'] = 0.
        # pd['spearr'] = 0.
        # pd['taub'] = 0.
        for i in range(x_np.shape[0]):
            pd['pearr'] += scipy.stats.pearsonr(x_np[i, :], x_rec_np[i, :])[0] / num_examples
            # pd['spearr'] += scipy.stats.spearmanr(x_np[i,:], x_rec_np[i,:])[0] / num_examples
            # pd['taub'] += scipy.stats.kendalltau(x_np[i,:], x_rec_np[i,:])[0] / num_examples
        # pd['cos'] = 1. - sklearn.metrics.pairwise.paired_cosine_distances(x_np, x_rec_np).mean()
        # print(pd['pearr'], pd['spearr'], pd['taub'])

        # if x_rec_logvar is not None compute loglikelihood of x in given distribution
        if x_rec_logvar is not None:
            pd['ll'] = float(self.decoder_x.logp_perx(x, x_rec, x_rec_logvar).mean().data.numpy()[0])
        else:
            pd['ll'] = np.nan
        return pd

    def eval_y_prediction(self, pred, proba, ylab):
        """
        Compute classification/regression evaluation metrics for prediction over Y
        """
        pd = dict()
        if self.type_y == 'discrete':
            pd['acc'] = (pred.int() == ylab.int()).data.float().sum() / ylab.size(0)
            # pd['f1'] = sklearn.metrics.f1_score(ylab.data.numpy(), pred.data.int().numpy())
            # pd['ppv'] = (ylab.data.numpy()[pred.data.numpy() == 1] == 1).mean()
            if self.dim_y == 2:
                try:
                    pd['auroc'] = sklearn.metrics.roc_auc_score(ylab.data.numpy(), proba.data[:, 1].numpy())
                except:
                    pd['auroc'] = np.nan
                pd['aupr'] = sklearn.metrics.average_precision_score(ylab.data.numpy(), proba.data[:, 1].numpy())
            else:
                try:
                    pd['auroc'] = sklearn.metrics.roc_auc_score(blk.one_hot(ylab, self.dim_y).data.numpy(),
                                                                proba.data.numpy(), average='macro')
                except:
                    pd['auroc'] = np.nan
                pd['aupr'] = sklearn.metrics.average_precision_score(blk.one_hot(ylab, self.dim_y).data.numpy(),
                                                                     proba.data.numpy(), average='macro')
        else:
            ylab_np = ylab.data.numpy().astype(float)
            pred_np = pred.data.view(-1).numpy().astype(float)
            pd['rmse'] = np.sqrt(((ylab_np - pred_np) ** 2).mean())
            # pd['mae'] = (ylab_np - pred_np).abs().mean()
            pd['r2'] = sklearn.metrics.r2_score(ylab_np, pred_np)
            pd['pearr'] = scipy.stats.pearsonr(ylab_np, pred_np)[0]
            # pd['spearr'] = scipy.stats.spearmanr(ylab_np, pred_np)[0]
            # pd['taub'] = scipy.stats.kendalltau(ylab_np, pred_np)[0]
        return pd

    def save_to_file(self, filename):
        """
        Save parameters of this object instance to a file.
        """
        torch.save(self.state_dict(), filename)  # save parameters
        # torch.save(self, filename + '.model') #save the whole model

    def load_params_from_file(self, filename):
        """
        Load parameters of from file.
        """
        self.load_state_dict(torch.load(filename))
