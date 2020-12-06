from twoboxTask.twobox import *
from twoboxTask.twobox_HMM import *

from sklearn.decomposition import PCA
from sklearn import random_projection
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from torch import optim

class twobox_IRC_torch():
    def __init__(self, discount, nq, nr, na, nl, parametersInit):
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.nl = nl
        self.n = (self.nq ** 2) * self.nr   # total number of states

        self.para = parametersInit  # dictionary of parameters

        self.point_all = []  # List of parameters
        self.log_likelihood_all = []  # List of likelihood

    def likelihood_tensor_ave(self, obs):
        """
        average log-likelihood of samples
        :param obs:
        :return:
        """
        ns = obs.shape[0]
        log_likelihood = 0
        for i in range(ns):
            log_likelihood += self.likelihood_tensor(obs[i])
        log_likelihood /= ns

        return log_likelihood

    def likelihood_tensor(self, obs):
        """
        log-likelihood of a single sample
        # obs is from one sample. [act, rew] at each time point
        """

        pi = torch.ones(self.nq ** 2) / self.nq / self.nq

        twobox = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, self.para)
        twobox.setupMDP()
        twobox.solveMDP_sfm()
        ThA = twobox.ThA
        softpolicy = twobox.softpolicy

        twoboxHMM = HMMtwobox(ThA, softpolicy, pi)

        log_likelihood = twoboxHMM.log_likelihood(obs, ThA, softpolicy)
        #log_likelihood /= len(obs)

        return log_likelihood

    def IRC_batch(self, obsN, lr, eps, batch_size, shuffle):
        obsN_loader = data_utils.DataLoader(obsN, batch_size, shuffle)

        epoch = 0
        optimizer = optim.SGD(list(self.para.values()), lr=lr)

        while True:
            for i, obsN_minibatch in enumerate(obsN_loader):
                # data contains data in one mini-batches, contains a few samples
                loss = - self.likelihood_tensor_ave(obsN_minibatch)

                self.point_all.append(self.para)
                self.log_likelihood_all.append(loss)
                print(self.para)
                print(-loss)

                optimizer.zero_grad()
                loss.backward()
                print(i, [p.grad for k, p in self.para.items()])
                optimizer.step()
                #print(self.para)
                print('\n\n')

            if epoch % 5 == 4:
                print("epoch: %d, loss: %1.3f" % (epoch + 1, loss))

            if len(self.log_likelihood_all) >= 2 and torch.abs(self.log_likelihood_all[-1] - self.log_likelihood_all[-2]) < eps:
                break
            if epoch >= 200:
                break

            epoch += 1
