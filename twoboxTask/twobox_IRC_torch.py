from twoboxTask.twobox import *
from twoboxTask.twobox_HMM import *
import pickle
import os
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn import random_projection
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from torch import optim
import numpy as np


class twobox_IRC_torch():
    def __init__(self, discount, nq, nr, na, nl, parametersInit):
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.nl = nl
        self.n = (self.nq ** 2) * self.nr   # total number of states

        self.para = parametersInit  # dictionary of parameters

        self.point_traj = []  # List of parameters
        self.log_likelihood_traj = []  # List of likelihood
        self.log_likelihood_whole = []

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
        #return softpolicy[0, 0]
        return log_likelihood

    def IRC_batch(self, obsN, lr, eps, batch_size, shuffle):
        obsN_loader = data_utils.DataLoader(obsN, batch_size, shuffle)

        epoch = 0
        optimizer = optim.SGD(list(self.para.values()), lr=lr)

        while True:
            loss_whole = 0
            for i, obsN_minibatch in enumerate(obsN_loader):
                # data contains data in one mini-batches, contains a few samples
                loss = - self.likelihood_tensor_ave(obsN_minibatch)
                loss_whole += loss

                para_temp = self.para.copy()
                for k, v in para_temp.items():
                    para_temp[k] = v.clone().detach()

                self.point_traj.append(para_temp)
                self.log_likelihood_traj.append(-loss)
                print(self.para)
                #print(-loss)

                optimizer.zero_grad()
                loss.backward()
                print('batch-', i+1, [p.grad for k, p in self.para.items()])
                optimizer.step()
                print("After update, the parameters are: \n", self.para)
                print('\n\n')


            if obsN.shape[0] != 1:
                self.log_likelihood_whole.append(loss_whole)

                print("epoch: %d, loss: %1.3f" % (epoch + 1, loss_whole))

                if len(self.log_likelihood_whole) >= 2 and torch.abs(
                        self.log_likelihood_whole[-1] - self.log_likelihood_whole[-2]) < eps:
                    break
            else:

                if epoch % 10 == 0:
                    self.save_traj()

                print("epoch: %d, loss: %1.3f" % (epoch + 1, loss))

                if len(self.log_likelihood_traj) >= 2 and torch.abs(
                        self.log_likelihood_traj[-1] - self.log_likelihood_traj[-2]) < eps:
                    print('IRC Finished')
                    break

            if epoch >= 200:
                break

            epoch += 1

    def save_traj(self):
        datestring_IRC = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')  # current time used to set file name

        path = os.getcwd()
        ### write data to file
        tra_dict = {'point_traj': self.point_traj,
                     'log_likelihood_traj': self.log_likelihood_traj}
        #IRC_output = open(path + '/Data/' + datestring_IRC + '_IRC_twobox' + '.pkl', 'wb')
        #pickle.dump(tra_dict, IRC_output)
        #IRC_output.close()

        with open(path + '/' + datestring_IRC + '_IRC_twobox' + '.pkl', 'wb') as IRC_output:
            pickle.dump(tra_dict, IRC_output)


    def contour_LL(self, obsN, step1 = 0.02, step2 = 0.02, N1 = 6, N2 = 6, proj = 'PCA'):

        ptraj_array = np.array([np.array([v.clone().detach().numpy() for k, v in self.point_traj[j].items()]).squeeze()
                                for j in range(len(self.point_traj))])

        if proj == 'rand':
            transformer = random_projection.GaussianRandomProjection(n_components=2)
            transformer.fit_transform(ptraj_array)
            projectionMat = transformer.components_
        elif proj == 'PCA':
            pca = PCA(n_components=2)
            pca.fit(np.unique((ptraj_array - ptraj_array[-1]), axis=0))
            projectionMat = pca.components_

        uOffset = - step1 * N1 / 2
        vOffset = - step2 * N2 / 2

        uValue = np.zeros(N1, dtype=np.single)
        vValue = np.zeros(N2, dtype=np.single)
        Qaux = np.zeros((N2, N1), dtype=np.single)  # log-Likelihood
        # Qaux2 = np.zeros((N2, N1))  # Expected complete data likelihood
        # Qaux3 = np.zeros((N2, N1))  # Entropy of latent posterior
        para_slice = []

        for i in range(N1):
            uValue[i] = step1 * (i) + uOffset
            for j in range(N2):
                vValue[j] = step2 * (j) + vOffset

                para_slicePoints = ptraj_array[-1] + uValue[i] * projectionMat[0] + vValue[j] * projectionMat[1]
                para_slice.append(para_slicePoints)
                para_array = np.copy(para_slicePoints)
                para = self.para.copy()
                for k, item in enumerate(self.para.items()):
                    para[item[0]] = torch.tensor(para_array[k])

                twobox = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para)
                twobox.setupMDP()

                if torch.any(twobox.ThA < 0) == True:
                    Qaux[j, i] = np.nan
                else:
                    twobox.solveMDP_sfm()
                    ThA = twobox.ThA
                    softpolicy = twobox.softpolicy
                    pi = torch.ones(self.nq * self.nq) / self.nq / self.nq  # initialize the estimation of the belief state
                    twoboxHMM = HMMtwobox(ThA, softpolicy, pi)

                    for n in range(obsN.shape[0]):
                        Qaux[j, i] += twoboxHMM.log_likelihood(obsN[n], ThA, softpolicy)  #given latent state

        contour_LL_mesh = Qaux
        contour_LL_mesh = np.nan_to_num(contour_LL_mesh, nan = np.nanmean(contour_LL_mesh))

        self.uValue = uValue
        self.vValue = vValue
        self.contour_LL_mesh = contour_LL_mesh
        self.point_2d = projectionMat.dot((ptraj_array - ptraj_array[-1]).T).T
        self.projectionMat = projectionMat

    def plot_contour_LL(self):
        # project the trajectories onto the plane

        fig_contour, ax = plt.subplots(figsize=(6, 6))
        uValuemesh, vValuemesh = np.meshgrid(self.uValue, self.vValue)
        plt.contourf(uValuemesh, vValuemesh, self.contour_LL_mesh,
                     np.arange(np.min(self.contour_LL_mesh),
                               np.max(self.contour_LL_mesh), 50), cmap='jet')

        plt.plot(self.point_2d[:, 0], self.point_2d[:, 1], marker='.', color='b')  # projected trajectories
        plt.plot(self.point_2d[-1, 0], self.point_2d[-1, 1], marker='*', color = 'g', markersize = 10) # final point
        # plt.plot(true_2d[0], true_2d[1], marker='o', color = 'g')           # true

        # ax.grid()
        ax.set_title('Likelihood of observed data')
        plt.xlabel(r'$u \mathbf{\theta}$', fontsize=10)
        plt.ylabel(r'$v \mathbf{\theta}$', fontsize=10)
        # plt.clabel(cs3, inline=1, fontsize=10)
        plt.colorbar()
        plt.show()
