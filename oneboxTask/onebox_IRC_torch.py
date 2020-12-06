from oneboxTask.onebox import *
from oneboxTask.onebox_HMM import *

from sklearn.decomposition import PCA
from sklearn import random_projection
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from torch import optim

class onebox_IRC_torch():
    def __init__(self, discount, nq, nr, na, nl, parametersInit):
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.nl = nl
        self.n = (self.nq ** self.nl) * self.nr   # total number of states

        self.para = parametersInit  # dictionary of parameters

        self.point_all = []  # List of parameters
        self.log_likelihood_all = []  # List of likelihood


    # def IRC_randomProj(self, obs, randProjNum = 1):
    #     for l in range(randProjNum):
    #         transformer = random_projection.GaussianRandomProjection(n_components=2)
    #         transformer.fit_transform(self.point_all)
    #         projectionMatRand = transformer.components_
    #         print(projectionMatRand)
    #
    #         # Contour of the likelihood
    #         step1 = 0.004  # for u (1st principle component)
    #         step2 = 0.004  # for v (2nd principle component)
    #         N1 = 4
    #         N2 = 4
    #         uOffset = - step1 * N1 / 2
    #         vOffset = - step2 * N2 / 2
    #
    #         uValue = np.zeros(N1)
    #         vValue = np.zeros(N2)
    #         Qaux1 = np.zeros((N2, N1))  # Likelihood with ground truth latent
    #         Qaux2 = np.zeros((N2, N1))  # Expected complete data likelihood
    #         Qaux3 = np.zeros((N2, N1))  # Entropy of latent posterior
    #         para_slice = []
    #         LL_slice = []
    #
    #         for i in range(N1):
    #             uValue[i] = step1 * (i) + uOffset
    #             for j in range(N2):
    #                 vValue[j] = step2 * (j) + vOffset
    #
    #                 para_slicePoints = self.point_all[-1] + uValue[i] * projectionMatRand[0] + \
    #                                    vValue[j] * projectionMatRand[1]
    #                 para_slice.append(para_slicePoints)
    #                 para = np.copy(para_slicePoints)
    #                 # para[2] = 0
    #                 # para[3] = 0
    #
    #                 onebox = oneboxMDP(self.discount, self.nq, self.nr, self.na, para)
    #                 onebox.setupMDP()
    #                 onebox.solveMDP_sfm()
    #                 ThA = onebox.ThA
    #                 softpolicy = onebox.softpolicy
    #                 latent_ini = np.ones(self.nq) / self.nq   # initialize the estimation of the belief state
    #                 onebpx_HMM = HMMonebox(ThA, softpolicy, latent_ini)
    #
    #                 # Qaux1[j, i] = oneboxHMM.likelihood(lat, obs, ThA, optpolicy)  #given latent state
    #                 Qaux2[j, i] = onebpx_HMM.computeQaux(obs, ThA, softpolicy)
    #                 Qaux3[j, i] = onebpx_HMM.latent_entr(obs)
    #                 LL_slice.append(Qaux2[j, i] + Qaux3[j, i])
    #
    #         Loglikelihood = Qaux2 + Qaux3
    #         Loglikelihood = np.nan_to_num(Loglikelihood, nan=-100000000000)
    #         max_point_idx = np.where(Loglikelihood == np.max(Loglikelihood))
    #         max_point = para_slice[max_point_idx[1][0] * N2 + max_point_idx[0][0]]
    #         print(np.max(Loglikelihood), max_point)
    #
    #         if np.max(Loglikelihood) > self.log_likelihood_all[-1]:
    #             self.point_all.append(max_point)
    #             self.log_likelihood_all.append(np.max(Loglikelihood))
    #
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

        pi = torch.ones(self.nq) / self.nq

        onebox_temp = oneboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, self.para)
        onebox_temp.setupMDP()
        onebox_temp.solveMDP_sfm()
        ThA = onebox_temp.ThA
        softpolicy = onebox_temp.softpolicy
        oneboxHMM = HMMonebox(ThA, softpolicy, pi)

        log_likelihood = oneboxHMM.log_likelihood(obs, ThA, softpolicy)
        log_likelihood /= len(obs)

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
                #print(self.para)
                #print(-loss)

                optimizer.zero_grad()
                loss.backward()
                #print(i, [p.grad for k, p in self.para.items()])
                optimizer.step()
                #print(self.para)
                #print('\n\n')

            if epoch % 5 == 4:
                print("epoch: %d, loss: %1.3f" % (epoch + 1, loss))

            if len(self.log_likelihood_all) >= 2 and torch.abs(self.log_likelihood_all[-1] - self.log_likelihood_all[-2]) < eps:
                break
            if epoch >= 200:
                break

            epoch += 1



    # def IRC(self, obs, lr, eps):
    #
    #     n_epochs = 0
    #     optimizer = optim.SGD(list(self.para.values()), lr=lr)
    #
    #     while True:
    #         optimizer.zero_grad()
    #         loss = - self.likelihood_tensor(obs)
    #
    #         self.point_all.append(self.para)
    #         self.log_likelihood_all.append(loss)
    #         # print(self.para)
    #         # print(-loss)
    #
    #         loss.backward()
    #         # print([p.grad for k, p in self.para.items()], '\n\n')
    #         optimizer.step()
    #
    #         if len(self.log_likelihood_all) >= 2 and \
    #                 torch.abs(self.log_likelihood_all[-1] - self.log_likelihood_all[-2]) < eps:
    #             break
    #         n_epochs += 1


    # def contour_LL(self, obs, step1 = 0.02, step2 = 0.02, N1 = 6, N2 = 6, proj = 'PCA'):
    #     projectionMat = np.zeros((2, len(self.point_all[-1])))
    #
    #     if proj == 'rand':
    #         transformer = random_projection.GaussianRandomProjection(n_components=2)
    #         transformer.fit_transform(self.point_all)
    #         projectionMat = transformer.components_
    #     elif proj == 'PCA':
    #         pca = PCA(n_components=2)
    #         pca.fit(np.unique((self.point_all - self.point_all[-1]), axis=0))
    #         projectionMat = pca.components_
    #
    #     uOffset = - step1 * N1 / 2
    #     vOffset = - step2 * N2 / 2
    #
    #     uValue = np.zeros(N1)
    #     vValue = np.zeros(N2)
    #     Qaux1 = np.zeros((N2, N1))  # Likelihood with ground truth latent
    #     Qaux2 = np.zeros((N2, N1))  # Expected complete data likelihood
    #     Qaux3 = np.zeros((N2, N1))  # Entropy of latent posterior
    #     para_slice = []
    #
    #     for i in range(N1):
    #         uValue[i] = step1 * (i) + uOffset
    #         for j in range(N2):
    #             vValue[j] = step2 * (j) + vOffset
    #
    #             para_slicePoints = self.point_all[-1] + uValue[i] * projectionMat[0] + vValue[j] * projectionMat[1]
    #             para_slice.append(para_slicePoints)
    #             para = np.copy(para_slicePoints)
    #             #para[2] = 0
    #             #para[3] = 0
    #
    #             onebox = oneboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para)
    #             onebox.setupMDP()
    #
    #             if np.any(onebox.ThA < 0) == True:
    #                 Qaux2[j, i] = np.nan
    #                 Qaux3[j, i] = np.nan
    #             else:
    #                 onebox.solveMDP_sfm()
    #                 ThA = onebox.ThA
    #                 softpolicy = onebox.softpolicy
    #                 latent_ini = np.ones(self.nq) / self.nq   # initialize the estimation of the belief state
    #                 oneHMM = HMMonebox(ThA, softpolicy, latent_ini)
    #
    #                 # Qaux1[j, i] = oneboxHMM.likelihood(lat, obs, ThA, optpolicy)  #given latent state
    #                 Qaux2[j, i] = oneHMM.computeQaux(obs, ThA, softpolicy)
    #                 Qaux3[j, i] = oneHMM.latent_entr(obs)
    #
    #         contour_LL_mesh = Qaux2 + Qaux3
    #         contour_LL_mesh = np.nan_to_num(contour_LL_mesh, nan = np.nanmean(contour_LL_mesh))
    #
    #         self.uValue = uValue
    #         self.vValue = vValue
    #         self.contour_LL_mesh = contour_LL_mesh
    #         self.point_2d = projectionMat.dot((self.point_all - self.point_all[-1]).T).T
    #         self.projectionMat = projectionMat
    #
    # def plot_contour_LL(self):
    #     # project the trajectories onto the plane
    #
    #     fig_contour, ax = plt.subplots(figsize=(6, 6))
    #     uValuemesh, vValuemesh = np.meshgrid(self.uValue, self.vValue)
    #     plt.contourf(uValuemesh, vValuemesh, self.contour_LL_mesh,
    #                  np.arange(np.min(self.contour_LL_mesh),
    #                            np.max(self.contour_LL_mesh), 50), cmap='jet')
    #
    #     plt.plot(self.point_2d[:, 0], self.point_2d[:, 1], marker='.', color='b')  # projected trajectories
    #     # plt.plot(point_2d[-1, 0], point_2d[-1, 1], marker='*', color = 'g', markersize = 10)        # final point
    #     # plt.plot(true_2d[0], true_2d[1], marker='o', color = 'g')           # true
    #
    #     # ax.grid()
    #     ax.set_title('Likelihood of observed data')
    #     plt.xlabel(r'$u \mathbf{\theta}$', fontsize=10)
    #     plt.ylabel(r'$v \mathbf{\theta}$', fontsize=10)
    #     # plt.clabel(cs3, inline=1, fontsize=10)
    #     plt.colorbar()
    #     plt.show()
    #
    #




