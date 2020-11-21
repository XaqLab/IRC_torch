from oneboxTask.onebox import *
from oneboxTask.onebox_HMM import *
from oneboxTask.onebox_grad import *
from sklearn.decomposition import PCA
from sklearn import random_projection
import matplotlib.pyplot as plt

class onebox_IRC():
    def __init__(self, discount, nq, nr, na, nl, parametersInit, LLInit):
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.nl = nl
        self.n = (self.nq ** self.nl) * self.nr   # total number of states
        self.point_all = parametersInit.copy()   # List of parameters
        self.log_likelihood_all = LLInit.copy()  # List of likelihood

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
    #                 pi = np.ones(self.nq) / self.nq   # initialize the estimation of the belief state
    #                 onebpx_HMM = HMMonebox(ThA, softpolicy, pi)
    #
    #                 # Qaux1[j, i] = oneboxHMM.likelihood(lat, obs, ThA, policy)  #given latent state
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

    def IRC(self, obs, learn_rate, alpha_rate):
        while True:
            alpha = 1

            """
            Initial point for gradient descent
            """
            p_last = self.point_all[-1]
            max_ll_last = self.log_likelihood_all[-1]
            print('The current point is:', p_last)
            print('The current log-likelihood is: \n {}'.format(max_ll_last))

            """
            Gradient of the parameters
            """
            oneboxd = oneboxMDPder(self.discount, self.nq, self.nr, self.na, p_last)
            oneboxd1st = oneboxd.dloglikelihhod_dpara_sim(obs)
            print('The current gradient is', oneboxd1st)

            """
            Check the log-likelihood with an updated parameter based on gradient descent (better? not guaranteed)
            """
            para_temp = p_last + alpha * learn_rate * np.array(oneboxd1st)
            onebox_temp = oneboxMDP(self.discount, self.nq, self.nr, self.na, para_temp)
            onebox_temp.setupMDP()
            onebox_temp.solveMDP_sfm()
            ThA = onebox_temp.ThA
            softpolicy = onebox_temp.softpolicy
            pi = np.ones(self.nq) / self.nq   # initialize the estimation of the belief state
            oneboxHMM_temp = HMMonebox(ThA, softpolicy, pi)
            max_ll_temp = oneboxHMM_temp.computeQaux(obs, ThA, softpolicy) + \
                          oneboxHMM_temp.latent_entr(obs)
            print(' Potential new log-likelihood:', max_ll_temp)

            """
            if the updated log-likelihood is not good enough, we need to use Armijo rule to find the step-size
            """
            while max_ll_temp < max_ll_last + 0.2 * alpha * learn_rate * np.array(oneboxd1st).dot(np.array(oneboxd1st)):
                alpha /= alpha_rate
                para_temp = p_last + alpha * learn_rate * np.array(oneboxd1st)
                print(para_temp)

                ## Check the ECDLL (old posterior, new parameters)
                onebox_new = oneboxMDP(self.discount, self.nq, self.nr, self.na, para_temp)
                onebox_new.setupMDP()
                onebox_new.solveMDP_sfm()
                ThA_new = onebox_new.ThA
                softpolicy_new = onebox_new.softpolicy

                oneboxHMM_new = HMMonebox(ThA_new, softpolicy_new, pi)
                max_ll_temp_new = oneboxHMM_new.computeQaux(obs, ThA_new, softpolicy_new
                                                            ) + oneboxHMM_new.latent_entr(obs)

                print('    Amijio temp log-likelihood: ', max_ll_temp_new, '  (alpha = ', alpha, ')')
                if alpha < 10 ** -6:
                    break

                max_ll_temp = max_ll_temp_new

            print('\n\n')
            self.point_all.append(para_temp)
            self.log_likelihood_all.append(max_ll_temp)

            if len(self.log_likelihood_all) >= 2 and np.abs(
                    self.log_likelihood_all[-1] - self.log_likelihood_all[-2]) < 5 * 10 ** -2:
                print("GD finish")
                break


    def contour_LL(self, obs, step1 = 0.02, step2 = 0.02, N1 = 6, N2 = 6, proj = 'PCA'):
        projectionMat = np.zeros((2, len(self.point_all[-1])))

        if proj == 'rand':
            transformer = random_projection.GaussianRandomProjection(n_components=2)
            transformer.fit_transform(self.point_all)
            projectionMat = transformer.components_
        elif proj == 'PCA':
            pca = PCA(n_components=2)
            pca.fit(np.unique((self.point_all - self.point_all[-1]), axis=0))
            projectionMat = pca.components_

        uOffset = - step1 * N1 / 2
        vOffset = - step2 * N2 / 2

        uValue = np.zeros(N1)
        vValue = np.zeros(N2)
        Qaux1 = np.zeros((N2, N1))  # Likelihood with ground truth latent
        Qaux2 = np.zeros((N2, N1))  # Expected complete data likelihood
        Qaux3 = np.zeros((N2, N1))  # Entropy of latent posterior
        para_slice = []

        for i in range(N1):
            uValue[i] = step1 * (i) + uOffset
            for j in range(N2):
                vValue[j] = step2 * (j) + vOffset

                para_slicePoints = self.point_all[-1] + uValue[i] * projectionMat[0] + vValue[j] * projectionMat[1]
                para_slice.append(para_slicePoints)
                para = np.copy(para_slicePoints)
                #para[2] = 0
                #para[3] = 0

                onebox = oneboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para)
                onebox.setupMDP()

                if np.any(onebox.ThA < 0) == True:
                    Qaux2[j, i] = np.nan
                    Qaux3[j, i] = np.nan
                else:
                    onebox.solveMDP_sfm()
                    ThA = onebox.ThA
                    softpolicy = onebox.softpolicy
                    pi = np.ones(self.nq) / self.nq   # initialize the estimation of the belief state
                    oneHMM = HMMonebox(ThA, softpolicy, pi)

                    # Qaux1[j, i] = oneboxHMM.likelihood(lat, obs, ThA, policy)  #given latent state
                    Qaux2[j, i] = oneHMM.computeQaux(obs, ThA, softpolicy)
                    Qaux3[j, i] = oneHMM.latent_entr(obs)

            contour_LL_mesh = Qaux2 + Qaux3
            contour_LL_mesh = np.nan_to_num(contour_LL_mesh, nan = np.nanmean(contour_LL_mesh))

            self.uValue = uValue
            self.vValue = vValue
            self.contour_LL_mesh = contour_LL_mesh
            self.point_2d = projectionMat.dot((self.point_all - self.point_all[-1]).T).T
            self.projectionMat = projectionMat

    def plot_contour_LL(self):
        # project the trajectories onto the plane

        fig_contour, ax = plt.subplots(figsize=(6, 6))
        uValuemesh, vValuemesh = np.meshgrid(self.uValue, self.vValue)
        plt.contourf(uValuemesh, vValuemesh, self.contour_LL_mesh,
                     np.arange(np.min(self.contour_LL_mesh),
                               np.max(self.contour_LL_mesh), 50), cmap='jet')

        plt.plot(self.point_2d[:, 0], self.point_2d[:, 1], marker='.', color='b')  # projected trajectories
        # plt.plot(point_2d[-1, 0], point_2d[-1, 1], marker='*', color = 'g', markersize = 10)        # final point
        # plt.plot(true_2d[0], true_2d[1], marker='o', color = 'g')           # true

        # ax.grid()
        ax.set_title('Likelihood of observed data')
        plt.xlabel(r'$u \mathbf{\theta}$', fontsize=10)
        plt.ylabel(r'$v \mathbf{\theta}$', fontsize=10)
        # plt.clabel(cs3, inline=1, fontsize=10)
        plt.colorbar()
        plt.show()






