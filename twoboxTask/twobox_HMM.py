import numpy as np
from utils.boxtask_func import *

class HMMtwobox:
    def __init__(self, state_transition, obs_emission, latent_ini):
        self.state_transition = state_transition
        self.obs_emission = obs_emission
        self.latent_ini = latent_ini

        self.S = len(self.latent_ini)  # number of possible values of the hidden state (2 boxes)
        self.R = 2
        self.L = 3
        self.Ss = int(sqrt(self.S))   # # number of possible values of the hidden state (for 1 box)

        self.latent_dim = len(self.latent_ini)  # number of hidden state


    def _states(self, r, l):
        temp = torch.arange(self.Ss)
        #torch.reshape(torch.arange(self.Ss), [1, self.Ss])
        return (l * self.S * self.R + tensorsum_torch(temp * self.R * self.Ss, r * self.Ss + temp)).long().squeeze()

    def forward(self, obs):
        T = obs.shape[0]        # length of a sample sequence

        act = obs[:, 0]   # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]   # observable, two possible values: 0 : not have; 1: have
        loc = obs[:, 2]   # location, three possible values

        #alpha = np.zeros((self.S, T))  # initialize alpha value for each belief value
        alpha = []
        #alpha[:, 0] = self.latent_ini * self.obs_emission[act[0], self._states(rew[0], loc[0])]
        alpha.append(self.latent_ini * self.obs_emission[act[0], self._states(rew[0], loc[0])])

        for t in range(1, T):
            alpha.append((torch.matmul(alpha[-1].t(), self.state_transition[act[t - 1]][
                torch.meshgrid(self._states(rew[t - 1], loc[t-1]), self._states(rew[t], loc[t]))]) *
                          self.obs_emission[act[t], self._states(rew[t], loc[t])]).t())
        alpha = torch.stack(alpha).squeeze().t()

        #     alpha[:,  t] = np.dot(alpha[:, t - 1], self.state_transition[act[t - 1]][
        #         np.ix_(self._states(rew[t-1], loc[t-1]), self._states(rew[t], loc[t]))]) \
        #                    * self.obs_emission[act[t], self._states(rew[t], loc[t])]
        return alpha



    def backward(self, obs):
        """
        Backward path
        :param obs: a sequence of observations
        :return: predict future observations
        """
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        # beta = np.zeros((self.S, T))
        # beta[:, -1] = 1
        beta = []
        beta.append(torch.ones(self.latent_dim, 1))

        for t in reversed(range(T - 1)):
            beta.append(torch.matmul(self.state_transition[act[t]][torch.meshgrid(self._states(rew[t], loc[t+1]),
                                                                                  self._states(rew[t + 1], loc[t+1]))],
                                     beta[-1] * self.obs_emission[act[t + 1], self._states(rew[t + 1], loc[t+1])].unsqueeze(-1)))
        beta = beta[::-1]
        beta = torch.stack(beta).squeeze().t()

            # beta[:, t] = np.dot(self.state_transition[act[t]][np.ix_(self._states(rew[t], loc[t]),
            #                                                          self._states(rew[t+1], loc[t+1]))],
            #                     beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1], loc[t + 1])])

        return beta


    def forward_scale(self, obs):

        T = obs.shape[0]        # length of a sample sequence

        act = obs[:, 0]   # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]   # observable, two possible values: 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        # alpha = np.zeros((self.S, T))   # initialize alpha value for each belief value
        # scale = np.zeros(T)
        alpha = []
        scale = []

        # alpha[:, 0] = self.latent_ini * self.obs_emission[act[0], self._states(rew[0], loc[0])]
        # scale[0] = np.sum(alpha[:, 0])
        # alpha[:, 0] = alpha[:, 0] / scale[0]
        alpha_t = (self.latent_ini * self.obs_emission[act[0], self._states(rew[0], loc[0])]).unsqueeze(-1)
        scale_t = torch.sum(alpha_t)
        alpha_t = alpha_t / scale_t
        alpha.append(alpha_t)
        scale.append(scale_t)

        for t in range(1, T):
            alpha_t = (torch.matmul(alpha[-1].t(), self.state_transition[act[t - 1]][
                torch.meshgrid(self._states(rew[t - 1], loc[t-1]), self._states(rew[t], loc[t]))]) * self.obs_emission[
                           act[t], self._states(rew[t], loc[t])]).t()
            scale_t = torch.sum(alpha_t)
            alpha_t = alpha_t / scale_t

            alpha.append(alpha_t)
            scale.append(scale_t)

            # alpha[:,  t] = np.dot(alpha[:, t - 1], self.state_transition[act[t - 1]][
            #     np.ix_(self._states(rew[t-1], loc[t-1]), self._states(rew[t], loc[t]))]) \
            #                * self.obs_emission[act[t], self._states(rew[t], loc[t])]
            # scale[t] = np.sum(alpha[:, t])
            # alpha[:, t] = alpha[:, t] / scale[t]
        alpha = torch.stack(alpha).squeeze().t()
        scale = torch.stack(scale)

        return alpha, scale

    def backward_scale(self, obs, scale):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        beta = []
        beta.append(torch.ones(self.latent_dim, 1))
        # beta = np.zeros((self.S, T))
        # beta[:, T - 1] = 1
        #beta[:, T - 1] = beta[:, T - 1] / scale[T - 1]

        for t in reversed(range(T - 1)):
            beta_t = torch.matmul(self.state_transition[act[t]][torch.meshgrid(self._states(rew[t], loc[t]),
                                                                               self._states(rew[t + 1], loc[t+1]))],
                                  beta[-1] * self.obs_emission[act[t + 1], self._states(rew[t + 1], loc[t+1])].unsqueeze(-1))
            beta_t = beta_t / scale[t + 1]
            beta.append(beta_t)

            # beta[:, t] = np.dot(self.state_transition[act[t]][np.ix_(self._states(rew[t], loc[t]), self._states(rew[t + 1], loc[t + 1]))],
            #                     beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1], loc[t + 1])])
            # beta[:, t] = beta[:, t] / scale[t + 1]
        beta = beta[::-1]
        beta = torch.stack(beta).squeeze().t()

        return beta

    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        #gamma = gamma / np.sum(gamma, 0)
        gamma = gamma / torch.sum(gamma, 0)

        return gamma

    def compute_xi(self, alpha, beta, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        #xi = np.zeros((T - 1, self.S, self.S))
        xi = []

        for t in range(T - 1):
            xi_t = torch.diag(alpha[:, t]).matmul(
                self.state_transition[act[t]][torch.meshgrid(self._states(rew[t], loc[t]), self._states(rew[t + 1], loc[t+1]))]
                ).matmul(torch.diag(beta[:, t + 1] * self.obs_emission[act[t + 1], self._states(rew[t + 1],loc[t+1])]))
            xi_t = xi_t / torch.sum(xi_t)
            xi.append(xi_t)

            # xi[t, :, :] = np.diag(alpha[:, t]).dot(
            #     self.state_transition[act[t]][np.ix_(self._states(rew[t], loc[t]), self._states(rew[t + 1], loc[t + 1]))]
            # ).dot(np.diag(beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1], loc[t + 1])]))
            # xi[t, :, :] = xi[t, :, :]/np.sum(xi[t, :, :])
        xi = torch.stack(xi)
        return xi

    def latent_entr(self, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        # # Entropy of all path that leads to a certain state at t certain time
        # Hpath = np.zeros((self.S, T))
        # # P(state at time t-1 | state at time t, observations up to time t)
        # lat_cond = np.zeros((T - 1, self.S, self.S))
        Hpath = []
        lat_cond = []

        alpha_scaled, _ = self.forward_scale(obs)
        Hpath.append(torch.zeros(self.latent_dim).unsqueeze(-1))
        #Hpath[:, 0] = 0

        for t in range(1, T):
            lat_cond_t = torch.diag(alpha_scaled[:, t - 1]).matmul(self.state_transition[act[t - 1]
                                                                   ][torch.meshgrid(self._states(rew[t - 1], loc[t-1]),
                                                                                    self._states(rew[t],loc[t]))])

            lat_cond_t = lat_cond_t / (lat_cond_t.sum(dim=0) + 1 * (lat_cond_t.sum(dim=0) == 0))

            Hpath_t = Hpath[-1].t().matmul(lat_cond_t) - torch.sum(
                lat_cond_t * torch.log(lat_cond_t + 10 ** -13 * (lat_cond_t == 0)), axis=0)

            lat_cond.append(lat_cond_t)
            Hpath.append(Hpath_t.t())
            # lat_cond[t - 1] = np.diag(alpha_scaled[:, t - 1]).dot(
            #     self.state_transition[act[t - 1]][np.ix_(self._states(rew[t - 1], loc[t - 1]), self._states(rew[t], loc[t]))])
            # lat_cond[t - 1] = lat_cond[t - 1] / (
            # np.sum(lat_cond[t - 1], axis=0) + 1 * (np.sum(lat_cond[t - 1], axis=0) == 0))
            #
            # Hpath[:, t] = Hpath[:, t - 1].dot(lat_cond[t - 1]) - np.sum(
            #     lat_cond[t - 1] * np.log(lat_cond[t - 1] + 10 ** -13 * (lat_cond[t - 1] == 0)), axis=0)
        lat_cond = torch.stack(lat_cond)
        Hpath = torch.stack(Hpath).squeeze().t()

        # lat_ent = np.sum(Hpath[:, -1] * alpha_scaled[:, -1]) - np.sum(
        #     alpha_scaled[:, -1] * np.log(alpha_scaled[:, -1] + 10 ** -13 * (alpha_scaled[:, -1] == 0)))
        lat_ent = torch.sum(Hpath[:, -1] * alpha_scaled[:, -1]) - torch.sum(
            alpha_scaled[:, -1] * torch.log(alpha_scaled[:, -1] + 10 ** -13 * (alpha_scaled[:, -1] == 0)))


        return lat_ent

    def log_likelihood(self, obs, Anew, Bnew):
        CDLL = self.computeQaux(obs, Anew, Bnew)
        lat_ento = self.latent_entr(obs)

        return lat_ento + CDLL
        #return lat_ento


    def computeQaux(self, obs, Anew, Bnew):
        '''
        computer the Q auxillary funciton, the expected complete data likelihood
        :param obs: observation sequence, used to calculate alpha, beta, gamma, xi
        :param Anew: updated state_transition transition matrix
        :param Bnew: updated obs_emission emission matrix
        :return: Q auxilary value
        '''
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        # alpha = self.forward(obs)
        # beta = self.backward(obs)
        alpha, scale = self.forward_scale(obs)
        beta = self.backward_scale(obs, scale)

        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(alpha, beta, obs)
        #Qaux1 = np.sum(np.log(self.latent_ini) * gamma[:, 0])
        Qaux1 = torch.sum(torch.log(self.latent_ini) * gamma[:, 0])
        Qaux2 = 0
        Qaux3 = 0

        for t in range(T - 1):
            Qaux2 += torch.sum(torch.log(Anew[act[t]][torch.meshgrid(self._states(rew[t],loc[t]), self._states(rew[t + 1],loc[t+1]))] +
                                    10 ** -13 * (Anew[act[t]][torch.meshgrid(self._states(rew[t], loc[t]), self._states(rew[t + 1], loc[t+1]))] == 0))
                             * xi[t])


        for t in range(T):
            Qaux3 += torch.sum(torch.log(Bnew[act[t], self._states(rew[t], loc[t])] +
                                    10 ** -13 * ( Bnew[act[t], self._states(rew[t], loc[t])] == 0)) * gamma[:, t])



        # xi_delta = np.zeros((T, self.S, self.S))

        # for t in range(T - 1):
        #     # Qaux2 += np.sum(np.log(10 ** -13 + Anew[act[t]][
        #     #   np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]) * xi[t, :, :])
        #
        #     Qaux2 += np.sum(np.log(Anew[act[t]][np.ix_(self._states(rew[t],loc[t]), self._states(rew[t + 1],loc[t+1]))] +
        #                            10 ** -13 * (
        #                            Anew[act[t]][np.ix_(self._states(rew[t],loc[t]), self._states(rew[t + 1], loc[t+1]))] == 0))
        #                     * xi[t, :, :])

            # xi_delta[t, lat[t], lat[t+1]] = 1
            # Qaux2 += np.sum(np.log(Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))] +
            #                       1 * (Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))] == 0))
            #                * xi_delta[t])    #to check the code for computing the Qaux

        # for t in range(T):
        #     # Qaux3 += np.sum(np.log(10 ** -13 + Bnew[act[t], self._states(rew[t])]) * gamma[:, t])
        #
        #     Qaux3 += np.sum(np.log(Bnew[act[t], self._states(rew[t], loc[t])] +
        #                            10 ** -13 * (Bnew[act[t], self._states(rew[t], loc[t])] == 0)) * gamma[:, t])

        Qaux = 1 * (Qaux1 + Qaux2) + 1 * Qaux3
        # print alpha
        # print beta
        # print Qaux1, Qaux2, Qaux3

        return Qaux

    def computeQauxDE(self, obs, Anew, Bnew, Anewde, Bnewde):

        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        # alpha = self.forward(obs)
        # beta = self.backward(obs)
        alpha, scale = self.forward_scale(obs)
        beta = self.backward_scale(obs, scale)

        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(alpha, beta, obs)
        dQaux1 = np.sum(np.log(self.latent_ini) * gamma[:, 0])
        dQaux2 = 0
        dQaux3 = 0

        for t in range(T - 1):
            # Qaux2 += np.sum(np.log(10 ** -13 + Anew[act[t]][
            #   np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]) * xi[t, :, :])

            Aelement = Anew[act[t]][np.ix_(self._states(rew[t],loc[t]), self._states(rew[t + 1],loc[t+1]))]
            Aelement_prime = Aelement + 1 * (Aelement == 0)
            dQaux2_ins = Anewde[act[t]][np.ix_(self._states(rew[t],loc[t]), self._states(rew[t + 1],loc[t+1]))
                         ] / (Aelement_prime) * (Aelement != 0) * xi[t, :, :]
            dQaux2 += np.sum(dQaux2_ins)


            Belement = Bnew[act[t], self._states(rew[t], loc[t])]
            Belement_prime = Belement + 1 * (Belement == 0)
            dQaux3_ins = Bnewde[act[t], self._states(rew[t], loc[t])] / Belement_prime * \
                         (Belement != 0) * gamma[:, t]
            dQaux3 += np.sum(dQaux3_ins)

            dQaux = dQaux1 + dQaux2 + dQaux3

        return dQaux