#import numpy as np
import torch

class HMMonebox:
    def __init__(self, state_transition, obs_emission, latent_ini):
        self.state_transition = state_transition
        self.obs_emission = obs_emission
        self.latent_ini = latent_ini
        self.latent_dim = len(self.latent_ini)  # number of hidden state

    def _states(self, r):
        return torch.arange(self.latent_dim * r, self.latent_dim * (r + 1))

    def forward(self, obs):
        """
        the forward path, used to estimate the state at a given time given all the observations
        with both filtering and smoothing
        :param obs: a sequence of observations
        :return: smoothed probability of state at a certain time
        """

        T = obs.shape[0]        # length of a sample sequence

        act = obs[:, 0]   # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]   # observable, two possible values: 0 : not have; 1: have

        alpha = []
        alpha.append((self.latent_ini * self.obs_emission[act[0], self._states(rew[0])]).unsqueeze(-1))
        for t in range(1, T):
            alpha.append( (torch.matmul(alpha[-1].t(), self.state_transition[act[t - 1]][
                torch.meshgrid(self._states(rew[t-1]), self._states(rew[t]))]) *
                         self.obs_emission[act[t], self._states(rew[t])]).t())
        alpha = torch.stack(alpha).squeeze().t()


        # alpha = torch.zeros(self.latent_dim, T)
        # # initialize alpha value for each belief value
        # alpha[:, 0] = self.latent_ini * self.obs_emission[act[0], self._states(rew[0])]
        #
        # for t in range(1, T):
        #     alpha[:,  t] = torch.matmul(alpha[:, t - 1], self.state_transition[act[t - 1]][
        #         torch.meshgrid(self._states(rew[t-1]), self._states(rew[t]))]) * self.obs_emission[act[t], self._states(rew[t])]

        return alpha


    def forward_scale(self, obs):

        T = obs.shape[0]        # length of a sample sequence

        act = obs[:, 0]   # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]   # observable, two possible values: 0 : not have; 1: have

        #alpha = torch.zeros(self.latent_dim, T)   # initialize alpha value for each belief value
        #scale = torch.zeros(T)
        alpha = []
        scale = []

        alpha_t = (self.latent_ini * self.obs_emission[act[0], self._states(rew[0])]).unsqueeze(-1)
        scale_t = torch.sum(alpha_t)
        alpha_t = alpha_t/ scale_t
        alpha.append(alpha_t)
        scale.append(scale_t)

        for t in range(1, T):
            alpha_t = (torch.matmul(alpha[-1].t(), self.state_transition[act[t - 1]][
                torch.meshgrid(self._states(rew[t - 1]), self._states(rew[t]))]) * self.obs_emission[
                              act[t], self._states(rew[t])]).t()
            scale_t = torch.sum(alpha_t)
            alpha_t = alpha_t / scale_t

            alpha.append(alpha_t)
            scale.append(scale_t)

        alpha = torch.stack(alpha).squeeze().t()
        scale = torch.stack(scale)

        # alpha[:, 0] = self.latent_ini * self.obs_emission[act[0], self._states(rew[0])]
        # scale[0] = torch.sum(alpha[:, 0])
        # alpha[:, 0] = alpha[:, 0] / scale[0]
        # for t in range(1, T):
        #     alpha[:,  t] = torch.matmul(alpha[:, t - 1], self.state_transition[act[t - 1]][
        #         torch.meshgrid(self._states(rew[t-1]), self._states(rew[t]))]) * self.obs_emission[act[t], self._states(rew[t])]
        #     scale[t] = torch.sum(alpha[:, t])
        #     alpha[:, t] = alpha[:, t] / scale[t]

        return alpha, scale

    def backward(self, obs):
        """
        Backward path
        :param obs: a sequence of observations
        :return: predict future observations
        """
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have

        beta = []
        beta.append(torch.ones(self.latent_dim, 1))
        for t in reversed(range(T - 1)):
            beta.append(torch.matmul(self.state_transition[act[t]][torch.meshgrid(self._states(rew[t]),
                                                                                  self._states(rew[t + 1]))],
                                     beta[-1] * self.obs_emission[act[t + 1], self._states(rew[t + 1])].unsqueeze(-1)))
        beta = beta[::-1]
        beta = torch.stack(beta).squeeze().t()

        # beta = torch.zeros(self.latent_dim, T)
        # beta[:, -1] = 1
        # for t in reversed(range(T - 1)):
        #     beta[:, t] = torch.matmul(self.state_transition[act[t]][torch.meshgrid(self._states(rew[t]), self._states(rew[t + 1]))],
        #                         beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1])])

        return beta

    def backward_scale(self, obs, scale):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have

        beta = []
        beta.append(torch.ones(self.latent_dim, 1))

        #beta = torch.zeros(self.latent_dim, T)
        #beta[:, T - 1] = 1

        for t in reversed(range(T - 1)):
            beta_t = torch.matmul(self.state_transition[act[t]][torch.meshgrid(self._states(rew[t]),
                                                                                  self._states(rew[t + 1]))],
                                     beta[-1] * self.obs_emission[act[t + 1], self._states(rew[t + 1])].unsqueeze(-1))
            beta_t = beta_t / scale[t + 1]
            beta.append(beta_t)

        beta = beta[::-1]
        beta = torch.stack(beta).squeeze().t()


        # for t in reversed(range(T - 1)):
        #     beta[:, t] = torch.matmul(self.state_transition[act[t]][torch.meshgrid(self._states(rew[t]),
        #                                                                            self._states(rew[t + 1]))],
        #                         beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1])])
        #     beta[:, t] = beta[:, t] / scale[t + 1]

        return beta

    def observation_prob(self, obs):
        """ P( entire observation sequence | state_transition, obs_emission, latent_ini ) """
        return torch.sum(self.forward(obs)[:, -1])

    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        #gamma = gamma    # dim = state # x T
        gamma = gamma / torch.sum(gamma, 0)

        return gamma

    def compute_xi(self, alpha, beta, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have

        #xi = torch.zeros(T - 1, self.latent_dim, self.latent_dim)
        xi = []

        for t in range(T - 1):
            xi_t = torch.diag(alpha[:, t]).matmul(self.state_transition[act[t]][torch.meshgrid(self._states(rew[t]), self._states(rew[t + 1]))]
                                                   ).matmul(torch.diag(beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1])]))
            xi_t = xi_t / torch.sum(xi_t)
            xi.append(xi_t)

            # xi[t, :, :] = torch.diag(alpha[:, t]).matmul(self.state_transition[act[t]][torch.meshgrid(self._states(rew[t]), self._states(rew[t + 1]))]
            #                                        ).matmul(torch.diag(beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1])]))
            # xi[t, :, :] = xi[t, :, :]/torch.sum(xi[t, :, :])
        xi = torch.stack(xi)
        return xi

    # def likelihood(self, lat, obs, Anew, Bnew):
    #     '''
    #     computer the likelihood given the ground truth latent state
    #     '''
    #     T = obs.shape[0]  # length of a sample sequence
    #
    #     act = obs[:, 0]  # 0: doing nothing; 1: press button
    #     rew = obs[:, 1]  # 0 : not have; 1: have
    #
    #     likeh1 = np.log(self.latent_ini[lat[0]])
    #     likeh2 = 0
    #     likeh3 = 0
    #
    #     for t in range(T - 1):
    #         likeh2 += np.log(Anew[act[t], self.latent_dim * rew[t] + lat[t], self.latent_dim * rew[t + 1] + lat[t + 1]] +
    #                          10 ** -13 * (Anew[act[t], self.latent_dim * rew[t] + lat[t], self.latent_dim * rew[t + 1] + lat[t + 1]] == 0))
    #
    #
    #     for t in range(T):
    #         likeh3 += np.log(Bnew[act[t], self.latent_dim * rew[t] + lat[t]] +
    #                          10 ** -13 * (Bnew[act[t], self.latent_dim * rew[t] + lat[t]] == 0))
    #
    #     likeh = 1 * (likeh1 + likeh2) + 1 * likeh3
    #
    #     return likeh


    # def realxi(self, lat, obs):
    #     # delta function of latent variable when the ground truth is known
    #     T = obs.shape[0]  # length of a sample sequence
    #
    #     xi_delta = torch.zeros(T, self.latent_dim, self.latent_dim)
    #
    #     for t in range(T - 1):
    #         xi_delta[t, lat[t], lat[t+1]] = 1
    #
    #     return xi_delta

    def latent_entr(self, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have

        # Entropy of all path that leads to a certain state at t certain time
        #Hpath = torch.zeros(self.latent_dim, T)
        # P(state at time t-1 | state at time t, observations up to time t)
        #lat_cond = torch.zeros(T - 1, self.latent_dim, self.latent_dim)

        Hpath = []
        lat_cond = []

        alpha_scaled, _ = self.forward_scale(obs)
        Hpath.append(torch.zeros(self.latent_dim).unsqueeze(-1))
        #Hpath[:, 0] = 0

        for t in range(1, T):
            lat_cond_t = torch.diag(alpha_scaled[:, t - 1]).matmul(self.state_transition[act[t - 1]
                                                                        ][torch.meshgrid(self._states(rew[t - 1]), self._states(rew[t]))])

            lat_cond_t = lat_cond_t / (lat_cond_t.sum(dim = 0) + 1 * (lat_cond_t.sum(dim = 0) == 0))

            Hpath_t = Hpath[-1].t().matmul(lat_cond_t) - torch.sum(lat_cond_t * torch.log(lat_cond_t + 10 ** -13 * (lat_cond_t==0)), axis = 0)

            lat_cond.append(lat_cond_t)
            Hpath.append(Hpath_t.t())
        lat_cond = torch.stack(lat_cond)
        Hpath = torch.stack(Hpath).squeeze().t()

            # lat_cond[t - 1] = torch.diag(alpha_scaled[:, t - 1]).matmul(self.state_transition[act[t - 1]
            #                                                             ][torch.meshgrid(self._states(rew[t - 1]), self._states(rew[t]))])
            # lat_cond[t - 1] = lat_cond[t - 1] / (torch.sum(lat_cond[t - 1], axis = 0)
            #                                      + 1 * (torch.sum(lat_cond[t - 1], axis = 0) == 0))
            #
            # Hpath[:, t] = Hpath[:, t - 1].matmul(lat_cond[t - 1]) - torch.sum(lat_cond[t - 1] * torch.log(lat_cond[t - 1] + 10 ** -13 * (lat_cond[t - 1]==0)), axis = 0)
            #
        lat_ent = torch.sum(Hpath[:, -1] * alpha_scaled[:, -1]) - torch.sum(
            alpha_scaled[:, -1] * torch.log(alpha_scaled[:, -1] + 10 ** -13 * (alpha_scaled[:, -1] == 0)))

        return lat_ent

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

        alpha, scale = self.forward_scale(obs)
        beta = self.backward_scale(obs, scale)

        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(alpha, beta, obs)
        Qaux1 = torch.sum(torch.log(self.latent_ini) * gamma[:, 0])
        Qaux2 = 0
        Qaux3 = 0


        for t in range(T - 1):
            Qaux2 += torch.sum(torch.log(Anew[act[t]][torch.meshgrid(self._states(rew[t]), self._states(rew[t + 1]))] +
                                    10 ** -13 * (Anew[act[t]][torch.meshgrid(self._states(rew[t]), self._states(rew[t + 1]))] == 0))
                             * xi[t])


        for t in range(T):
            Qaux3 += torch.sum(torch.log(Bnew[act[t], self._states(rew[t])] +
                                    10 ** -13 * ( Bnew[act[t], self._states(rew[t])] == 0)) * gamma[:, t])

        Qaux = 1 * (Qaux1 + Qaux2) + 1 * Qaux3


        return Qaux

    def log_likelihood(self, obs, Anew, Bnew):
        CDLL = self.computeQaux(obs, Anew, Bnew)
        lat_ento = self.latent_entr(obs)

        return lat_ento + CDLL