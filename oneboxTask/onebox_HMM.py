import numpy as np


class HMMonebox:
    def __init__(self, state_transition, obs_emission, latent_ini):
        self.state_transition = state_transition
        self.obs_emission = obs_emission
        self.latent_ini = latent_ini
        self.latent_dim = len(self.latent_ini)  # number of hidden state

    def _states(self, r):
        return range(self.latent_dim * r, self.latent_dim * (r + 1))

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

        alpha = np.zeros((self.latent_dim, T))   # initialize alpha value for each belief value
        alpha[:, 0] = self.latent_ini * self.obs_emission[act[0], self._states(rew[0])]

        for t in range(1, T):
            alpha[:,  t] = np.dot(alpha[:, t - 1], self.state_transition[act[t - 1]][
                np.ix_(self._states(rew[t-1]), self._states(rew[t]))]) * self.obs_emission[act[t], self._states(rew[t])]
        return alpha


    def forward_scale(self, obs):

        T = obs.shape[0]        # length of a sample sequence

        act = obs[:, 0]   # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]   # observable, two possible values: 0 : not have; 1: have

        alpha = np.zeros((self.latent_dim, T))   # initialize alpha value for each belief value
        scale = np.zeros(T)

        alpha[:, 0] = self.latent_ini * self.obs_emission[act[0], self._states(rew[0])]
        scale[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / scale[0]

        for t in range(1, T):
            alpha[:,  t] = np.dot(alpha[:, t - 1], self.state_transition[act[t - 1]][
                np.ix_(self._states(rew[t-1]), self._states(rew[t]))]) * self.obs_emission[act[t], self._states(rew[t])]
            scale[t] = np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] / scale[t]

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

        beta = np.zeros((self.latent_dim, T))
        beta[:, -1] = 1
        for t in reversed(range(T - 1)):
            beta[:, t] = np.dot(self.state_transition[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))],
                                beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1])])

        return beta

    def backward_scale(self, obs, scale):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have

        beta = np.zeros((self.latent_dim, T))
        beta[:, T - 1] = 1

        for t in reversed(range(T - 1)):
            beta[:, t] = np.dot(self.state_transition[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))],
                                beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1])])
            beta[:, t] = beta[:, t] / scale[t + 1]

        return beta

    def observation_prob(self, obs):
        """ P( entire observation sequence | state_transition, obs_emission, latent_ini ) """
        return np.sum(self.forward(obs)[:, -1])

    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, 0)

        return gamma

    def compute_xi(self, alpha, beta, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have

        xi = np.zeros((T - 1, self.latent_dim, self.latent_dim))

        for t in range(T - 1):
            xi[t, :, :] = np.diag(alpha[:, t]).dot(self.state_transition[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]
                                                   ).dot(np.diag(beta[:, t+1] * self.obs_emission[act[t + 1], self._states(rew[t + 1])]))
            xi[t, :, :] = xi[t, :, :]/np.sum(xi[t, :, :])

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


    def realxi(self, lat, obs):
        # delta function of latent variable when the ground truth is known
        T = obs.shape[0]  # length of a sample sequence

        xi_delta = np.zeros((T, self.latent_dim, self.latent_dim))

        for t in range(T - 1):
            xi_delta[t, lat[t], lat[t+1]] = 1

        return xi_delta

    def latent_entr(self, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have

        # Entropy of all path that leads to a certain state at t certain time
        Hpath = np.zeros((self.latent_dim, T))
        # P(state at time t-1 | state at time t, observations up to time t)
        lat_cond = np.zeros((T - 1, self.latent_dim, self.latent_dim))

        alpha_scaled, _ = self.forward_scale(obs)
        Hpath[:, 0] = 0

        for t in range(1, T):
            lat_cond[t - 1] = np.diag(alpha_scaled[:, t - 1]).dot(self.state_transition[act[t - 1]][np.ix_(self._states(rew[t - 1]), self._states(rew[t]))])
            lat_cond[t - 1] = lat_cond[t - 1] / (np.sum(lat_cond[t - 1], axis = 0) + 1 * (np.sum(lat_cond[t - 1], axis = 0) == 0))

            Hpath[:, t] = Hpath[:, t - 1].dot(lat_cond[t - 1]) - np.sum(lat_cond[t - 1] * np.log(lat_cond[t - 1] + 10 ** -13 * (lat_cond[t - 1]==0)), axis = 0)

        lat_ent = np.sum(Hpath[:, -1] * alpha_scaled[:, -1]) - np.sum(alpha_scaled[:, -1] * np.log(alpha_scaled[:, -1] + 10 ** -13 * (alpha_scaled[:, -1] == 0)))

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
        Qaux1 = np.sum(np.log(self.latent_ini) * gamma[:, 0])
        Qaux2 = 0
        Qaux3 = 0


        for t in range(T - 1):
            Qaux2 += np.sum(np.log(Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))] +
                                    10 ** -13 * (Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))] == 0))
                             * xi[t, :, :])


        for t in range(T):
            Qaux3 += np.sum(np.log(Bnew[act[t], self._states(rew[t])] +
                                    10 ** -13 * ( Bnew[act[t], self._states(rew[t])] == 0)) * gamma[:, t])

        Qaux = 1 * (Qaux1 + Qaux2) + 1 * Qaux3


        return Qaux

    def log_likelihood(self, obs, Anew, Bnew):
        return self.latent_entr(obs) + self.computeQaux(obs, Anew, Bnew)