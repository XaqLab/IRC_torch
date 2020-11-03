from oneboxTask.onebox import *


class oneboxMDPder(oneboxMDP):
    """
    Derivatives of different functions with respect to the parameters
    """
    def __init__(self, discount, nq, nr, na, parameters, initial_valueV = 0):
        oneboxMDP.__init__(self, discount, nq, nr, na, parameters)

        self.setupMDP()
        self.solveMDP_sfm(initial_value = initial_valueV)

    def dloglikelihhod_dpara_sim(self, obs):
        L = len(self.parameters)
        pi = np.ones(self.nq) / self.nq 
        # Numcol = np.rint(self.parameters[7]).astype(int) # number of colors
        # Ncol = Numcol - 1  # number value: 0 top Numcol-1
        # 

        oneboxHMM = HMMonebox(self.ThA, self.softpolicy, pi)
        # log_likelihood =  oneboxHMM.computeQaux(obs, self.ThA, self.softpolicy) + \
        #                   oneboxHMM.latent_entr(obs)
        log_likelihood = oneboxHMM.log_likelihood(obs, self.ThA, self.softpolicy)

        perturb = 10 ** -6

        dloglikelihhod_dpara = np.zeros(L)


        for i in range(L):
            para_perturb = self.parameters.copy()
            para_key = list(self.parameters.items())[i][0]
            para_perturb[para_key] += perturb
            #para_perturb[i] = para_perturb[i] + perturb

            onebox_perturb = oneboxMDP(self.discount, self.nq, self.nr, self.na, para_perturb)
            onebox_perturb.setupMDP()
            onebox_perturb.solveMDP_sfm()
            ThA_perturb = onebox_perturb.ThA
            policy_perturb = onebox_perturb.softpolicy
            oneboxHMM_perturb = HMMonebox(ThA_perturb, policy_perturb, pi)

            # log_likelihood_perturb = oneboxHMM_perturb.computeQaux(obs, ThA_perturb, policy_perturb) + \
            #                          oneboxHMM_perturb.latent_entr(obs)
            log_likelihood_perturb = oneboxHMM_perturb.log_likelihood(obs, ThA_perturb, policy_perturb)

            dloglikelihhod_dpara[i] = (log_likelihood_perturb - log_likelihood) / perturb

        return dloglikelihhod_dpara



