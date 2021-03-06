'''
This incorporates the oneboxtask_ini and oneboxMDPsolver and oneboxGenerate into one file with oneboxMDP object

'''

from __future__ import division
from utils.boxtask_func import *
from utils.MDPclass import *
from oneboxTask.onebox_HMM import *
import torch
import numpy.matlib
from scipy.linalg import block_diag
from numpy.linalg import inv


# we need two different transition matrices, one for each of the following actions:
a0 = 0  # a0 = do nothing
pb = 1  # pb  = push button


class oneboxMDP:
    """
    model onebox problem, set up the transition matrices and reward based on the given parameters,
    and solve the MDP problem, return the optimal optpolicy
    """
    def __init__(self, discount, nq, nr, na, nl, parameters):
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.nl = nl
        self.n = (self.nq ** self.nl) * self.nr  # total number of states
        self.parameters = parameters
        self.ThA = []   #torch.empty(self.na, self.n, self.n)
        self.ThA_t = [] #torch.zeros(self.na, self.n, self.n)
        self.R = []     #torch.zeros(self.na, self.n, self.n)

    def setupMDP(self):
        """
        Based on the parameters, create transition matrices and reward function.
        Implement the codes in file 'oneboxtask_ini.py'
        :return:
                ThA: transition probability,
                     shape: (# of action) * (# of states, old state) * (# of states, new state)
                R: reward function
                   shape: (# of action) * (# of states, old state) * (# of states, new state)
        """

        food_missed = self.parameters['food_missed']   # available food dropped back into box after button press
        app_rate = self.parameters['app_rate']    # reward becomes available
        disapp_rate = self.parameters['disapp_rate']   # available food disappears
        food_consumed = self.parameters['food_consumed']    # food in mouth is consumed
        push_button_cost = self.parameters['push_button_cost']
        belief_diffusion = self.parameters['belief_diffusion']
        reward = 1

        # initialize probability distribution over states (belief and world)
        pr0 = torch.tensor([1, 0])  # (r=0, r=1) initially no food in mouth p(R=0)=1.
        ### pb0 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)
        pb0 = torch.cat((torch.tensor([1]), torch.zeros(self.nq - 1)))

        ph0 = tensorkronn(pr0, pb0)
        # kronecker product of these initial distributions
        # Note that this ordering makes the subsequent products easiest

        # setup single-variable transition matrices
        #Tr = np.array([[1, food_consumed], [0, 1 - food_consumed]])  # consume reward
        Tr = torch.tensor([[1, 0], [0, 1]]) + food_consumed * torch.tensor([[0, 1], [0, -1]])

        # belief transition matrix
        # Tb = beliefTransitionMatrix(app_rate, disapp_rate, nq, eta)
        # softened the belief transition matrix with 2-dimensional Gaussian distribution
        Tb = beliefTransitionMatrixGaussian(app_rate, disapp_rate, self.nq, belief_diffusion)

        # ACTION: do nothing
        #self.ThA_t[a0, :, :] = tensorkronn(Tr, Tb)
        self.ThA_t.append(tensorkronn(Tr, Tb))
        # kronecker product of these transition matrices

        # ACTION: push button
        ###bL = (np.array(range(self.nq)) + 1 / 2) / self.nq
        bL = (torch.tensor(range(self.nq)) + 1 / 2) / self.nq

        # Trb = torch.cat((torch.cat((1 - bL, torch.zeros(self.nq))).unsqueeze(-2),
        #                  torch.zeros(self.nq - 2, 2 * self.nq),
        #                  torch.cat((food_missed * bL, torch.zeros(self.nq))).unsqueeze(-2),
        #                  torch.cat(((1 - food_missed) * bL, 1 - bL)).unsqueeze(-2),
        #                  torch.zeros((self.nq - 2), 2 * self.nq),
        #                  torch.cat((torch.zeros( self.nq), bL)).unsqueeze(-2)), dim=0)

        Trb = torch.cat((torch.cat((1 - bL, torch.zeros(self.nq))).unsqueeze(-2),
                         torch.zeros(self.nq - 2, 2 * self.nq),
                         torch.cat((torch.zeros(self.nq), torch.zeros(self.nq))).unsqueeze(-2),
                         torch.cat((bL, 1 - bL)).unsqueeze(-2),
                         torch.zeros((self.nq - 2), 2 * self.nq),
                         torch.cat((torch.zeros(self.nq), bL)).unsqueeze(-2)), dim=0) + \
              torch.cat((torch.cat((torch.zeros(self.nq), torch.zeros(self.nq))).unsqueeze(-2),
                         torch.zeros(self.nq - 2, 2 * self.nq),
                         torch.cat((bL, torch.zeros(self.nq))).unsqueeze(-2),
                         torch.cat(( - bL, torch.zeros(self.nq))).unsqueeze(-2),
                         torch.zeros((self.nq - 2), 2 * self.nq),
                         torch.cat((torch.zeros(self.nq), torch.zeros(self.nq))).unsqueeze(-2)), dim=0) * food_missed



        self.ThA_t.append(Trb.matmul(self.ThA_t[a0]))  # first wait for an usual time, then pb

        # Trb = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
        #                       np.zeros((self.nq - 2, 2 * self.nq)),
        #                       np.array([np.insert([np.zeros(self.nq)], 0, food_missed * bL)]),
        #                       np.array([np.insert([(1 - food_missed) * bL], self.nq, 1 - bL)]),
        #                       np.zeros(((self.nq - 2), 2 * self.nq)),
        #                       np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)
        # self.ThA[pb, :, :] = Trb.dot(self.ThA[a0, :, :])  # first wait for an usual time, then pb
        #self.ThA[pb, :, :] = Trb

        # Reward_h = tensorsumm(np.array([[0, reward]]), np.zeros((1, self.nq)))
        # Reward_a = - np.array([0, push_button_cost])
        Reward_h = tensorsumm_torch(torch.tensor([[0, 1]]) * reward,
                                    torch.zeros(1, self.nq)).squeeze()
        Reward_a = - torch.tensor([0, 1]) * push_button_cost

        [R1, R2, R3] = torch.meshgrid(Reward_a, Reward_h, Reward_h)
        Reward = R1 + R3
        self.R = Reward

        self.ThA_t = torch.stack(self.ThA_t)
        for i in range(self.na):
            self.ThA.append(torch.t(self.ThA_t[i]))
        self.ThA = torch.stack(self.ThA)

    def solveMDP_op(self, epsilon = 10**-6, niterations = 10000):
        """
        Solve the MDP problem with value iteration
        Implement the codes in file 'oneboxMDPsolver.py'

        :param discount: temporal discount
        :param epsilon: stopping criterion used in value iteration
        :param niterations: value iteration
        :return:
                Q: Q value function
                   shape: (# of actions) * (# of states)
                optpolicy: the optimal optpolicy based on the maximum Q value
                        shape: # of states, take integer values indicating the action
                softpolicy: probability of choosing each action
                            shape: (# of actions) * (# of states)
        """

        # value iteration
        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations)
        vi.run()
        self.Q = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value

        #self.optpolicy = np.array(vi.optpolicy)
        self.optpolicy = vi.policy

        ## optpolicy iteration
        #latent_ini = mdp.ValueIteration(self.ThA, self.R, self.discount, epsilon, niterations)
        #latent_ini.run()
        #self.Q = self._QfromV(latent_ini)
        #self.optpolicy = np.array(latent_ini.optpolicy)


    def solveMDP_sfm(self, epsilon = 10**-6, niterations = 10000, initial_value=0):
        """
        Solve the MDP problem with value iteration
        Implement the codes in file 'oneboxMDPsolver.py'

        :param discount: temporal discount
        :param epsilon: stopping criterion used in value iteration
        :param niterations: value iteration
        :return:
                Q: Q value function
                   shape: (# of actions) * (# of states)
                optpolicy: softmax optpolicy
        """

        policy_temperature = self.parameters['policy_temperature']
        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        vi.run(policy_temperature)
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        #self.softpolicy = np.array(vi.softpolicy)
        self.softpolicy = vi.softpolicy

        #return  vi.V


    def _QfromV(self, ValueIteration):
        #Q = torch.zeros(ValueIteration.state_transition, ValueIteration.S) # Q is of shape: num of actions * num of states
        Q = []
        for a in range(ValueIteration.state_transition):
            Q.append(ValueIteration.R[a] + ValueIteration.discount * \
                                            ValueIteration.P[a].matmul(ValueIteration.V))
        Q = torch.stack(Q)
        return Q

# class onebox_generate(oneboxMDP):
#     """
#     This class generates the data based on the object oneboxMDP. The parameters, and thus the transition matrices and
#     the rewrd function, are shared for the oneboxMDP and this data generator class.
#     """
#     def __init__(self, discount, nq, nr, na, parameters, parameters_exp,
#                  sampleTime, sampleNum):
#         oneboxMDP.__init__(self, discount, nq, nr, na, parameters)
#
#         self.parameters_exp = parameters_exp
#         self.sampleNum = sampleNum
#         self.sampleTime = sampleTime
#
#         self.action = np.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
#         self.hybrid = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
#         # Here it is the joint state of reward and belief
#         self.belief = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
#         self.reward = np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
#         self.trueState = np.zeros((self.sampleNum, self.sampleTime))
#
#         self.setupMDP()
#         self.solveMDP_op()
#         self.solveMDP_sfm()
#
#     def dataGenerate_op(self, belief_ini, rew_ini):
#         """
#         This is a function that belongs to the oneboxMDP class. In the oneboxGenerate.py file, this function is implemented
#         as a separate class, since at that time, the oneboxMDP class was not defined.
#         In this file, all the functions are implemented under a single class.
#
#         :return: the obseravations
#         """
#
#         beta = self.parameters[0]  # available food dropped back into box after button press
#         gamma = self.parameters[1]  # reward becomes available
#         epsilon = self.parameters[2]  # available food disappears
#         rho = self.parameters[3]  # food in mouth is consumed
#
#         gamma_e = self.parameters_exp[0]
#         epsilon_e = self.parameters_exp[1]
#
#
#         for i in range(self.sampleNum):
#             for t in range(self.sampleTime):
#                 if t == 0:
#                     self.trueState[i, t] = np.random.binomial(1, gamma_e)
#
#                     self.reward[i, t], self.belief[i, t] = rew_ini, belief_ini
#                     self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]    # This is for one box only
#                     self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
#                             # action is based on optimal optpolicy
#                 else:
#                     if self.action[i, t-1] != pb:
#                         stattemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :], size = 1)
#                         self.hybrid[i, t] = np.argmax(stattemp)
#                         self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)
#                         self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
#
#                         if self.trueState[i, t - 1] == 0:
#                             self.trueState[i, t] = np.random.binomial(1, gamma_e)
#                         else:
#                             self.trueState[i, t] = 1 - np.random.binomial(1, epsilon_e)
#                     else:
#                         #### for pb action, wait for usual time and then pb  #############
#                         if self.trueState[i, t - 1] == 0:
#                             self.trueState[i, t - 1] = np.random.binomial(1, gamma_e)
#                         else:
#                             self.trueState[i, t - 1] = 1 - np.random.binomial(1, epsilon_e)
#                         #### for pb action, wait for usual time and then pb  #############
#
#                         if self.trueState[i, t - 1] == 0:
#                             self.trueState[i, t] = self.trueState[i, t-1]
#                             self.belief[i, t] = 0
#                             if self.reward[i, t-1]==0:
#                                 self.reward[i, t] = 0
#                             else:
#                                 self.reward[i, t] = np.random.binomial(1, 1 - rho)
#                         else:
#                             self.trueState[i, t] = np.random.binomial(1, beta)
#
#                             if self.trueState[i, t] == 1: # is dropped back after bp
#                                 self.belief[i, t] = self.nq - 1
#                                 if self.reward[i, t - 1] == 0:
#                                     self.reward[i, t] = 0
#                                 else:
#                                     self.reward[i, t] = np.random.binomial(1, 1 - rho)
#                             else: # not dropped back
#                                 self.belief[i, t] = 0
#                                 self.reward[i, t] = 1  # give some reward
#
#                             #self.trueState[i, t] = 0  # if true world is one, pb resets it to zero
#                             #self.belief[i, t] = 0
#                             #self.reward[i, t] = 1  # give some reward
#
#                         self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
#                         self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
#
#
#     def dataGenerate_sfm(self, belief_ini, rew_ini):
#         """
#         This is a function that belongs to the oneboxMDP class. In the oneboxGenerate.py file, this function is implemented
#         as a separate class, since at that time, the oneboxMDP class was not defined.
#         In this file, all the functions are implemented under a single class.
#
#         :return: the observations
#         """
#
#         beta = self.parameters[0]  # available food dropped back into box after button press
#         gamma = self.parameters[1]  # reward becomes available
#         epsilon = self.parameters[2]  # available food disappears
#         rho = self.parameters[3]  # food in mouth is consumed
#
#         gamma_e = self.parameters_exp[0]
#         epsilon_e = self.parameters_exp[1]
#
#         for i in range(self.sampleNum):
#             for t in range(self.sampleTime):
#                 if t == 0:
#                     self.trueState[i, t] = np.random.binomial(1, gamma_e)
#
#                     self.reward[i, t], self.belief[i, t] = rew_ini, belief_ini
#                     self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]    # This is for one box only
#                     self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
#                             # action is based on softmax optpolicy
#                 else:
#                     if self.action[i, t-1] != pb:
#                         stattemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :], size = 1)
#                         self.hybrid[i, t] = np.argmax(stattemp)
#                             # not pressing button, hybrid state evolves probabilistically
#                         self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)
#                         self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
#
#                         if self.trueState[i, t - 1] == 0:
#                             self.trueState[i, t] = np.random.binomial(1, gamma_e)
#                         else:
#                             self.trueState[i, t] = 1 - np.random.binomial(1, epsilon_e)
#                     else:   # press button
#                         #### for pb action, wait for usual time and then pb  #############
#                         if self.trueState[i, t - 1] == 0:
#                             self.trueState[i, t - 1] = np.random.binomial(1, gamma_e)
#                         else:
#                             self.trueState[i, t - 1] = 1 - np.random.binomial(1, epsilon_e)
#                         #### for pb action, wait for usual time and then pb  #############
#
#                         if self.trueState[i, t - 1] == 0:
#                             self.trueState[i, t] = self.trueState[i, t-1]
#                             self.belief[i, t] = 0
#                             if self.reward[i, t-1]==0:
#                                 self.reward[i, t] = 0
#                             else:
#                                 self.reward[i, t] = np.random.binomial(1, 1 - rho)
#                                         # With probability 1- rho, reward is 1, not consumed
#                                         # with probability rho, reward is 0, consumed
#                         # if true world is one, pb resets it to zero with probability
#                         else:
#                             self.trueState[i, t] = np.random.binomial(1, beta)
#
#                             if self.trueState[i, t] == 1: # is dropped back after bp
#                                 self.belief[i, t] = self.nq - 1
#                                 if self.reward[i, t - 1] == 0:
#                                     self.reward[i, t] = 0
#                                 else:
#                                     self.reward[i, t] = np.random.binomial(1, 1 - rho)
#                             else: # not dropped back
#                                 self.belief[i, t] = 0
#                                 self.reward[i, t] = 1  # give some reward
#
#                         self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
#                         self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
#
#
#
#     def _chooseAction(self, pvec):
#         # Generate action according to multinomial distribution
#         stattemp = np.random.multinomial(1, pvec)
#         return np.argmax(stattemp)
#
#
#
#
# class oneboxMDPder(oneboxMDP):
#     """
#     derivative of different functions with respect to the parameters
#     """
#
#     def __init__(self, discount, nq, nr, na, parameters):
#         oneboxMDP.__init__(self, discount, nq, nr, na, parameters)
#
#         self.setupMDP()
#         self.solveMDP_sfm()
#
#     def transitionDerivative(self):
#         """
#         calcualte the derivative of the transition probability with respect to the parameters
#         :return: derivatives
#         """
#         beta = self.parameters[0]   # available food dropped back into box after button press
#         gamma = self.parameters[1]    # reward becomes available
#         epsilon = self.parameters[2]   # available food disappears
#         rho = self.parameters[3]    # food in mouth is consumed
#         pushButtonCost = self.parameters[4]
#         Reward = 1
#
#         dThAdepsilon = np.zeros(self.ThA.shape)
#         dThAdgamma = np.zeros(self.ThA.shape)
#         dThAdbeta = np.zeros(self.ThA.shape)
#         dThAdrho = np.zeros(self.ThA.shape)
#
#         Tr = np.array([[1, rho], [0, 1 - rho]])  # consume reward
#         Tb = beliefTransitionMatrixGaussian(gamma, epsilon, self.nq, sigmaTb)
#         bL = (np.array(range(self.nq)) + 1 / 2) / self.nq
#         Trb = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
#                               np.zeros((self.nq - 2, 2 * self.nq)),
#                               np.array([np.insert([np.zeros(self.nq)], 0, beta * bL)]),
#                               np.array([np.insert([(1 - beta) * bL], self.nq, 1 - bL)]),
#                               np.zeros(((self.nq - 2), 2 * self.nq)),
#                               np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)
#
#
#         ##########################################################################
#         ######## first allow some usual time, then button press ##################
#         ##########################################################################
#         # derivative of the belief-reward transition dynamic with respect to gamma and epsilon
#         dTbdgamma, dTbdepsilon = beliefTransitionMatrixGaussianDerivative(gamma, epsilon, self.nq, sigmaTb)
#         dThAdgamma[a0, :, :] = kronn(Tr, dTbdgamma)
#         dThAdgamma[pb, :, :] = np.dot(Trb, dThAdgamma[a0, :, :])
#         for i in range(self.na):
#             dThAdgamma[i, :, :] = dThAdgamma[i, :, :].T
#
#         dThAdepsilon[a0, :, :] = kronn(Tr, dTbdepsilon)
#         dThAdepsilon[pb, :, :] = np.dot(Trb, dThAdepsilon[a0, :, :])
#         for i in range(self.na):
#             dThAdepsilon[i, :, :] = dThAdepsilon[i, :, :].T
#
#         # derivative with respect to rho (appears only in Tr)
#         dTrdrho = np.array([[0, 1], [0, -1]])
#         dThAdrho[a0, :, :] = kronn(dTrdrho, Tb)
#         dThAdrho[pb, :, :] = Trb.dot(dThAdrho[a0, :, :])
#         for i in range(self.na):
#             dThAdrho[i, :, :] = dThAdrho[i, :, :].T
#
#         # derivative with respect to beta (appears only in Trb with button presing)
#         dTrbdbeta = np.concatenate((np.zeros((self.nq - 1, 2 * self.nq)),
#                               np.array([np.insert([np.zeros(self.nq)], 0, 1 * bL)]),
#                               np.array([np.insert([(-1) * bL], self.nq, np.zeros(self.nq))]),
#                               np.zeros(((self.nq - 1), 2 * self.nq))), axis=0)
#         dThAdbeta[a0, :, :] = np.zeros(self.ThA[a0, :, :].shape)
#         dThAdbeta[pb, :, :] = dTrbdbeta.dot(self.ThA[a0, :, :].T).T
#
#         '''
#         ##########################################################################
#         ######## only button press ##################
#         ##########################################################################
#         # derivative of the belief-reward transition dynamic with respect to gamma and epsilon
#         dTbdgamma, dTbdepsilon = beliefTransitionMatrixGaussianDerivative(gamma, epsilon, self.nq, sigmaTb)
#         dThAdgamma[a0, :, :] = kronn(Tr, dTbdgamma)
#         dThAdgamma[pb, :, :] = np.zeros(self.ThA[pb, :, :].shape)
#         #dThAdgamma[pb, :, :] = np.dot(Trb, dThAdgamma[a0, :, :])
#         for i in range(self.na):
#             dThAdgamma[i, :, :] = dThAdgamma[i, :, :].T
#
#         dThAdepsilon[a0, :, :] = kronn(Tr, dTbdepsilon)
#         dThAdepsilon[pb, :, :] = np.zeros(self.ThA[pb, :, :].shape)
#         #dThAdepsilon[pb, :, :] = np.dot(Trb, dThAdepsilon[a0, :, :])
#         for i in range(self.na):
#             dThAdepsilon[i, :, :] = dThAdepsilon[i, :, :].T
#
#         # derivative with respect to rho (appears only in Tr)
#         dTrdrho = np.array([[0, 1], [0, -1]])
#         dThAdrho[a0, :, :] = kronn(dTrdrho, Tb)
#         dThAdrho[pb, :, :] = np.zeros(self.ThA[pb, :, :].shape)
#         #dThAdrho[pb, :, :] = Trb.dot(dThAdrho[a0, :, :])
#         for i in range(self.na):
#             dThAdrho[i, :, :] = dThAdrho[i, :, :].T
#
#         # derivative with respect to beta (appears only in Trb with button presing)
#         dTrbdbeta = np.concatenate((np.zeros((self.nq - 1, 2 * self.nq)),
#                                     np.array([np.insert([np.zeros(self.nq)], 0, 1 * bL)]),
#                                     np.array([np.insert([(-1) * bL], self.nq, np.zeros(self.nq))]),
#                                     np.zeros(((self.nq - 1), 2 * self.nq))), axis=0)
#         dThAdbeta[a0, :, :] = np.zeros(self.ThA[a0, :, :].shape)
#         dThAdbeta[pb, :, :] = dTrbdbeta.T
#         #dThAdbeta[pb, :, :] = dTrbdbeta.dot(self.ThA[a0, :, :].T).T
#                             # When calculating this derivative, the transpose on ThA has already been made
#                             # thus need to transpose back first, and then multiply; finally transpose back again
#         '''
#
#         return dThAdbeta, dThAdgamma, dThAdepsilon, dThAdrho
#
#     def _dpolicydQ(self):
#         """
#         derivative of the softmax optpolicy with respect to Q value
#         :return: dpdQ, shape: latent_dim * latent_dim
#         """
#         latent_dim = self.Qsfm.size
#         Qexp = np.exp(self.Qsfm / temperatureQ)  # shapa: na * n
#
#         Qexpstack = np.reshape(Qexp, latent_dim)
#         Qexp_diag = np.diag(Qexpstack)
#
#         Qexp_suminv = np.ones(self.n) / np.sum(Qexp, axis = 0)
#         dpdQ = 1 / temperatureQ * np.diag(np.tile(Qexp_suminv, self.na)).dot(Qexp_diag) - \
#                1 / temperatureQ * Qexp_diag.dot(np.tile(np.diag(Qexp_suminv ** 2),
#                                                     (self.na, self.na))).dot(Qexp_diag)
#
#         return dpdQ
#
#     def dQdpara(self):
#         '''
#         This function is does not give the exact gradient, since the Q value
#         is not convergencet, and Q iteration equation is not exactly equal
#         :return:
#         '''
#         latent_dim = self.Qsfm.size
#         Qstack = np.reshape(self.Qsfm, latent_dim)
#         softpolicystack = np.reshape(self.softpolicy, latent_dim)
#
#         Rstack = np.reshape(self.R[:, 0, :], latent_dim)
#
#         softpolicydiag = np.diag(softpolicystack)    # optpolicy, diagonal matrix
#         ThAstack = np.reshape(np.tile(self.ThA, (1, self.na)), (self.n * self.na, self.n * self.na))
#                    # stack transition probability
#         Qdiag = np.diag(Qstack)
#         dpdQ = self._dpolicydQ()
#
#         # gradient of Q with respect to r
#         constant_r = np.concatenate((np.zeros(self.n), - 1 * np.ones(self.n)), axis = 0)  # k2, vector
#         dQdpara_r = inv(np.eye(latent_dim) - self.discount * ThAstack.
#                         dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_r)
#
#
#         dThAdbeta, dThAdgamma, dThAdepsilon, dThAdrho = self.transitionDerivative()
#
#         # gradient of Q with respect to beta
#         dThAblock_beta = block_diag(dThAdbeta[a0], dThAdbeta[pb])  # used to calculate k1
#         constant_beta = dThAblock_beta.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
#                                            dot(Qstack * softpolicystack) + Rstack)
#         dQdpara_beta = inv(np.eye(latent_dim) - self.discount * ThAstack.
#                         dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_beta)
#
#         # gradient of Q with respect to gamma
#         dThAblock_gamma = block_diag(dThAdgamma[a0], dThAdgamma[pb])  # used to calculate k1
#         constant_gamma = dThAblock_gamma.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
#                                            dot(Qstack * softpolicystack) + Rstack)
#         dQdpara_gamma = inv(np.eye(latent_dim) - self.discount * ThAstack.
#                         dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_gamma)
#
#
#         # gradient of Q with respect to epsilon
#         dThAblock_epsilon = block_diag(dThAdepsilon[a0], dThAdepsilon[pb])  # used to calculate k1
#         constant_epsilon = dThAblock_epsilon.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
#                                            dot(Qstack * softpolicystack) + Rstack)
#         dQdpara_epsilon = inv(np.eye(latent_dim) - self.discount * ThAstack.
#                         dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_epsilon)
#
#         # gradient of Q with respect to rho
#         dThAblock_rho = block_diag(dThAdrho[a0], dThAdrho[pb])  # used to calculate k1
#         constant_rho = dThAblock_rho.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
#                                            dot(Qstack * softpolicystack) + Rstack)
#         dQdpara_rho = inv(np.eye(latent_dim) - self.discount * ThAstack.
#                         dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_rho)
#
#         return dQdpara_beta, dQdpara_gamma, dQdpara_epsilon, dQdpara_rho, dQdpara_r
#
#     def dQdpara_sim(self):
#         delta = 10 ** -6
#
#         beta = self.parameters[0]   # available food dropped back into box after button press
#         gamma = self.parameters[1]    # reward becomes available
#         epsilon = self.parameters[2]   # available food disappears
#         rho = self.parameters[3]    # food in mouth is consumed
#         pushButtonCost = self.parameters[4]
#
#         parameters1 = [beta, gamma, epsilon, rho, pushButtonCost+delta]
#         one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
#         one1.setupMDP()
#         one1.solveMDP_sfm()
#         dQdpara_r = (one1.Qsfm - self.Qsfm) / delta
#         dQdpara_r = np.reshape(dQdpara_r, dQdpara_r.size)
#
#         parameters1 = [beta+delta, gamma, epsilon, rho, pushButtonCost]
#         one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
#         one1.setupMDP()
#         one1.solveMDP_sfm()
#         dQdpara_beta = (one1.Qsfm - self.Qsfm) / delta
#         dQdpara_beta = np.reshape(dQdpara_beta, dQdpara_beta.size)
#
#         parameters1 = [beta, gamma+delta, epsilon, rho, pushButtonCost]
#         one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
#         one1.setupMDP()
#         one1.solveMDP_sfm()
#         dQdpara_gamma = (one1.Qsfm - self.Qsfm) / delta
#         dQdpara_gamma = np.reshape(dQdpara_gamma, dQdpara_gamma.size)
#
#         parameters1 = [beta, gamma, epsilon + delta, rho, pushButtonCost]
#         one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
#         one1.setupMDP()
#         one1.solveMDP_sfm()
#         dQdpara_epsilon = (one1.Qsfm - self.Qsfm) / delta
#         dQdpara_epsilon = np.reshape(dQdpara_epsilon, dQdpara_epsilon.size)
#
#         parameters1 = [beta, gamma, epsilon, rho + delta, pushButtonCost]
#         one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
#         one1.setupMDP()
#         one1.solveMDP_sfm()
#         dQdpara_rho = (one1.Qsfm - self.Qsfm) / delta
#         dQdpara_rho = np.reshape(dQdpara_rho, dQdpara_rho.size)
#
#         return dQdpara_beta, dQdpara_gamma, dQdpara_epsilon, dQdpara_rho, dQdpara_r
#
#     def dpdpara(self):
#
#         # Derivative of the softmax optpolicy with respect to the parameters
#
#         dpdQ = self._dpolicydQ()
#         #dQdpara_beta, dQdpara_gamma, dQdpara_epsilon, dQdpara_rho, dQdpara_r= self.dQdpara()
#         dQdpara_beta, dQdpara_gamma, dQdpara_epsilon, dQdpara_rho, dQdpara_r = self.dQdpara_sim()
#
#         dpdpara_r = dpdQ.dot(dQdpara_r)
#         dpdpara_beta = dpdQ.dot(dQdpara_beta)
#         dpdpara_gamma = dpdQ.dot(dQdpara_gamma)
#         dpdpara_epsilon = dpdQ.dot(dQdpara_epsilon)
#         dpdpara_rho = dpdQ.dot(dQdpara_rho)
#
#         dpdpara_r = np.reshape(dpdpara_r, (self.na, self.n))
#         dpdpara_beta = np.reshape(dpdpara_beta, (self.na, self.n))
#         dpdpara_gamma = np.reshape(dpdpara_gamma, (self.na, self.n))
#         dpdpara_epsilon = np.reshape(dpdpara_epsilon, (self.na, self.n))
#         dpdpara_rho = np.reshape(dpdpara_rho, (self.na, self.n))
#
#         return dpdpara_beta, dpdpara_gamma, dpdpara_epsilon, dpdpara_rho, dpdpara_r
#
#     def dQauxdpara_sim(self, obs, para_new):
#
#         # Derivative of the Q auxiliary function with respect to the parameters
#         # Calculated numerically by perturbing the parameters
#
#         latent_ini = np.ones(self.nq) / self.nq
#         oneboxHMM = HMMonebox(self.ThA, self.softpolicy, latent_ini)  #old parameter to calculate alpha, beta, gamma, xi
#
#         delta = 10 ** -6
#
#         beta = para_new[0]   # available food dropped back into box after button press
#         gamma = para_new[1]    # reward becomes available
#         epsilon = para_new[2]   # available food disappears
#         rho = para_new[3]    # food in mouth is consumed
#         pushButtonCost = para_new[4]
#
#         onebox_new = oneboxMDP(self.discount, self.nq, self.nr, self.na, para_new)
#         onebox_new.setupMDP()
#         onebox_new.solveMDP_sfm()
#         Qaux = oneboxHMM.computeQaux(obs, onebox_new.ThA, onebox_new.softpolicy)
#
#         # para1 = [beta + delta, gamma, epsilon, rho, pushButtonCost]
#         # one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
#         # one1.setupMDP()
#         # one1.solveMDP_sfm()
#         # Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
#         # dQauxdpara_beta = (Qaux1 - Qaux) / delta
#
#         para1 = [beta, gamma + delta, epsilon, rho, pushButtonCost]
#         one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
#         one1.setupMDP()
#         one1.solveMDP_sfm()
#         Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
#         dQauxdpara_gamma = (Qaux1 - Qaux) / delta
#
#         para1 = [beta, gamma, epsilon + delta, rho, pushButtonCost]
#         one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
#         one1.setupMDP()
#         one1.solveMDP_sfm()
#         Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
#         dQauxdpara_epsilon = (Qaux1 - Qaux) / delta
#
#         # para1 = [beta, gamma, epsilon, rho + delta, pushButtonCost]
#         # one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
#         # one1.setupMDP()
#         # one1.solveMDP_sfm()
#         # Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
#         # dQauxdpara_rho = (Qaux1 - Qaux) / delta
#         #
#         para1 = [beta, gamma, epsilon, rho, pushButtonCost + delta]
#         one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
#         one1.setupMDP()
#         one1.solveMDP_sfm()
#         Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
#         dQauxdpara_r = (Qaux1 - Qaux) / delta
#
#         # return dQauxdpara_beta, dQauxdpara_gamma, dQauxdpara_epsilon, dQauxdpara_rho, dQauxdpara_r
#         return 0, dQauxdpara_gamma, dQauxdpara_epsilon, 0, dQauxdpara_r
#
#     def dQauxdpara(self, obs, para_new):
#
#         # Derivative of the Q auxiliary function with respect to the parameters
#         # Calculated analytically
#
#         latent_ini = np.ones(self.nq) / self.nq
#         oneboxHMM = HMMonebox(self.ThA, self.softpolicy, latent_ini)
#
#         onebox_newde = oneboxMDPder(self.discount, self.nq, self.nr, self.na, para_new)
#
#         dQauxdpara_beta = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
#                                                   onebox_newde.transitionDerivative()[0], onebox_newde.dpdpara()[0])
#         dQauxdpara_gamma = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
#                                                    onebox_newde.transitionDerivative()[1], onebox_newde.dpdpara()[1])
#         dQauxdpara_epsilon = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
#                                                      onebox_newde.transitionDerivative()[2], onebox_newde.dpdpara()[2])
#         dQauxdpara_rho = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
#                                                  onebox_newde.transitionDerivative()[3], onebox_newde.dpdpara()[3])
#         dQauxdpara_r = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
#                                                np.zeros(onebox_newde.ThA.shape), onebox_newde.dpdpara()[4])
#
#         return dQauxdpara_beta, dQauxdpara_gamma, dQauxdpara_epsilon, dQauxdpara_rho, dQauxdpara_r
#
#
#


