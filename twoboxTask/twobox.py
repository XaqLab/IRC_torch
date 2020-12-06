from utils.boxtask_func import *
from utils.MDPclass import *
#from twoboxTask.twobox_HMM import *
import collections

import numpy.matlib
from scipy.linalg import block_diag
from numpy.linalg import inv

# we need five different transition matrices, one for each of the following actions:
a0 = 0    # a0 = do nothing
g0 = 1    # g0 = go to location 0
g1 = 2    # g1 = go toward box 1 (via location 0 if from 2)
g2 = 3    # g2 = go toward box 2 (via location 0 if from 1)
pb = 4    # pb  = push button

class twoboxMDP:
    def __init__(self, discount, nq, nr, na, nl, parameters):
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.nl = nl   # number of locations
        self.n = (self.nq ** 2) * self.nr * self.nl   # total number of states
        self.parameters = parameters  # [beta, app_rate1, epsilon, food_consumed]

        self.ThA = []  # torch.empty(self.na, self.n, self.n)
        self.ThA_t = []  # torch.zeros(self.na, self.n, self.n)
        self.R = []  # torch.zeros(self.na, self.n, self.n)

        # self.ThA = np.zeros((self.na, self.n, self.n))
        # self.R = np.zeros((self.na, self.n, self.n))

    def setupMDP(self):
        """
        Based on the parameters, create transition matrices and reward function.
        Implement the codes in file 'twoboxtask_ini.py'
        :return:
                ThA: transition probability,
                    shape: (# of action) * (# of states, old state) * (# of states, new state)
                R: reward function
                    shape: (# of action) * (# of states, old state) * (# of states, new state)
        """

        food_missed = self.parameters['food_missed']  # available food dropped back into box after button press
        app_rate1 = self.parameters['app_rate1']  # reward becomes available
        disapp_rate1 = self.parameters['disapp_rate1']  # available food disappears
        app_rate2 = self.parameters['app_rate2']  # reward becomes available
        disapp_rate2 = self.parameters['disapp_rate2']  # available food disappears
        trip_prob = self.parameters['trip_prob'] # animal trips, doesn't go to target location
        direct_prob = self.parameters['direct_prob']  # animal goes right to target, skipping location 0
        food_consumed = self.parameters['food_consumed']  # food in mouth is consumed
        push_button_cost = self.parameters['push_button_cost']
        travel_cost = self.parameters['travel_cost']
        grooming_reward = self.parameters['grooming_reward']
        belief_diffusion = self.parameters['belief_diffusion']
        policy_temperature = self.parameters['policy_temperature']
        reward = 1

        # initialize probability distribution over states (belief and world)
        pr0 = torch.tensor([1, 0])  # (r=0, r=1) initially no food in mouth p(R=0)=1.
        pl0 = torch.tensor([1, 0, 0])  # (l=0, l=1, l=2) initial location is at L=0
        #pb10 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)
        #pb20 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)
        pb10 = torch.cat((torch.tensor([1]), torch.zeros(self.nq - 1)))
        pb20 = torch.cat((torch.tensor([1]), torch.zeros(self.nq - 1)))


        #ph0 = kronn(pl0, pb10, pr0, pb20)
        ph0 = tensorkronn(pl0, pb10, pr0, pb20)
        # kronecker product of these initial distributions
        # Note that this ordering makes the subsequent products easiest

        # setup single-variable transition matrices
        #Tr = np.array([[1, food_consumed], [0, 1 - food_consumed]])  # consume reward
        Tr = torch.tensor([[1, 0], [0, 1]]) + food_consumed * torch.tensor([[0, 1], [0, -1]])

        # Tb1 = beliefTransitionMatrix(app_rate11, disapp_rate1, nq, eta)
        # Tb2 = beliefTransitionMatrix(app_rate12, disapp_rate2, nq, eta)
        Tb1 = beliefTransitionMatrixGaussian(app_rate1, disapp_rate1, self.nq, belief_diffusion)
        Tb2 = beliefTransitionMatrixGaussian(app_rate2, disapp_rate2, self.nq, belief_diffusion)

        # ACTION: do nothing
        #self.ThA[a0, :, :] = kronn(np.identity(self.nl), Tb1, Tr, Tb2)
        self.ThA_t.append(tensorkronn(torch.eye(self.nl), Tb1, Tr, Tb2))
        # kronecker product of these transition matrices

        # ACTION: go to location 0/1/2
        # Tl0 = np.array(
        #     [[1, 1 - trip_prob, 1 - trip_prob], [0, trip_prob, 0], [0, 0, trip_prob]])  # go to loc 0 (with error of trip_prob)
        # Tl1 = np.array([[trip_prob, 0, 1 - trip_prob - direct_prob], [1 - trip_prob, 1, direct_prob],
        #                 [0, 0, trip_prob]])  # go to box 1 (with error of trip_prob)
        # Tl2 = np.array([[trip_prob, 1 - trip_prob - direct_prob, 0], [0, trip_prob, 0],
        #                 [1 - trip_prob, direct_prob, 1]])  # go to box 2 (with error of trip_prob)
        # self.ThA[g0, :, :] = kronn(Tl0, Tb1, Tr, Tb2)
        # self.ThA[g1, :, :] = kronn(Tl1, Tb1, Tr, Tb2)
        # self.ThA[g2, :, :] = kronn(Tl2, Tb1, Tr, Tb2)
        Tl0 = torch.tensor(
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]]) + torch.tensor(
            [[0, -1, -1], [0, 1, 0], [0, 0, 1]]) * trip_prob
        # go to loc 0 (with error of trip_prob)
        Tl1 = torch.tensor([[0, 0, 1], [1, 1, 0], [0, 0, 0]]) + \
              torch.tensor([[1, 0, -1], [-1, 0, 0],[0, 0, 1]]) * trip_prob + \
              torch.tensor([[0, 0, -1], [0, 0, 1],[0, 0, 0]]) * direct_prob
        # go to box 1 (with error of trip_prob)
        Tl2 = torch.tensor([[0, 1, 0], [0, 0, 0], [1, 0, 1]])   + \
              torch.tensor([[1, -1, 0], [0, 1, 0], [-1, 0, 0]]) * trip_prob  + \
              torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) * direct_prob
        # go to box 2 (with error of trip_prob)
        self.ThA_t.append(tensorkronn(Tl0, Tb1, Tr, Tb2))
        self.ThA_t.append(tensorkronn(Tl1, Tb1, Tr, Tb2))
        self.ThA_t.append(tensorkronn(Tl2, Tb1, Tr, Tb2))

        # ACTION: push button
        #bL = (np.array(range(self.nq)) + 1 / 2) / self.nq
        bL = (torch.tensor(range(self.nq)) + 1 / 2) / self.nq

        # Trb2 = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
        #                        np.zeros((self.nq - 2, 2 * self.nq)),
        #                        np.array([np.insert([np.zeros(self.nq)], 0, food_missed * bL)]),
        #                        np.array([np.insert([(1 - food_missed) * bL], self.nq, 1 - bL)]),
        #                        np.zeros(((self.nq - 2), 2 * self.nq)),
        #                        np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)
        Trb2 = torch.cat((torch.cat((1 - bL, torch.zeros(self.nq))).unsqueeze(-2),
                         torch.zeros(self.nq - 2, 2 * self.nq),
                         torch.cat((torch.zeros(self.nq), torch.zeros(self.nq))).unsqueeze(-2),
                         torch.cat((bL, 1 - bL)).unsqueeze(-2),
                         torch.zeros((self.nq - 2), 2 * self.nq),
                         torch.cat((torch.zeros(self.nq), bL)).unsqueeze(-2)), dim=0) + \
              torch.cat((torch.cat((torch.zeros(self.nq), torch.zeros(self.nq))).unsqueeze(-2),
                         torch.zeros(self.nq - 2, 2 * self.nq),
                         torch.cat((bL, torch.zeros(self.nq))).unsqueeze(-2),
                         torch.cat((- bL, torch.zeros(self.nq))).unsqueeze(-2),
                         torch.zeros((self.nq - 2), 2 * self.nq),
                         torch.cat((torch.zeros(self.nq), torch.zeros(self.nq))).unsqueeze(-2)), dim=0) * food_missed

        idx = torch.arange(self.nq * self.nr)
        idx1 = idx.reshape(self.nr, self.nq).t().reshape(1, -1).squeeze()
        Tb1r = Trb2[torch.meshgrid(idx1, idx1)]
        #Tb1r = torch.from_numpy(reversekron(Trb2.detach().numpy(), np.array([2, self.nq])))

        # Th = block_diag(np.identity(self.nq * self.nr * self.nq),
        #                 np.kron(Tb1r, np.identity(self.nq)),
        #                 np.kron(np.identity(self.nq), Trb2))
        Th = torch.block_diag(
            torch.eye(self.nq * self.nr * self.nq),
            tensorkronn(Tb1r, torch.eye(self.nq)),
            tensorkronn(torch.eye(self.nq), Trb2))
        self.ThA_t.append(Th.matmul(self.ThA_t[a0]))
        # wait for usual time, then pres button
        # self.ThA[pb, :, :] = Th

        # Reward_h = tensorsumm(np.array([[grooming_reward, 0, 0]]),
        #                       np.zeros((1, self.nq)),
        #                       np.array([[0, reward]]),
        #                       np.zeros((1, self.nq)))
        # Reward_a = - np.array([0, travel_cost, travel_cost, travel_cost, push_button_cost])
        Reward_h = tensorsumm_torch(torch.tensor([[1, 0, 0]]) * grooming_reward,
                              torch.zeros(1, self.nq),
                              torch.tensor([[0, 1]]) * reward,
                              torch.zeros(1, self.nq)).squeeze()
        Reward_a = - torch.tensor([0, 1, 1, 1, 0]) * travel_cost -\
                   torch.tensor([0, 0, 0, 0, 1]) * push_button_cost


        [R1, R2, R3] = torch.meshgrid(Reward_a, Reward_h, Reward_h)
        Reward = R1 + R3
        # R = Reward[:, 0, :].T
        self.R = Reward

        self.ThA_t = torch.stack(self.ThA_t)
        for i in range(self.na):
            self.ThA.append(torch.t(self.ThA_t[i]))
        self.ThA = torch.stack(self.ThA)

        # for i in range(self.na):
        #     self.ThA[i, :, :] = self.ThA[i, :, :].T

    def solveMDP_op(self, epsilon = 0.001, niterations = 10000, initial_value=0):
        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        # optimal optpolicy, stopping criterion changed to "converged Qvalue"
        vi.run()
        self.Q = self._QfromV(vi)
        self.optpolicy = np.array(vi.policy)
        self.Vop = vi.V

    def solveMDP_sfm(self, epsilon = 0.001, niterations = 10000, initial_value=0):

        policy_temperature = self.parameters['policy_temperature']

        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        vi.run(policy_temperature)
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        #self.softpolicy = np.array(vi.softpolicy)
        #self.Vsfm = vi.V

        self.softpolicy = vi.softpolicy

        #return vi.V

    def _QfromV(self, ValueIteration):
        # Q = torch.zeros(ValueIteration.state_transition, ValueIteration.S) # Q is of shape: num of actions * num of states
        Q = []

        for a in range(ValueIteration.A):
            Q.append(ValueIteration.R[a] + ValueIteration.discount * \
                     ValueIteration.P[a].matmul(ValueIteration.V))
        Q = torch.stack(Q)
        return Q

    # def _QfromV(self, ValueIteration):
    #     Q = np.zeros((ValueIteration.state_transition, ValueIteration.S)) # Q is of shape: na * n
    #     for a in range(ValueIteration.state_transition):
    #         Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
    #                                         ValueIteration.P[a].dot(ValueIteration.V)
    #     return Q