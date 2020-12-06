from oneboxTask.onebox import *
import pickle
import os
from datetime import datetime
import numpy as np

def onebox_data(parameters, parameters_exp, sample_length, sample_number, nq, nr, na, nl, discount,
                policy, belief_ini, rew_ini):

    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')   # current time used to set file name

    food_missed = parameters['food_missed']  # available food dropped back into box after button press
    app_rate = parameters['app_rate']  # reward becomes available
    disapp_rate = parameters['disapp_rate']  # available food disappears
    food_consumed = parameters['food_consumed']  # food in mouth is consumed
    push_button_cost = parameters['push_button_cost']
    belief_diffusion = parameters['belief_diffusion']
    policy_temperature = parameters['policy_temperature']

    app_rate_experiment = parameters_exp['app_rate_experiment']
    disapp_rate_experiment = parameters_exp['disapp_rate_experiment']

    """ Gnerate data"""
    print("\nGenerating data...")
    T = sample_length
    N = sample_number
    oneboxdata = onebox_generate(discount, nq, nr, na, nl, parameters, parameters_exp, T, N)
    oneboxdata.data_generate(policy, belief_ini, rew_ini)  # softmax optpolicy

    belief = oneboxdata.belief
    action = oneboxdata.action
    reward = oneboxdata.reward
    trueState = oneboxdata.trueState

    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data
    obsN = torch.cat([action, reward], dim = -1)  # includes the action and the observable states
    latN = torch.cat([belief])
    truthN = torch.cat([trueState])
    dataN = torch.cat([obsN, latN, truthN], dim = -1)

    path = os.getcwd()
    # write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}
    data_output = open(path + '/Data/' + datestring + '_dataN_onebox' + '.pkl', 'wb')
    pickle.dump(data_dict, data_output)
    data_output.close()

    # write all model parameters to file
    para_dict = {'discount': discount,
                 'nq': nq,
                 'nr': nr,
                 'na': na,
                 'nl': nl,
                 'food_missed': food_missed,
                 'app_rate': app_rate,
                 'disapp_rate': disapp_rate,
                 'food_consumed': food_consumed,
                 'push_button_cost': push_button_cost,
                 'app_rate_experiment': app_rate_experiment,
                 'disapp_rate_experiment': disapp_rate_experiment,
                 'belief_diffusion': belief_diffusion,
                 'policy_temperature': policy_temperature
                 }

    # create a file that saves the parameter dictionary using pickle
    para_output = open(path + '/Data/' + datestring + '_para_onebox' + '.pkl', 'wb')
    pickle.dump(para_dict, para_output)
    para_output.close()

    print('Data stored in files')

    return obsN, latN, truthN, datestring


class onebox_generate(oneboxMDP):
    """
    This class generates the data based on the object oneboxMDP. The parameters, and thus the transition matrices and
    the rewrd function, are shared for the oneboxMDP and this data generator class.
    """
    def __init__(self, discount, nq, nr, na, nl, parameters, parameters_exp,
                 sampleTime, sampleNum):
        oneboxMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.parameters_exp = parameters_exp
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime

        self.action = [] #np.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
        self.hybrid = [] #np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.belief = [] #np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = [] #np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
        self.trueState = [] #np.zeros((self.sampleNum, self.sampleTime))

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()

    def data_generate(self, policy, belief_ini, rew_ini):
        """
        This is a function that belongs to the oneboxMDP class. In the oneboxGenerate.py file, this function is implemented
        as a separate class, since at that time, the oneboxMDP class was not defined.
        In this file, all the functions are implemented under a single class.

        :return: the obseravations
        """

        food_missed = self.parameters['food_missed']  # available food dropped back into box after button press
        food_consumed = self.parameters['food_consumed']  # food in mouth is consumed

        app_rate_experiment = self.parameters_exp['app_rate_experiment']
        disapp_rate_experiment = self.parameters_exp['disapp_rate_experiment']


        for i in range(self.sampleNum):
            self.action.append([])
            self.hybrid.append([])
            self.belief.append([])
            self.reward.append([])
            self.trueState.append([])

            if rew_ini == 'rand':
                rew_ini = torch.randint(self.nr, (1,)).long()
            if belief_ini == 'rand':
                belief_ini = torch.randint(self.nq, (1,)).long()

            for t in range(self.sampleTime):
                if t == 0:
                    self.trueState[i].append(torch.bernoulli(app_rate_experiment).long())
                    self.reward[i].append(rew_ini)
                    self.belief[i].append(belief_ini)
                    self.hybrid[i].append(self.reward[i][-1] * self.nq + self.belief[i][-1])
                    # self.trueState[i, t] = np.random.binomial(1, app_rate_experiment)
                    # self.reward[i, t], self.belief[i, t] = rew_ini, belief_ini
                    # self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]  # This is for one box only
                    if policy == 'opt':
                        #self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
                        self.action[i].append(self.optpolicy[self.hybrid[i][-1]])
                    elif policy == 'sfm':
                        #self.action[i, t] = self._choose_action(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
                        self.action[i].append(self._choose_action(self.softpolicy.t()[self.hybrid[i][-1]].squeeze()))
                else:
                    if self.action[i][-1] != pb:
                        # stattemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :],
                        #                                  size=1)

                        self.hybrid[i].append(torch.multinomial(self.ThA[self.action[i][-1],
                                                                self.hybrid[i][-1], :].squeeze(), 1, replacement=True))

                        self.reward[i].append(torch.floor_divide(self.hybrid[i][-1], self.nq))
                        self.belief[i].append(torch.fmod(self.hybrid[i][-1], self.nq))

                        #self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)

                        #self.hybrid[i, t] = np.argmax(stattemp)
                        #self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)
                        #self.action[i, t] = self.optpolicy[self.hybrid[i, t]]

                        if self.trueState[i][- 1] == 0:
                            #self.trueState[i, t] = np.random.binomial(1, app_rate_experiment)
                            self.trueState[i].append(torch.bernoulli(app_rate_experiment).long())
                        else:
                            #self.trueState[i, t] = 1 - np.random.binomial(1, disapp_rate_experiment)
                            self.trueState[i].append(1 - torch.bernoulli(disapp_rate_experiment).long())
                    else:
                        # for pb action, wait for usual time and then pb  #############
                        if self.trueState[i][-1] == 0:
                            statetemp = torch.bernoulli(app_rate_experiment).long()
                        else:
                            statetemp = 1 - torch.bernoulli(disapp_rate_experiment).long()
                        # for pb action, wait for usual time and then pb  #############

                        if statetemp == 0:
                            self.trueState[i].append(statetemp)
                            self.belief[i].append(torch.LongTensor([0]))
                            if self.reward[i][-1] == 0:
                                self.reward[i].append(torch.LongTensor([0]))
                            else:
                                self.reward[i].append(torch.bernoulli(1 - food_consumed).long())
                        else:
                            #self.trueState[i, t] = np.random.binomial(1, food_missed.detach().numpy())
                            self.trueState[i].append(torch.bernoulli(food_missed).long())

                            if self.trueState[i][-1] == 1:  # is dropped back after bp
                                self.belief[i].append(torch.LongTensor([self.nq - 1]).long())
                                if self.reward[i][- 1] == 0:
                                    self.reward[i].append(torch.LongTensor([0]).long())
                                else:
                                    self.reward[i].append(torch.bernoulli(1 - food_consumed).long())
                                    #self.reward[i, t] = np.random.binomial(1, 1 - food_consumed.detach().numpy())
                            else:  # not dropped back
                                self.belief[i].append(torch.LongTensor([0]).long())
                                self.reward[i].append(torch.LongTensor([1]).long())
                                #self.reward[i, t] = 1  # give some reward

                        self.hybrid[i].append(self.reward[i][-1] * self.nq + self.belief[i][-1])
                    if policy == 'opt':
                        #self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
                        self.action[i].append(self.optpolicy[self.hybrid[i][-1]])
                    elif policy == 'sfm':
                        #self.action[i, t] = self._choose_action(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
                        self.action[i].append(self._choose_action(self.softpolicy.t()[self.hybrid[i][-1]].squeeze()))

            self.action[-1] = torch.stack(self.action[-1])
            self.hybrid[-1] = torch.stack(self.hybrid[-1] )
            self.belief[-1] = torch.stack(self.belief[-1])
            self.reward[-1] = torch.stack(self.reward[-1])
            self.trueState[-1] = torch.stack(self.trueState[-1])

        self.action = torch.stack(self.action)
        self.hybrid = torch.stack(self.hybrid)
        self.belief = torch.stack(self.belief)
        self.reward = torch.stack(self.reward)
        self.trueState = torch.stack(self.trueState)



    # def data_generate_op(self, belief_ini = 0, rew_ini = 0):
    #     """
    #     This is a function that belongs to the oneboxMDP class. In the oneboxGenerate.py file, this function is implemented
    #     as a separate class, since at that time, the oneboxMDP class was not defined.
    #     In this file, all the functions are implemented under a single class.
    #
    #     :return: the obseravations
    #     """
    #
    #     food_missed = self.parameters['food_missed']  # available food dropped back into box after button press
    #     food_consumed = self.parameters['food_consumed']  # food in mouth is consumed
    #
    #     app_rate_experiment = self.parameters_exp['app_rate_experiment']
    #     disapp_rate_experiment = self.parameters_exp['disapp_rate_experiment']
    #
    #     for i in range(self.sampleNum):
    #         for t in range(self.sampleTime):
    #             if t == 0:
    #                 self.trueState[i, t] = np.random.binomial(1, app_rate_experiment)
    #                 self.reward[i, t], self.belief[i, t] = rew_ini, belief_ini
    #                 self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]    # This is for one box only
    #                 self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
    #                         # action is based on optimal optpolicy
    #             else:
    #                 if self.action[i, t-1] != pb:
    #                     stattemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :], size = 1)
    #                     self.hybrid[i, t] = np.argmax(stattemp)
    #                     self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)
    #                     self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
    #
    #                     if self.trueState[i, t - 1] == 0:
    #                         self.trueState[i, t] = np.random.binomial(1, app_rate_experiment)
    #                     else:
    #                         self.trueState[i, t] = 1 - np.random.binomial(1, disapp_rate_experiment)
    #                 else:
    #                     # for pb action, wait for usual time and then pb  #############
    #                     if self.trueState[i, t - 1] == 0:
    #                         self.trueState[i, t - 1] = np.random.binomial(1, app_rate_experiment)
    #                     else:
    #                         self.trueState[i, t - 1] = 1 - np.random.binomial(1, disapp_rate_experiment)
    #                     # for pb action, wait for usual time and then pb  #############
    #
    #                     if self.trueState[i, t - 1] == 0:
    #                         self.trueState[i, t] = self.trueState[i, t-1]
    #                         self.belief[i, t] = 0
    #                         if self.reward[i, t-1]==0:
    #                             self.reward[i, t] = 0
    #                         else:
    #                             self.reward[i, t] = np.random.binomial(1, 1 - food_consumed)
    #                     else:
    #                         self.trueState[i, t] = np.random.binomial(1, food_missed)
    #
    #                         if self.trueState[i, t] == 1: # is dropped back after bp
    #                             self.belief[i, t] = self.nq - 1
    #                             if self.reward[i, t - 1] == 0:
    #                                 self.reward[i, t] = 0
    #                             else:
    #                                 self.reward[i, t] = np.random.binomial(1, 1 - food_consumed)
    #                         else: # not dropped back
    #                             self.belief[i, t] = 0
    #                             self.reward[i, t] = 1  # give some reward
    #
    #                     self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
    #                     self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
    #
    #
    # def data_generate_sfm(self, belief_ini = 0, rew_ini = 0):
    #     """
    #     This is a function that belongs to the oneboxMDP class. In the oneboxGenerate.py file, this function is implemented
    #     as a separate class, since at that time, the oneboxMDP class was not defined.
    #     In this file, all the functions are implemented under a single class.
    #
    #     :return: the observations
    #     """
    #
    #     food_missed = self.parameters['food_missed']  # available food dropped back into box after button press
    #     food_consumed = self.parameters['food_consumed']  # food in mouth is consumed
    #
    #     app_rate_experiment = self.parameters_exp['app_rate_experiment']
    #     disapp_rate_experiment = self.parameters_exp['disapp_rate_experiment']
    #
    #     for i in range(self.sampleNum):
    #         for t in range(self.sampleTime):
    #             if t == 0:
    #                 self.trueState[i, t] = np.random.binomial(1, app_rate_experiment)
    #                 self.reward[i, t], self.belief[i, t] = rew_ini, belief_ini
    #                 self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]    # This is for one box only
    #                 self.action[i, t] = self._choose_action(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
    #                 # action is based on softmax optpolicy
    #             else:
    #                 if self.action[i, t-1] != pb:
    #                     stattemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :], size = 1)
    #                     self.hybrid[i, t] = np.argmax(stattemp)
    #                         # not pressing button, hybrid state evolves probabilistically
    #                     self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)
    #                     self.action[i, t] = self._choose_action(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
    #
    #                     if self.trueState[i, t - 1] == 0:
    #                         self.trueState[i, t] = np.random.binomial(1, app_rate_experiment)
    #                     else:
    #                         self.trueState[i, t] = 1 - np.random.binomial(1, disapp_rate_experiment)
    #                 else:   # press button
    #                     #### for pb action, wait for usual time and then pb  #############
    #                     if self.trueState[i, t - 1] == 0:
    #                         self.trueState[i, t - 1] = np.random.binomial(1, app_rate_experiment)
    #                     else:
    #                         self.trueState[i, t - 1] = 1 - np.random.binomial(1, disapp_rate_experiment)
    #                     #### for pb action, wait for usual time and then pb  #############
    #
    #                     if self.trueState[i, t - 1] == 0:
    #                         self.trueState[i, t] = self.trueState[i, t-1]
    #                         self.belief[i, t] = 0
    #                         if self.reward[i, t-1]==0:
    #                             self.reward[i, t] = 0
    #                         else:
    #                             self.reward[i, t] = np.random.binomial(1, 1 - food_consumed)
    #                                     # With probability 1- food_consumed, reward is 1, not consumed
    #                                     # with probability food_consumed, reward is 0, consumed
    #                     # if true world is one, pb resets it to zero with probability
    #                     else:
    #                         self.trueState[i, t] = np.random.binomial(1, food_missed)
    #
    #                         if self.trueState[i, t] == 1: # is dropped back after bp
    #                             self.belief[i, t] = self.nq - 1
    #                             if self.reward[i, t - 1] == 0:
    #                                 self.reward[i, t] = 0
    #                             else:
    #                                 self.reward[i, t] = np.random.binomial(1, 1 - food_consumed)
    #                         else: # not dropped back
    #                             self.belief[i, t] = 0
    #                             self.reward[i, t] = 1  # give some reward
    #
    #                     self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
    #                     self.action[i, t] = self._choose_action(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
    #

    def _choose_action(self, pvec):
        # Generate action according to multinomial distribution
        # stattemp = np.random.multinomial(1, pvec)
        # return np.argmax(stattemp)

        return torch.multinomial(pvec, 1, replacement=True)



