from twoboxTask.twobox import *
import pickle
import os
from datetime import datetime
#import numpy as np

def twobox_data(parameters, parameters_exp, sample_length, sample_number, nq, nr, na, nl, discount,
                policy, belief1_ini, belief2_ini, rew_ini, loc_ini):

    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')  # current time used to set file name

    food_missed = parameters['food_missed']  # available food dropped back into box after button press
    app_rate1 = parameters['app_rate1']  # reward becomes available
    disapp_rate1 = parameters['disapp_rate1']  # available food disappears
    app_rate2 = parameters['app_rate2']  # reward becomes available
    disapp_rate2 = parameters['disapp_rate2']  # available food disappears
    trip_prob = parameters['trip_prob']  # animal trips, doesn't go to target location
    direct_prob = parameters['direct_prob']  # animal goes right to target, skipping location 0
    food_consumed = parameters['food_consumed']  # food in mouth is consumed
    push_button_cost = parameters['push_button_cost']
    travel_cost = parameters['travel_cost']
    grooming_reward = parameters['grooming_reward']
    belief_diffusion = parameters['belief_diffusion']
    policy_temperature = parameters['policy_temperature']
    reward = 1

    app_rate1_experiment = parameters_exp['app_rate1_experiment']
    disapp_rate1_experiment = parameters_exp['disapp_rate1_experiment']
    app_rate2_experiment = parameters_exp['app_rate2_experiment']
    disapp_rate2_experiment = parameters_exp['disapp_rate2_experiment']

    """ Gnerate data"""
    print("\nGenerating data...")
    T = sample_length
    N = sample_number

    twoboxdata = twobox_generate(discount, nq, nr, na, nl, parameters, parameters_exp, T, N)
    twoboxdata.data_generate(policy, belief1_ini, belief2_ini, rew_ini, loc_ini)

    action = twoboxdata.action
    location = twoboxdata.location
    belief1 = twoboxdata.belief1
    belief2 = twoboxdata.belief2
    reward = twoboxdata.reward
    trueState1 = twoboxdata.trueState1
    trueState2 = twoboxdata.trueState2

    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data
    obsN = torch.cat((action, reward, location), dim = -1)  # includes the action and the observable states
    latN = torch.cat((belief1, belief2), dim = -1)
    truthN = torch.cat((trueState1, trueState2), dim = -1)
    dataN = torch.cat((obsN, latN, truthN), dim = -1)

    path = os.getcwd()
    ### write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}
    data_output = open(path + '/Data/' + datestring + '_dataN_twobox' + '.pkl', 'wb')
    pickle.dump(data_dict, data_output)
    data_output.close()

    ### write all model parameters to file
    para_dict = {'discount': discount,
                 'nq': nq,
                 'nr': nr,
                 'nl': nl,
                 'na': na,
                 'food_missed': food_missed,
                 'app_rate1': app_rate1,
                 'app_rate2': app_rate2,
                 'disapp_rate1': disapp_rate1,
                 'disapp_rate2': disapp_rate2,
                 'trip_prob': trip_prob,
                 'direct_prob': direct_prob,
                 'food_consumed': food_consumed,
                 'reward': reward,
                 'grooming_reward': grooming_reward,
                 'travel_cost': travel_cost,
                 'push_button_cost': push_button_cost,
                 'belief_diffusion': belief_diffusion,
                 'policy_temperature': policy_temperature,
                 'app_rate1_experiment': app_rate1_experiment,
                 'disapp_rate1_experiment': disapp_rate1_experiment,
                 'app_rate2_experiment': app_rate2_experiment,
                 'disapp_rate2_experiment': disapp_rate2_experiment
                 }

    para_output = open(path + '/Data/' + datestring + '_para_twobox' + '.pkl', 'wb')
    pickle.dump(para_dict, para_output)
    para_output.close()
    # print(para_pkl['disappRate2'])

    print('Data stored in files')
    return obsN, latN, truthN, datestring


class twobox_generate(twoboxMDP):
    def __init__(self, discount, nq, nr, na, nl, parameters, parameters_exp,
                 sampleTime, sampleNum):
        twoboxMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.parameters_exp = parameters_exp
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime

        self.action = [] #np.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
        self.hybrid = [] #np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.location = [] #np.empty((sampleNum, sampleTime), int)  # initialize location state
        self.belief1 = [] #np.empty((self.sampleNum, self.sampleTime), int)
        self.belief2 = [] #np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = [] #np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
        self.trueState1 = [] #np.zeros((self.sampleNum, self.sampleTime))
        self.trueState2 = [] #np.zeros((self.sampleNum, self.sampleTime))

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()


    # def dataGenerate_op(self, belief1_ini, rew_ini, belief2_ini, loc_ini):
    #     self.trueState1 = np.zeros((self.sampleNum, self.sampleTime))
    #     self.trueState2 = np.zeros((self.sampleNum, self.sampleTime))
    #
    #     ## Parameters
    #     # beta = self.parameters[0]     # available food dropped back into box after button press
    #     # gamma1 = self.parameters[1]   # reward becomes available in box 1
    #     # gamma2 = self.parameters[2]   # reward becomes available in box 2
    #     # delta = self.parameters[3]    # animal trips, doesn't go to target location
    #     # direct = self.parameters[4]   # animal goes right to target, skipping location 0
    #     # epsilon1 = self.parameters[5] # available food disappears from box 1
    #     # epsilon2 = self.parameters[6] # available food disappears from box 2
    #     # rho = self.parameters[7]      # food in mouth is consumed
    #     # # eta = .0001                 # random diffusion of belief
    #     # # State rewards
    #     # Reward = self.parameters[8]   # reward per time step with food in mouth
    #     # Groom = self.parameters[9]     # location 0 reward
    #     # # Action costs
    #     # travelCost = self.parameters[10]
    #     # pushButtonCost = self.parameters[11]
    #
    #     beta = 0     # available food dropped back into box after button press
    #     gamma1 = self.parameters[0]   # reward becomes available in box 1
    #     gamma2 = self.parameters[1]   # reward becomes available in box 2
    #     delta = 0    # animal trips, doesn't go to target location
    #     direct = 0   # animal goes right to target, skipping location 0
    #     epsilon1 = self.parameters[2] # available food disappears from box 1
    #     epsilon2 = self.parameters[3] # available food disappears from box 2
    #     rho = 1      # food in mouth is consumed
    #     # State rewards
    #     Reward = 1   # reward per time step with food in mouth
    #     Groom = self.parameters[4]     # location 0 reward
    #     # Action costs
    #     travelCost = self.parameters[5]
    #     pushButtonCost = self.parameters[6]
    #
    #     Tb1 = beliefTransitionMatrixGaussian(gamma1, epsilon1, self.nq, sigmaTb)
    #     Tb2 = beliefTransitionMatrixGaussian(gamma2, epsilon2, self.nq, sigmaTb)
    #
    #     ## Generate data
    #     for n in range(self.sampleNum):
    #         for t in range(self.sampleTime):
    #             if t == 0:
    #                 # Initialize the true world states, sensory information and latent states
    #                 self.trueState1[n, t] = np.random.binomial(1, gamma1)
    #                 self.trueState2[n, t] = np.random.binomial(1, gamma2)
    #
    #                 self.location[n, t], self.belief1[n, t], self.reward[n, t], self.belief2[
    #                     n, t] = loc_ini, belief1_ini, rew_ini, belief2_ini
    #                 self.hybrid[n, t] = self.location[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) + \
    #                                     self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for optpolicy choosing
    #                 self.action[n, t] = self.optpolicy[self.hybrid[n, t]]
    #             else:
    #                 # variables evolve with dynamics
    #                 if self.action[n, t - 1] != pb:
    #                     acttemp = np.random.multinomial(1, self.ThA[self.action[n, t - 1], self.hybrid[n, t - 1], :], size=1)
    #                     self.hybrid[n, t] = np.argmax(acttemp)
    #
    #                     self.location[n, t] = divmod(self.hybrid[n, t], self.nq * self.nr * self.nq)[0]
    #                     self.belief1[n, t] = divmod(self.hybrid[n, t] - self.location[n, t] * (self.nq * self.nr * self.nq), self.nr * self.nq)[0]
    #                     self.reward[n, t] = divmod(self.hybrid[n, t] - self.location[n, t] * (self.nq * self.nr * self.nq) \
    #                                                - self.belief1[n, t] * (self.nr * self.nq), self.nq)[0]
    #                     self.belief2[n, t] = self.hybrid[n, t] - self.location[n, t] * (self.nq * self.nr * self.nq) - \
    #                                          self.belief1[n, t] * (self.nr * self.nq) - self.reward[n, t] * self.nq
    #
    #                     self.action[n, t] = self.optpolicy[self.hybrid[n, t]]
    #
    #                     # button not pressed, then true world dynamic is not affected by actions
    #                     if self.trueState1[n, t - 1] == 0:
    #                         self.trueState1[n, t] = np.random.binomial(1, gamma1)
    #                     else:
    #                         self.trueState1[n, t] = 1 - np.random.binomial(1, epsilon1)
    #
    #                     if self.trueState2[n, t - 1] == 0:
    #                         self.trueState2[n, t] = np.random.binomial(1, gamma2)
    #                     else:
    #                         self.trueState2[n, t] = 1 - np.random.binomial(1, epsilon2)
    #
    #                 if self.action[n, t - 1] == pb:  # press button
    #                     self.location[n, t] = self.location[n, t - 1]  # pressing button does not change location
    #
    #                     if self.location[n, t] == 1:  # consider location 1 case
    #                         self.belief2[n, t] = np.argmax(np.random.multinomial(1, Tb2[:, self.belief2[n, t - 1]], size=1))
    #                         # belief on box 2 is independent on box 1
    #                         if self.trueState2[n, t - 1] == 0:
    #                             self.trueState2[n, t] = np.random.binomial(1, gamma2)
    #                         else:
    #                             self.trueState2[n, t] = 1 - np.random.binomial(1, epsilon2)
    #
    #                         if self.trueState1[n, t - 1] == 0:
    #                             self.trueState1[n, t] = self.trueState1[n, t - 1]
    #                             # if true world is zero, pb does not change real state
    #                             # assume that the real state does not change during button press
    #                             self.belief1[n, t] = 0  # after open the box, the animal is sure that there is no food there
    #                             if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
    #                                 self.reward[n, t] = 0
    #                             else:
    #                                 self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
    #                         else:
    #                             self.trueState1[n, t] = 0  # if true world is one, pb resets it to zero
    #                             self.belief1[n, t] = 0
    #                             self.reward[n, t] = 1  # give some reward
    #
    #                     if self.location[n, t] == 2:  # consider location 2 case
    #                         self.belief1[n, t] = np.argmax(np.random.multinomial(1, Tb1[:, self.belief1[n, t - 1]], size=1))
    #                         # belief on box 1 is independent on box 2
    #                         if self.trueState1[n, t - 1] == 0:
    #                             self.trueState1[n, t] = np.random.binomial(1, gamma1)
    #                         else:
    #                             self.trueState1[n, t] = 1 - np.random.binomial(1, epsilon1)
    #
    #                         if self.trueState2[n, t - 1] == 0:
    #                             self.trueState2[n, t] = self.trueState2[n, t - 1]
    #                             # if true world is zero, pb does not change real state
    #                             # assume that the real state does not change during button press
    #                             self.belief2[n, t] = 0  # after open the box, the animal is sure that there is no food there
    #                             if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
    #                                 self.reward[n, t] = 0
    #                             else:
    #                                 self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
    #                         else:
    #                             self.trueState2[n, t] = 0  # if true world is one, pb resets it to zero
    #                             self.belief2[n, t] = 0
    #                             self.reward[n, t] = 1  # give some reward
    #
    #                 self.hybrid[n, t] = self.location[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) \
    #                                     + self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for optpolicy choosing
    #                 self.action[n, t] = self.optpolicy[self.hybrid[n, t]]
    #


    def data_generate(self, policy, belief1_ini, belief2_ini, rew_ini, loc_ini):

        food_missed = self.parameters['food_missed']  # available food dropped back into box after button press
        app_rate1 = self.parameters['app_rate1']  # reward becomes available
        disapp_rate1 = self.parameters['disapp_rate1']  # available food disappears
        app_rate2 = self.parameters['app_rate2']  # reward becomes available
        disapp_rate2 = self.parameters['disapp_rate2']  # available food disappears
        trip_prob = self.parameters['trip_prob']  # animal trips, doesn't go to target location
        direct_prob = self.parameters['direct_prob']  # animal goes right to target, skipping location 0
        food_consumed = self.parameters['food_consumed']  # food in mouth is consumed
        push_button_cost = self.parameters['push_button_cost']
        travel_cost = self.parameters['travel_cost']
        grooming_reward = self.parameters['grooming_reward']
        belief_diffusion = self.parameters['belief_diffusion']
        policy_temperature = self.parameters['policy_temperature']
        reward = 1

        app_rate1_experiment = self.parameters_exp['app_rate1_experiment']
        disapp_rate1_experiment = self.parameters_exp['disapp_rate1_experiment']
        app_rate2_experiment = self.parameters_exp['app_rate2_experiment']
        disapp_rate2_experiment = self.parameters_exp['disapp_rate2_experiment']


        Tb1 = beliefTransitionMatrixGaussian(app_rate1, disapp_rate1, self.nq, belief_diffusion)
        Tb2 = beliefTransitionMatrixGaussian(app_rate2, disapp_rate2, self.nq, belief_diffusion)

        ## Generate data
        for i in range(self.sampleNum):
            self.action.append([])
            self.hybrid.append([])
            self.belief1.append([])
            self.belief2.append([])
            self.location.append([])
            self.reward.append([])
            self.trueState1.append([])
            self.trueState2.append([])


            if rew_ini == 'rand':
                rew_ini = torch.randint(self.nr, (1,)).long()
            else:
                rew_ini = torch.LongTensor([rew_ini]).long()

            if loc_ini == 'rand':
                loc_ini = torch.randint(self.nl, (1,)).long()
            else:
                loc_ini = torch.LongTensor([loc_ini]).long()

            if belief1_ini == 'rand':
                belief1_ini = torch.randint(self.nq, (1,)).long()
            else:
                belief1_ini = torch.LongTensor([belief1_ini]).long()

            if belief2_ini == 'rand':
                belief2_ini = torch.randint(self.nq, (1,)).long()
            else:
                belief2_ini = torch.LongTensor([belief2_ini]).long()

            for t in range(self.sampleTime):
                if t == 0:
                    # Initialize the true world states, sensory information and latent states
                    # self.trueState1[i, t] = np.random.binomial(1, app_rate1_experiment)
                    # self.trueState2[i, t] = np.random.binomial(1, app_rate2_experiment)

                    self.trueState1[i].append(torch.bernoulli(app_rate1_experiment).long())
                    self.trueState2[i].append(torch.bernoulli(app_rate2_experiment).long())
                    self.reward[i].append(rew_ini)
                    self.location[i].append(loc_ini)
                    self.belief1[i].append(belief1_ini)
                    self.belief2[i].append(belief2_ini)
                    self.hybrid[i].append(self.location[i][-1] * (self.nq * self.nr * self.nq) + self.belief1[i][-1] * (
                                self.nr * self.nq) + self.reward[i][-1] * self.nq + self.belief2[i][-1] )  # hybrid state, for optpolicy choosing

                    # self.location[i, t], self.belief1[i, t], self.reward[i, t], self.belief2[
                    #     i, t] = loc_ini, belief1_ini, rew_ini, belief2_ini
                    # self.hybrid[i, t] = self.location[i, t] * (self.nq * self.nr * self.nq) + self.belief1[i, t] * (self.nr * self.nq) + \
                    #                     self.reward[i, t] * self.nq + self.belief2[i, t]  # hybrid state, for optpolicy choosing
                    #self.action[i, t] = self._chooseAction(self.softpolicy.T[self.hybrid[i, t]])
                    if policy == 'opt':
                        # self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
                        self.action[i].append(self.optpolicy[self.hybrid[i][-1]])
                    elif policy == 'sfm':
                        # self.action[i, t] = self._choose_action(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
                        self.action[i].append(self._choose_action(self.softpolicy.t()[self.hybrid[i][-1]].squeeze()))

                else:
                    if self.action[i][- 1] == pb and self.location[i][- 1] == 0:
                        self.action[i][- 1] = torch.LongTensor([a0]).long()  # cannot press button at location 0

                    # variables evolve with dynamics
                    if self.action[i][- 1] != pb:
                        # acttemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :], size=1)
                        # self.hybrid[i, t] = np.argmax(acttemp)
                        self.hybrid[i].append(torch.multinomial(self.ThA[self.action[i][-1],
                                                                self.hybrid[i][-1], :].squeeze(), 1, replacement=True))

                        self.location[i].append(torch.floor_divide(self.hybrid[i][-1], self.nq * self.nr * self.nq))
                        self.belief1[i].append(torch.floor_divide(self.hybrid[i][-1] - self.location[i][-1] * (self.nq * self.nr * self.nq),
                                                                self.nr * self.nq))
                        self.reward[i].append(torch.floor_divide(self.hybrid[i][-1] - self.location[i][-1] * (self.nq * self.nr * self.nq)
                                                               - self.belief1[i][-1] * (self.nr * self.nq), self.nq))
                        self.belief2[i].append(self.hybrid[i][-1] - self.location[i][-1] * (self.nq * self.nr * self.nq) - \
                                             self.belief1[i][-1] * (self.nr * self.nq) - self.reward[i][-1] * self.nq)



                        # button not pressed, then true world dynamic is not affected by actions
                        if self.trueState1[i][- 1] == 0:
                            self.trueState1[i].append(torch.bernoulli(app_rate1_experiment).long())
                        else:
                            self.trueState1[i].append(1 - torch.bernoulli(disapp_rate1_experiment).long())

                        if self.trueState2[i][- 1] == 0:
                            self.trueState2[i].append(torch.bernoulli(app_rate2_experiment).long())
                        else:
                            self.trueState2[i].append(1 - torch.bernoulli(disapp_rate2_experiment).long())

                    if self.action[i][- 1] == pb:  # press button
                        self.location[i].append(self.location[i][- 1] ) # pressing button does not change location

                        #### for pb action, wait for usual time and then pb  #############
                        if self.trueState1[i][- 1] == 0:
                            statetemp1 = torch.bernoulli(app_rate1_experiment).long()
                        else:
                            statetemp1 = 1 - torch.bernoulli(disapp_rate1_experiment).long()

                        if self.trueState2[i][- 1] == 0:
                            statetemp2 = torch.bernoulli(app_rate2_experiment).long()
                        else:
                            statetemp2 = 1 - torch.bernoulli(disapp_rate2_experiment).long()
                        #### for pb action, wait for usual time and then pb  #############

                        if self.location[i][-1] == 1:  # consider location 1 case
                            self.belief2[i].append(torch.multinomial(Tb2[:, self.belief2[i][- 1]].squeeze(), 1,
                                                                     replacement=True))
                            if statetemp2 == 0:
                                self.trueState2[i].append(torch.bernoulli(app_rate2_experiment).long())
                            else:
                                self.trueState2[i].append(1 - torch.bernoulli(disapp_rate2_experiment).long())

                            #self.belief2[i, t] = np.argmax(np.random.multinomial(1, Tb2[:, self.belief2[i, t - 1]], size=1))
                            # belief on box 2 is independent on box 1

                            # if self.trueState2[i, t - 1] == 0:
                            #     self.trueState2[i, t] = np.random.binomial(1, app_rate2_experiment)
                            # else:
                            #     self.trueState2[i, t] = 1 - np.random.binomial(1, disapp_rate2_experiment)

                            if statetemp1 == 0:
                                #self.trueState1[i, t] = self.trueState1[i, t - 1]
                                self.trueState1[i].append(statetemp1)
                                # if true world is zero, pb does not change real state
                                # assume that the real state does not change during button press
                                self.belief1[i].append(torch.LongTensor([0]))  # after open the box, the animal is sure that there is no food there
                                if self.reward[i][-1] == 0:
                                    self.reward[i].append(torch.LongTensor([0]))
                                else:
                                    self.reward[i].append(torch.bernoulli(1 - food_consumed).long())
                            else:
                                self.trueState1[i].append(torch.bernoulli(food_missed).long())
                                if self.trueState1[i][-1] == 1:  # is dropped back after bp
                                    self.belief1[i].append(torch.LongTensor([self.nq - 1]).long())
                                    if self.reward[i][- 1] == 0:
                                        self.reward[i].append(torch.LongTensor([0]).long())
                                    else:
                                        self.reward[i].append(torch.bernoulli(1 - food_consumed).long())
                                        # self.reward[i, t] = np.random.binomial(1, 1 - food_consumed.detach().numpy())
                                else:  # not dropped back
                                    self.belief1[i].append(torch.LongTensor([0]).long())
                                    self.reward[i].append(torch.LongTensor([1]).long())

                        if self.location[i][-1] == 2:  # consider location 2 case
                            self.belief1[i].append(torch.multinomial(Tb1[:, self.belief1[i][- 1]].squeeze(), 1,
                                                                     replacement=True))
                            # belief on box 1 is independent on box 2
                            if statetemp1 == 0:
                                self.trueState1[i].append(torch.bernoulli(app_rate1_experiment).long())
                            else:
                                self.trueState1[i].append(1 - torch.bernoulli(disapp_rate1_experiment).long())

                            if statetemp2 == 0:
                                self.trueState2[i].append(statetemp2)
                                # if true world is zero, pb does not change real state
                                # assume that the real state does not change during button press
                                self.belief2[i].append(torch.LongTensor([0]))   # after open the box, the animal is sure that there is no food there
                                if self.reward[i][-1] == 0:
                                    self.reward[i].append(torch.LongTensor([0]))
                                else:
                                    self.reward[i].append(torch.bernoulli(1 - food_consumed).long())
                            else:
                                self.trueState2[i].append(torch.bernoulli(food_missed).long())
                                if self.trueState2[i][-1] == 1:  # is dropped back after bp
                                    self.belief2[i].append(torch.LongTensor([self.nq - 1]).long())
                                    if self.reward[i][- 1] == 0:
                                        self.reward[i].append(torch.LongTensor([0]).long())
                                    else:
                                        self.reward[i].append(torch.bernoulli(1 - food_consumed).long())
                                        # self.reward[i, t] = np.random.binomial(1, 1 - food_consumed.detach().numpy())
                                else:  # not dropped back
                                    self.belief2[i].append(torch.LongTensor([0]).long())
                                    self.reward[i].append(torch.LongTensor([1]).long())


                    self.hybrid[i].append(self.location[i][-1] * (self.nq * self.nr * self.nq) + self.belief1[i][-1] * (self.nr * self.nq) \
                                        + self.reward[i][-1] * self.nq + self.belief2[i][-1])  # hybrid state, for optpolicy choosing
                    if policy == 'opt':
                        # self.action[i, t] = self.optpolicy[self.hybrid[i, t]]
                        self.action[i].append(self.optpolicy[self.hybrid[i][-1]])
                    elif policy == 'sfm':
                        # self.action[i, t] = self._choose_action(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
                        self.action[i].append(self._choose_action(self.softpolicy.t()[self.hybrid[i][-1]].squeeze()))

            self.action[-1] = torch.stack(self.action[-1])
            self.hybrid[-1] = torch.stack(self.hybrid[-1])
            self.belief1[-1] = torch.stack(self.belief1[-1])
            self.belief2[-1] = torch.stack(self.belief2[-1])
            self.reward[-1] = torch.stack(self.reward[-1])
            self.location[-1] = torch.stack(self.location[-1])
            self.trueState1[-1] = torch.stack(self.trueState1[-1])
            self.trueState2[-1] = torch.stack(self.trueState2[-1])

        self.action = torch.stack(self.action)
        self.hybrid = torch.stack(self.hybrid)
        self.belief1 = torch.stack(self.belief1)
        self.belief2 = torch.stack(self.belief2)
        self.reward = torch.stack(self.reward)
        self.location = torch.stack(self.location)
        self.trueState1 = torch.stack(self.trueState1)
        self.trueState2 = torch.stack(self.trueState2)


    def _choose_action(self, pvec):
        # Generate action according to multinomial distribution
        return torch.multinomial(pvec, 1, replacement=True)









