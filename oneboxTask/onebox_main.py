from __future__ import division
from oneboxTask.onebox import *
from oneboxTask.onebox_generate import *
from oneboxTask.onebox_IRC_torch import *
import pickle
import os
import torch
import matplotlib.pyplot as plt
import collections
import time
import torch.utils.data as data_utils

def likelihood_tensor(para, obs, nq, nr, na, nl, discount):

    """
    maximum log-likelihood
    """

    pi = torch.ones(nq) / nq

    onebox_temp = oneboxMDP(discount, nq, nr, na, nl, para)
    onebox_temp.setupMDP()
    onebox_temp.solveMDP_sfm()
    ThA = onebox_temp.ThA
    softpolicy = onebox_temp.softpolicy
    oneboxHMM = HMMonebox(ThA, softpolicy, pi)

    log_likelihood = oneboxHMM.log_likelihood(obs, ThA, softpolicy)

    return log_likelihood

LR = 10**-3*5
EPS = 0.001
BATCH_SIZE = 2

def main():
    exsiting_data = True
    datestamp = '11222020(2327)'

    if exsiting_data == False:
        nq = 5 #torch.LongTensor([5])
        nr = 2 #torch.LongTensor([2])
        na = 2 #torch.LongTensor([2])
        nl = 1 #torch.LongTensor([1])
        discount = 0.99 #torch.tensor([0.99])

        app_rate = .1
        disapp_rate = .1
        food_missed = .1 #0
        food_consumed = .9 #1
        belief_diffusion = .1
        policy_temperature = .06
        push_button_cost = .3

        app_rate_experiment = .3
        disapp_rate_experiment = .1

        sample_length = 500
        sample_number = 1

        app_rate = torch.autograd.Variable(torch.tensor([app_rate]),requires_grad=True)
        disapp_rate = torch.autograd.Variable(torch.tensor([disapp_rate]),requires_grad=True)
        food_missed = torch.autograd.Variable(torch.tensor([food_missed]),requires_grad=True) #0
        food_consumed = torch.autograd.Variable(torch.tensor([food_consumed]),requires_grad=True) #.99 #1
        belief_diffusion = torch.autograd.Variable(torch.tensor([belief_diffusion]),requires_grad=True) #.1
        policy_temperature = torch.autograd.Variable(torch.tensor([policy_temperature]),requires_grad=True) #.061
        push_button_cost = torch.autograd.Variable(torch.tensor([push_button_cost]),requires_grad=True) #.3

        app_rate_experiment = torch.autograd.Variable(torch.tensor([app_rate_experiment]), requires_grad=True)
        disapp_rate_experiment = torch.autograd.Variable(torch.tensor([disapp_rate_experiment]), requires_grad=True)

        parameters_agent = collections.OrderedDict()
        parameters_agent['food_missed'] = food_missed
        parameters_agent['app_rate'] = app_rate
        parameters_agent['disapp_rate'] = disapp_rate
        parameters_agent['food_consumed'] = food_consumed
        parameters_agent['push_button_cost'] = push_button_cost
        parameters_agent['belief_diffusion'] = belief_diffusion
        parameters_agent['policy_temperature'] = policy_temperature

        parameters_exp = collections.OrderedDict()
        parameters_exp['app_rate_experiment'] = app_rate_experiment
        parameters_exp['disapp_rate_experiment'] = disapp_rate_experiment

        obsN, latN, truthN, datestring = onebox_data(parameters_agent, parameters_exp,
                                                     sample_length = sample_length,
                                                     sample_number = sample_number,
                                                     nq = nq, nr = nr, na = na, nl = nl,
                                                     discount = discount, policy = 'sfm',
                                                     belief_ini = 'rand', rew_ini = 'rand')
    else:
        #############################################
        ####### use generated data  #################
        #############################################

        path = os.getcwd()
        dataN_pkl_file = open(path + '/Data/' + datestamp + '_dataN_onebox.pkl', 'rb')
        dataN_pkl = pickle.load(dataN_pkl_file)
        dataN_pkl_file.close()

        obsN = dataN_pkl['observations']

        sample_number = obsN.shape[0]
        sample_length = obsN.shape[1]

        para_pkl_file = open(path + '/Data/' + datestamp + '_para_onebox.pkl', 'rb')
        para_pkl = pickle.load(para_pkl_file)
        para_pkl_file.close()

        discount = para_pkl['discount']
        nq = para_pkl['nq']
        nr = para_pkl['nr']
        na = para_pkl['na']
        nl = para_pkl['nl']
        food_missed = para_pkl['food_missed']
        app_rate = para_pkl['app_rate']
        disapp_rate = para_pkl['disapp_rate']
        food_consumed = para_pkl['food_consumed']
        push_button_cost = para_pkl['push_button_cost']
        app_rate_experiment = para_pkl['app_rate_experiment']
        disapp_rate_experiment = para_pkl['disapp_rate_experiment']
        belief_diffusion = para_pkl['belief_diffusion']
        policy_temperature = para_pkl['policy_temperature']

        parameters_agent = collections.OrderedDict()
        parameters_agent['food_missed'] = food_missed
        parameters_agent['app_rate'] = app_rate
        parameters_agent['disapp_rate'] = disapp_rate
        parameters_agent['food_consumed'] = food_consumed
        parameters_agent['push_button_cost'] = push_button_cost
        parameters_agent['belief_diffusion'] = belief_diffusion
        parameters_agent['policy_temperature'] = policy_temperature

        parameters_exp = collections.OrderedDict()
        parameters_exp['app_rate_experiment'] = app_rate_experiment
        parameters_exp['disapp_rate_experiment'] = disapp_rate_experiment
        #############################################
        #|||||||| use generated data  |||||||||||||||
        #############################################

    """
    look into data of one sample
    """
    # len = 50
    # obs = obsN[0, :len, :]   #act, rew
    # trueS = truthN[0, :len].squeeze()
    # lat = latN[0, :len].squeeze()
    #
    # plt.plot(obs[:, 0], 'b')
    # plt.plot(obs[:, 1], 'r')
    # plt.plot(trueS * 1.5, 'g')
    # plt.plot(lat, 'g--')
    # plt.show()


    """
    check gradient
    """
    # ll = likelihood_tensor(parameters_agent, obs, nq, nr, na, nl, discount)
    # ll.backward()
    # print([p.grad for k, p in parameters_agent.items()])

    """
    IRC
    """
    obsN = torch.reshape(obsN, (4, -1 ,obsN.shape[-1]))

    pointIni = parameters_agent

    # print(pointIni, '\n')
    # IRC_monkey1 = onebox_IRC_torch(discount, nq, nr, na, nl, pointIni)
    # start_time1 = time.time()
    # IRC_monkey1.IRC_batch(obsN,
    #                       lr=LR, eps=EPS, batch_size = 1, shuffle = False)
    # print('total time = ', time.time() - start_time1)
    # print(IRC_monkey1.log_likelihood_traj)
    # print('\n\n')

    #print(pointIni)
    IRC_monkey = onebox_IRC_torch(discount, nq, nr, na, nl, pointIni)
    start_time = time.time()
    IRC_monkey.IRC_batch(obsN, lr = LR, eps = EPS, batch_size = BATCH_SIZE, shuffle = True)
    print('total time = ', time.time() - start_time)
    print(IRC_monkey.log_likelihood_all[-1])

    plt.plot(IRC_monkey.log_likelihood_all)
    plt.show()

    print("finish")


if __name__ == "__main__":
    main()

