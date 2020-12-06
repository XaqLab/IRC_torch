from __future__ import division
from numpy import linalg as LA
from twoboxTask.twobox import *
from twoboxTask.twobox_HMM import *
from twoboxTask.twobox_IRC_torch import *
import pickle
import sys
from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
import collections
import time
import torch

def likelihood_tensor(para, obs, nq, nr, na, nl, discount):

    """
    maximum log-likelihood
    """

    pi = torch.ones(nq ** 2) / nq / nq

    twobox = twoboxMDP(discount, nq, nr, na, nl, para)
    twobox.setupMDP()
    twobox.solveMDP_sfm()
    ThA = twobox.ThA
    softpolicy = twobox.softpolicy

    twoboxHMM = HMMtwobox(ThA, softpolicy, pi)
    #complete_likelihood_old = twoboxHMM.computeQaux(obs, ThA, softpolicy)
    #latent_entropy = twoboxHMM.latent_entr(obs)
    #log_likelihood = complete_likelihood_old + latent_entropy

    #return softpolicy[0 ,0]
    return twoboxHMM.log_likelihood(obs, ThA, softpolicy)

LR = 10**-5*5
EPS = 0.1
BATCH_SIZE = 20

def main():

    sample_length = 1000
    sample_number = 1

    app_rate1 = .3
    disapp_rate1 = .1
    app_rate2 = .3
    disapp_rate2 = .1
    food_missed = .1 #0
    food_consumed = .9 #1
    belief_diffusion = .1
    policy_temperature = .1 #.06
    push_button_cost = .2
    grooming_reward = .3
    travel_cost = .1 #.1
    trip_prob = .05
    direct_prob = .1

    app_rate1_experiment = .3
    disapp_rate1_experiment = .1
    app_rate2_experiment = .3
    disapp_rate2_experiment = .1

    app_rate1 = torch.autograd.Variable(torch.tensor([app_rate1]), requires_grad=True)
    disapp_rate1 = torch.autograd.Variable(torch.tensor([disapp_rate1]), requires_grad=True)
    app_rate2 = torch.autograd.Variable(torch.tensor([app_rate2]), requires_grad=True)
    disapp_rate2 = torch.autograd.Variable(torch.tensor([disapp_rate2]), requires_grad=True)
    food_missed = torch.autograd.Variable(torch.tensor([food_missed]), requires_grad=True)  # 0
    food_consumed = torch.autograd.Variable(torch.tensor([food_consumed]), requires_grad=True)  # .99 #1
    belief_diffusion = torch.autograd.Variable(torch.tensor([belief_diffusion]), requires_grad=True)  # .1
    policy_temperature = torch.autograd.Variable(torch.tensor([policy_temperature]), requires_grad=True)  # .061
    push_button_cost = torch.autograd.Variable(torch.tensor([push_button_cost]), requires_grad=True)  # .3
    grooming_reward = torch.autograd.Variable(torch.tensor([grooming_reward]), requires_grad=True)  # .3
    travel_cost = torch.autograd.Variable(torch.tensor([travel_cost]), requires_grad=True)  # .3
    direct_prob = torch.autograd.Variable(torch.tensor([direct_prob]), requires_grad=True)  # .3
    trip_prob = torch.autograd.Variable(torch.tensor([trip_prob]), requires_grad=True)  # .3

    app_rate1_experiment = torch.autograd.Variable(torch.tensor([app_rate1_experiment]), requires_grad=True)
    disapp_rate1_experiment = torch.autograd.Variable(torch.tensor([disapp_rate1_experiment]), requires_grad=True)
    app_rate2_experiment = torch.autograd.Variable(torch.tensor([app_rate2_experiment]), requires_grad=True)
    disapp_rate2_experiment = torch.autograd.Variable(torch.tensor([disapp_rate2_experiment]), requires_grad=True)

    parameters_agent = collections.OrderedDict()
    parameters_agent['food_missed'] = food_missed
    parameters_agent['app_rate1'] = app_rate1
    parameters_agent['disapp_rate1'] = disapp_rate1
    parameters_agent['app_rate2'] = app_rate2
    parameters_agent['disapp_rate2'] = disapp_rate2
    parameters_agent['food_consumed'] = food_consumed
    parameters_agent['push_button_cost'] = push_button_cost
    parameters_agent['belief_diffusion'] = belief_diffusion
    parameters_agent['policy_temperature'] = policy_temperature
    parameters_agent['direct_prob'] = direct_prob
    parameters_agent['trip_prob'] = trip_prob
    parameters_agent['grooming_reward'] = grooming_reward
    parameters_agent['travel_cost'] = travel_cost

    parameters_exp = collections.OrderedDict()
    parameters_exp['app_rate1_experiment'] = app_rate1_experiment
    parameters_exp['disapp_rate1_experiment'] = disapp_rate1_experiment
    parameters_exp['app_rate2_experiment'] = app_rate2_experiment
    parameters_exp['disapp_rate2_experiment'] = disapp_rate2_experiment


    nq = 3
    na = 5
    nr = 2
    nl = 3  # center location, box1, box 2
    discount = 0.99

    # obsN, latN, truthN, datestring = twobox_data(parameters_agent, parameters_exp,
    #                                              sample_length = sample_length,
    #                                              sample_number = sample_number,
    #                                              nq = nq, nr = nr, na = na, nl = nl, discount = discount, optpolicy = 'sfm')

    path = os.getcwd()
    dataN_pkl_file = open(path + '/Data/12052020(0153)_dataN_twobox.pkl', 'rb')
    dataN_pkl = pickle.load(dataN_pkl_file)
    dataN_pkl_file.close()
    obsN = dataN_pkl['observations']

    obs = obsN[:, :20]


    """
    Check transition probability, reward functions, and log-likelihood
    """
    # p_last = parameters_agent.copy()
    # twobox = twoboxMDP(discount, nq, nr, na, nl, p_last)
    # twobox.setupMDP()
    # twobox.solveMDP_sfm()
    # ThA = twobox.ThA
    # softpolicy = twobox.softpolicy
    # #print(softpolicy)
    #
    # pi = torch.ones(nq ** 2) / nq / nq
    # twoboxHMM = HMMtwobox(ThA, softpolicy, pi)
    # #alpha = twoboxHMM.forward(obs)
    # #beta = twoboxHMM.backward(obs)
    # alpha, scale = twoboxHMM.forward_scale(obs)
    # beta = twoboxHMM.backward_scale(obs, scale)
    # xi = twoboxHMM.compute_xi(alpha, beta, obs)
    # complete_likelihood_old = twoboxHMM.computeQaux(obs, ThA, softpolicy)
    # latent_entropy = twoboxHMM.latent_entr(obs)
    # log_likelihood = complete_likelihood_old + latent_entropy
    # print(log_likelihood)

    """
    check gradient
    """
    # start_time = time.time()
    # ll = likelihood_tensor(parameters_agent, obs, nq, nr, na, nl, discount)
    # ll.backward()
    # print([p.grad for k, p in parameters_agent.items()])
    # print('total time = ', time.time() - start_time)

    """
    IRC
    """
    #obsN = torch.reshape(obsN, (4, -1, obsN.shape[-1]))

    pointIni = parameters_agent
    IRC_monkey = twobox_IRC_torch(discount, nq, nr, na, nl, pointIni)
    start_time = time.time()
    IRC_monkey.IRC_batch(obs, lr=LR, eps=EPS, batch_size=BATCH_SIZE, shuffle=True)
    print('total time = ', time.time() - start_time)
    print(IRC_monkey.log_likelihood_all[-1])

    plt.plot(IRC_monkey.log_likelihood_all)
    plt.show()

    print(1)


if __name__ == "__main__":
    main()