from __future__ import division
from numpy import linalg as LA
from twoboxTask.twobox import *
from twoboxTask.twobox_HMM import *
from twoboxTask.twobox_generate import *
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

def plot_data(act, rew, loc, lat, truth, showlen, startT, nq):
    #fig_posterior = plt.figure(figsize=(15, 10))
    #showlen = 100
    #startT = 440

    endT = startT + showlen
    showT = range(startT, endT)

    fig_posterior, [ax1, ax_loc, ax2] = plt.subplots(3, 1, figsize=(15, 10))

    # ax1 = plt.subplot(gs1[1])
    # ax1 = fig_posterior.add_subplot(512)
    #ax1.imshow(belief1_est[showT].T, interpolation='Nearest', cmap='gray', origin='lower', aspect='auto')
    ax1.plot(lat[showT, 0], color='dodgerblue', markersize=10, linewidth=3.0)
    ax1.plot(truth[showT, 0] * (nq-1), 'r.', markersize=10, linewidth=3.0)

    # ax1.set(title = 'belief of box 1 based on estimated parameters')
    # ax1.get_yaxis().labelpad = 70
    ax1.yaxis.set_label_coords(-0.1, 0.25)
    ax1.set_ylabel('Marginal belief \n about box 1', rotation=360, fontsize=22)
    ax1.set_xlim([0, showlen])
    ax1.set_xticks([])
    #ax1.set_yticks([0, nq - 1])
    ax1.set_yticklabels(['0', '1'])
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # ax_loc = plt.subplot(gs1[2])
    # ax_loc = fig_posterior.add_subplot(513)
    ax_loc.plot((np.remainder(loc[showT] + 1, 3) - 1) * 10, 'g.-', markersize=12, linewidth=5)
    # ax_loc.plot((np.remainder(loc[showT]+1, 3) - 1 ) * 10, 'm-')
    # ax_loc.plot(act[showT] // 4 * 10 * (np.abs(loc[showT]* 2 - 1.5) - 0.5 - 1),
    #              'v', markersize = 5)
    # ax_loc.plot(rew[showT] * 9, 'c*')
    box1_r = act[showT] // 4 * 7 * np.remainder(loc[showT] + 1, 3) * np.insert(rew[showT][1:], -1, 0) * 1.0
    box2_r = act[showT] // 4 * 7 * (np.remainder(loc[showT] + 1, 3) - 2) * np.insert(rew[showT][1:], -1, 0) * 1.0
    box1_n = act[showT] // 4 * 7 * np.remainder(loc[showT] + 1, 3) * (1 - np.insert(rew[showT][1:], -1, 0)) * 1.0 * (
                loc[showT] != 0)
    box2_n = act[showT] // 4 * 7 * (
                (np.remainder(loc[showT] + 1, 3) - 2) * (1 - np.insert(rew[showT][1:], -1, 0))) * 1.0 * (
                         loc[showT] != 0)
    box1_r[box1_r == 0] = np.nan
    box2_n[box2_n == 0] = np.nan
    box2_r[box2_r == 0] = np.nan
    box1_n[box1_n == 0] = np.nan
    ax_loc.plot(box2_r, '^', c='red', markersize=15)
    ax_loc.plot(box1_n, 'v', c='blue', markersize=15)
    ax_loc.plot(box2_n, '^', c='blue', markersize=15)
    ax_loc.plot(box1_r, 'v', c='red', markersize=15)

    ax_loc.set_xlim([0, showlen])
    ax_loc.spines['top'].set_visible(False)
    ax_loc.spines['right'].set_visible(False)
    ax_loc.spines['bottom'].set_visible(False)
    ax_loc.spines['left'].set_visible(False)
    ax_loc.set_ylim([-16,16])
    ax_loc.set_yticks([])
    ax_loc.set_xticks([])

    # ax2 = plt.subplot(gs1[4])
    # ax2 = fig_posterior.add_subplot(515)
    #ax2.imshow(belief2_est[showT].T, interpolation='Nearest', cmap='gray', origin='lower', aspect='auto')
    ax2.plot(lat[showT, 1], color='dodgerblue', markersize=10, linewidth=3.0)
    ax2.plot(truth[showT, 1] * (nq - 1), 'r.', markersize=10, linewidth=3.0)

    # ax2.set(title = 'belief of box 2 based on estimated parameters')
    ax2.set_xlabel('time', fontsize=18)
    # ax2.get_yaxis().labelpad = 70
    ax2.yaxis.set_label_coords(-0.1, 0.25)
    ax2.set_ylabel('Marginal belief \n about box 2', rotation=360, fontsize=22)
    ax2.set_xlim([0, showlen])
    ax2.tick_params(axis='both', which='major', labelsize=18)
    # ax2.set_yticks([0, nq - 1])
    ax2.set_yticklabels(['0', '1'])

    plt.tight_layout()
    plt.show()


LR = 10**-6*5
EPS = 0.1
BATCH_SIZE = 1


def main():
    exsiting_data = True
    datestamp = '12062020(1931)'

    if exsiting_data is False:

        nq = 5  # torch.LongTensor([5])
        nr = 2  # torch.LongTensor([2])
        na = 5  # torch.LongTensor([2])
        nl = 3  # torch.LongTensor([1])
        discount = 0.99  # torch.tensor([0.99])

        sample_length = 10000
        sample_number = 1

        app_rate1 = .1
        disapp_rate1 = .1
        app_rate2 = .2
        disapp_rate2 = 0.1
        food_missed = .1  # 0
        food_consumed = .9  # 1
        belief_diffusion = .1
        policy_temperature = .1  # .06
        push_button_cost = .2
        grooming_reward = .3
        travel_cost = .1  # .1
        trip_prob = .05
        direct_prob = .1

        app_rate1_experiment = .1
        disapp_rate1_experiment = .2
        app_rate2_experiment = .2
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

        obsN, latN, truthN, datestring = twobox_data(parameters_agent, parameters_exp,
                                                     sample_length=sample_length,
                                                     sample_number=sample_number,
                                                     nq=nq, nr=nr, na=na, nl=nl,
                                                     discount=discount, policy='sfm',
                                                     belief1_ini='rand', belief2_ini='rand',
                                                     rew_ini='rand', loc_ini=0)

    else:
        path = os.getcwd()
        dataN_pkl_file = open(path + '/Data/' + datestamp + '_dataN_twobox.pkl', 'rb')
        dataN_pkl = pickle.load(dataN_pkl_file)
        dataN_pkl_file.close()

        obsN = dataN_pkl['observations']
        latN = dataN_pkl['beliefs']
        truthN = dataN_pkl['trueStates']
        sample_number = obsN.shape[0]
        sample_length = obsN.shape[1]

        para_pkl_file = open(path + '/Data/' + datestamp + '_para_twobox.pkl', 'rb')
        para_pkl = pickle.load(para_pkl_file)
        para_pkl_file.close()

        discount = para_pkl['discount']
        nq = para_pkl['nq']
        nr = para_pkl['nr']
        na = para_pkl['na']
        nl = para_pkl['nl']

        food_missed = para_pkl['food_missed']
        app_rate1 = para_pkl['app_rate1']
        disapp_rate1 = para_pkl['disapp_rate1']
        app_rate2 = para_pkl['app_rate2']
        disapp_rate2 = para_pkl['disapp_rate2']
        food_consumed = para_pkl['food_consumed']
        push_button_cost = para_pkl['push_button_cost']
        belief_diffusion = para_pkl['belief_diffusion']
        policy_temperature = para_pkl['policy_temperature']
        direct_prob = para_pkl['direct_prob']
        trip_prob = para_pkl['trip_prob']
        grooming_reward = para_pkl['grooming_reward']
        travel_cost = para_pkl['travel_cost']
        app_rate1_experiment = para_pkl['app_rate1_experiment']
        disapp_rate1_experiment = para_pkl['disapp_rate1_experiment']
        app_rate2_experiment = para_pkl['app_rate2_experiment']
        disapp_rate2_experiment = para_pkl['disapp_rate2_experiment']

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

    obsN = obsN[:, :1000, :]

    """
    plot the data
    """
    # obs = obsN.squeeze()[:1000]  # action, reward, location
    # lat = latN.squeeze()
    # truth = truthN.squeeze()
    # act = obs[:, 0]
    # rew = obs[:, 1]
    # loc = obs[:, 2]
    # plot_data(act, rew, loc, lat, truth, 50, 0, nq)
    # print(1)

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
    # obs = obsN.squeeze()
    # start_time = time.time()
    # ll = likelihood_tensor(parameters_agent, obs, nq, nr, na, nl, discount)
    # ll.backward()
    # print([p.grad for k, p in parameters_agent.items()])
    # print('total time = ', time.time() - start_time)

    """
    IRC
    """
    obsN = torch.reshape(obsN, (1, -1, obsN.shape[-1]))

    pointIni = parameters_agent
    IRC_monkey = twobox_IRC_torch(discount, nq, nr, na, nl, pointIni)
    start_time = time.time()
    IRC_monkey.IRC_batch(obsN, lr=LR, eps=EPS, batch_size=BATCH_SIZE, shuffle=True)
    print('total time = ', time.time() - start_time)
    print(IRC_monkey.log_likelihood_traj[-1])

    plt.plot(IRC_monkey.log_likelihood_traj)
    plt.show()

    print(1)


if __name__ == "__main__":
    main()