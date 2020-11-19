from __future__ import division
from numpy import linalg as LA
from oneboxTask.onebox import *
from oneboxTask.onebox_generate import *
from oneboxTask.onebox_IRC import *
import pickle
import sys
from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
import collections
import time
import numdifftools as nd

# E_MAX_ITER = 200 # 100    # maximum number of iterations of E-step
# GD_THRESHOLD = 0.01 # 0.01      # stopping criteria of M-step (gradient descent)
# E_EPS = 10 ** -6                  # stopping criteria of E-step
# M_LR_INI = 8 * 10 ** -6           # initial learning rate in the gradient descent step
# LR_DEC =  4                       # number of times that the learning rate can be reduced


def likelihood_tensor(food_missed, app_rate, disapp_rate, food_consumed, push_button_cost, belief_diffusion, policy_temperature,
                      obs, nq, nr, na, nl, discount):

    #food_missed, app_rate, disapp_rate, food_consumed, push_button_cost, belief_diffusion, policy_temperature = para
    parameters_agent = {'food_missed': food_missed,
                        'app_rate': app_rate,
                        'disapp_rate': disapp_rate,
                        'food_consumed': food_consumed,
                        'push_button_cost': push_button_cost,
                        'belief_diffusion': belief_diffusion,
                        'policy_temperature': policy_temperature
                        }
    parameters_agent = collections.OrderedDict(sorted(parameters_agent.items()))
    #print(parameters_agent.keys())

    # parameters_exp = {'app_rate_experiment': app_rate_experiment,
    #                   'disapp_rate_experiment': disapp_rate_experiment
    #                   }
    # parameters_exp = collections.OrderedDict(sorted(parameters_exp.items()))

    # nq = 5
    # na = 2
    # nr = 2
    # nl = 1
    # discount = 0.99

    # obsN, latN, truthN, datestring = onebox_data(parameters_agent, parameters_exp,
    #                                              sample_length = sample_length,
    #                                              sample_number = sample_number,
    #                                              nq = nq, nr = nr, na = na, discount = discount, policy = 'sfm')

    path = os.getcwd()
    dataN_pkl_file = open(path + '/Data/10272020(1115)_dataN_onebox.pkl', 'rb')
    dataN_pkl = pickle.load(dataN_pkl_file)
    dataN_pkl_file.close()
    obsN = dataN_pkl['observations']

    # actN = obsN[:, :, 0]
    # rewN = obsN[:, :, 1]

    """
    maximum log-likelihood
    """
    # obs = np.squeeze(obsN)
    #obs = np.squeeze(obsN)[:10, :]
    #
    pi = torch.ones(nq) / nq
    # pi[0].backward()
    # food_missed.grad
    onebox_temp = oneboxMDP(discount, nq, nr, na, nl, parameters_agent)
    onebox_temp.setupMDP()
    onebox_temp.solveMDP_sfm()
    ThA = onebox_temp.ThA
    softpolicy = onebox_temp.softpolicy
    oneboxHMM = HMMonebox(ThA, softpolicy, pi)

    #alpha = oneboxHMM.forward(obs)
    #beta = oneboxHMM.backward(obs)

    #alpha, scale = oneboxHMM.forward_scale(obs)
    #beta = oneboxHMM.backward_scale(obs, scale)
    #xi = oneboxHMM.compute_xi(alpha, beta, obs)
    #lat_ent = oneboxHMM.latent_entr(obs)
    #print(lat_ent)
    log_likelihood = oneboxHMM.log_likelihood(obs, ThA, softpolicy)
    #print(log_likelihood)
    #print(1)

    #return log_likelihood.clone().detach().requires_grad_(True)

    return log_likelihood

def main():
    start_time = time.time()


    nq = 5
    nr = 2
    na = 2
    nl = 1
    discount = 0.99

    app_rate = .1
    disapp_rate = .1
    food_missed = .1 #0
    food_consumed = .9 #1
    belief_diffusion = .1
    policy_temperature = .06
    push_button_cost = .3

    app_rate_experiment = .3
    disapp_rate_experiment = .1

    parameters_agent = {'food_missed': food_missed,
                        'app_rate': app_rate,
                        'disapp_rate': disapp_rate,
                        'food_consumed': food_consumed,
                        'push_button_cost': push_button_cost,
                        'belief_diffusion': belief_diffusion,
                        'policy_temperature': policy_temperature
                        }
    parameters_agent = collections.OrderedDict(sorted(parameters_agent.items()))

    parameters_exp = {'app_rate_experiment': app_rate_experiment,
                      'disapp_rate_experiment': disapp_rate_experiment
                      }
    parameters_exp = collections.OrderedDict(sorted(parameters_exp.items()))

    sample_length = 1000
    sample_number = 1


    # obsN, latN, truthN, datestring = onebox_data(parameters_agent, parameters_exp,
    #                                              sample_length = sample_length,
    #                                              sample_number = sample_number,
    #                                              nq = nq, nr = nr, na = na, nl = nl, discount = discount, policy = 'sfm')

    #############################################
    ####### use generated data  #################
    #############################################
    path = os.getcwd()
    dataN_pkl_file = open(path + '/Data/10272020(1115)_dataN_onebox.pkl', 'rb')
    dataN_pkl = pickle.load(dataN_pkl_file)
    dataN_pkl_file.close()

    para_pkl_file = open(path + '/Data/10272020(1115)_para_onebox.pkl', 'rb')
    para_pkl = pickle.load(para_pkl_file)
    para_pkl_file.close()

    # discount = para_pkl['discount']
    # nq = para_pkl['nq']
    # nr = para_pkl['nr']
    # na = para_pkl['na']
    # nl = para_pkl['nl']
    # food_missed = para_pkl['food_missed']
    # app_rate = para_pkl['app_rate']
    # disapp_rate = para_pkl['disapp_rate']
    # food_consumed = para_pkl['food_consumed']
    # push_button_cost = para_pkl['push_button_cost']
    # app_rate_experiment = para_pkl['app_rate_experiment']
    # disapp_rate_experiment = para_pkl['disapp_rate_experiment']

    obsN = dataN_pkl['observations']

    #############################################
    #|||||||| use generated data  |||||||||||||||
    #############################################

    app_rate = torch.autograd.Variable(torch.tensor([app_rate]),requires_grad=True)
    disapp_rate = torch.autograd.Variable(torch.tensor([disapp_rate]),requires_grad=True)
    food_missed = torch.autograd.Variable(torch.tensor([food_missed]),requires_grad=True) #0
    food_consumed = torch.autograd.Variable(torch.tensor([food_consumed]),requires_grad=True) #.99 #1
    belief_diffusion = torch.autograd.Variable(torch.tensor([belief_diffusion]),requires_grad=True) #.1
    policy_temperature = torch.autograd.Variable(torch.tensor([policy_temperature]),requires_grad=True) #.061
    push_button_cost = torch.autograd.Variable(torch.tensor([push_button_cost]),requires_grad=True) #.3


    obs = np.squeeze(obsN)[:10, :]
    # # #
    # # # pointIni = list(parameters_agent.values())
    # # # point_all = [np.array(pointIni) + (np.random.rand(len(pointIni)) * 2 - 1) * 0.005 * 0,
    # # #               np.array(pointIni) + (np.random.rand(len(pointIni)) * 2 - 1) * 0.005 * 0]
    # #
    # # log_likelihood_all = [-10 ** 6, -10 ** 6]
    # #
    # p_last = parameters_agent.copy()
    # #
    # pi = torch.ones(nq) / nq
    # onebox_temp = oneboxMDP(discount, nq, nr, na, p_last)
    # onebox_temp.setupMDP()
    # onebox_temp.solveMDP_sfm()
    # ThA = onebox_temp.ThA
    # softpolicy = onebox_temp.softpolicy

    #
    # oneboxHMM = HMMonebox(ThA, softpolicy, pi)
    # log_likelihood = oneboxHMM.log_likelihood(obs, ThA, softpolicy)
    # print(log_likelihood)
    # print(1)

    # use the built-in function to get gradient
    # para = [food_missed, app_rate, disapp_rate, food_consumed, push_button_cost, belief_diffusion, policy_temperature]
    # g = nd.Gradient(likelihood_gradient)
    # print(g(para))

    #print("gradient")
    ll = likelihood_tensor(food_missed, app_rate, disapp_rate, food_consumed, push_button_cost, belief_diffusion, policy_temperature,
                           obs, nq, nr, na, nl, discount)

    print(ll)
    #food_missed.retain_grad()
    #para = [food_missed, app_rate, disapp_rate, food_consumed, push_button_cost, belief_diffusion, policy_temperature]
    ll.backward()
    print(app_rate.grad,
          belief_diffusion.grad,
          disapp_rate.grad,
          food_consumed.grad,
          food_missed.grad,
          policy_temperature.grad,
          push_button_cost.grad
          )
    print(time.time() - start_time)


    print("finish")


if __name__ == "__main__":
    main()

