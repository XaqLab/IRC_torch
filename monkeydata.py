from twoboxTask.twobox_IRC_torch import *
import scipy.io as sio

pre_processed = True

# pre-process data
if not pre_processed:
    """
    import data
    """
    data = sio.loadmat('NeuralDatafromNeda/behav_valid.mat')
    sch = data['sch']
    binsize = 200

    """
    index of sessions without NaN values
    """
    idx_nonan = list(set([i for i in range(len(data['bLocY'][0])) if len(data['bLocY'][0][i])!= 0]) &
         set([i for i in range(len(data['bLocX'][0])) if len(data['bLocX'][0][i])!= 0]) &
         set([i for i in range(len(data['b1PushedTimes'][0])) if len(data['b1PushedTimes'][0][i])!= 0]) &
         set([i for i in range(len(data['b2PushedTimes'][0])) if len(data['b2PushedTimes'][0][i])!= 0]) &
         set([i for i in range(len(data['rew1DelTimes'][0])) if len(data['rew1DelTimes'][0][i])!= 0]) &
         set([i for i in range(len(data['rew2DelTimes'][0])) if len(data['rew2DelTimes'][0][i])!= 0]) &
         set([i for i in range(len(data['bLocX'][0])) if data['bLocX'][0][i].shape[1]!= 0])
        )

    """
    check which schedule appears most
    """
    schs = []
    for i in range(len(sch.squeeze()[idx_nonan])):
        sch_i = sch.squeeze()[idx_nonan][i].squeeze()
        if len(sch_i) == 3 and not np.isnan(sch_i[0]) and not np.isnan(sch_i[1]):
            schs.append(sch_i)
        if len(sch_i) == 6 and not np.isnan(sch_i[3]) and not np.isnan(sch_i[4]):
            schs.append(sch_i[:3])
            schs.append(sch_i[3:6])
        elif len(sch_i) == 9 and not np.isnan(sch_i[6]) and not np.isnan(sch_i[7]):
            schs.append(sch_i[:3])
            schs.append(sch_i[3:6])
            schs.append(sch_i[6:9])
        elif len(sch_i) == 12 and not np.isnan(sch_i[9]) and not np.isnan(sch_i[10]):
            schs.append(sch_i[:3])
            schs.append(sch_i[3:6])
            schs.append(sch_i[6:9])
            schs.append(sch_i[9:12])

    keys = set([(s[0], s[1]) for s in schs])
    keys_hist = [(key, sum([s[2] for s in schs if np.all(s[:2] == np.array(key))])) for key in keys]
    keys_hist.sort(key = lambda x:(x[1], -x[0][0]), reverse = True)
    target_schedule = keys_hist[0][0]

    idx_target = [ind for ind in idx_nonan
           if len(sch.squeeze()[ind][0]) ==3 and np.all(sch.squeeze()[ind][0, 0:2] == target_schedule)
           or len(sch.squeeze()[ind][0]) ==6 and (np.all(sch.squeeze()[ind][0, 0:2] == target_schedule)
                                                  or  np.all(sch.squeeze()[ind][0, 3:5] == target_schedule))
           or len(sch.squeeze()[ind][0]) ==9 and (np.all(sch.squeeze()[ind][0, 0:2] == target_schedule)
                                                  or np.all(sch.squeeze()[ind][0, 3:5] == target_schedule)
                                                  or np.all(sch.squeeze()[ind][0, 6:8] == target_schedule))
           or len(sch.squeeze()[ind][0]) ==12 and (np.all(sch.squeeze()[ind][0, 0:2] == target_schedule)
                                                  or np.all(sch.squeeze()[ind][0, 3:5] == target_schedule)
                                                  or np.all(sch.squeeze()[ind][0, 6:8] == target_schedule)
                                                  or np.all(sch.squeeze()[ind][0, 9:11] == target_schedule))]

    sess_use_matlab = 74

    sess_use = sess_use_matlab - 1
    b1PushedTimes = data['b1PushedTimes'][0][sess_use][0]
    b2PushedTimes = data['b2PushedTimes'][0][sess_use][0]

    rew1DelTimes = data['rew1DelTimes'][0][sess_use][0]
    rew2DelTimes = data['rew2DelTimes'][0][sess_use][0]

    bLocX = data['bLocX'][0][sess_use][0].astype(float)
    bLocY = data['bLocY'][0][sess_use][0].astype(float)

    """
    eliminate the bins without location information
    """
    b1PushedBins = np.rint(b1PushedTimes/binsize).astype(int)
    b2PushedBins = np.rint(b2PushedTimes/binsize).astype(int)
    rew1DelBins = np.rint(rew1DelTimes/binsize).astype(int)
    rew2DelBins = np.rint(rew2DelTimes/binsize).astype(int)

    b1PushedBins = b1PushedBins[np.where(b1PushedBins <= len(bLocX))]
    b2PushedBins = b2PushedBins[np.where(b2PushedBins <= len(bLocX))]
    rew1DelBins = rew1DelBins[np.where(rew1DelBins <= len(bLocX))]
    rew2DelBins = rew2DelBins[np.where(rew2DelBins <= len(bLocX))]


    rewDelBins = mergeArrays(rew1DelBins, rew2DelBins)
    rewDel_subsess = [rewDelBins[0:34], rewDelBins[34:100],
                      rewDelBins[100:134],rewDelBins[134:]] #rew_del with different schedules


    b1PushedBins_subsess = [[], [], [], []]
    b2PushedBins_subsess = [[], [], [], []]

    b1PushedBins_subsess_del = [[], [], [], []]
    b2PushedBins_subsess_del = [[], [], [], []]

    ind = 0
    for i in range(len(rewDel_subsess)):
        start = rewDel_subsess[i][0]
        end = rewDel_subsess[i][-1]

        # print(b1PushedBins[ind], start, end)
        while ind < len(b1PushedBins) and b1PushedBins[ind] <= end:
            # print('add', b1PushedBins[ind])
            b1PushedBins_subsess[i].append(b1PushedBins[ind])

            if (b1PushedBins[ind] in rewDel_subsess[i] or b1PushedBins[ind] + 1 in rewDel_subsess[i]
                    or b1PushedBins[ind] + 2 in rewDel_subsess[i]
                    or b1PushedBins[ind] + 3 in rewDel_subsess[i]
                    or b1PushedBins[ind] + 4 in rewDel_subsess[i]):
                b1PushedBins_subsess_del[i].append(b1PushedBins[ind])

            ind += 1

    ind = 0
    for i in range(len(rewDel_subsess)):
        start = rewDel_subsess[i][0]
        end = rewDel_subsess[i][-1]

        # print(b1PushedBins[ind], start, end)
        while ind < len(b2PushedBins) and b2PushedBins[ind] <= end:
            b2PushedBins_subsess[i].append(b2PushedBins[ind])

            if (b2PushedBins[ind] in rewDel_subsess[i]
                    or b2PushedBins[ind] + 1 in rewDel_subsess[i]
                    or b2PushedBins[ind] + 2 in rewDel_subsess[i]
                    or b2PushedBins[ind] + 3 in rewDel_subsess[i]
                    or b2PushedBins[ind] + 4 in rewDel_subsess[i]):
                b2PushedBins_subsess_del[i].append(b2PushedBins[ind])

            ind += 1

    print('Schedule of sub-session in the {}-th trial is: {}'.format(sess_use_matlab, sch.squeeze()[sess_use_matlab - 1]))

    print('Number of button pressing on box 1 in each session: {}'.format([len(i) for i in b1PushedBins_subsess]))
    print('Number of button pressing on box 2 in each session: {}'.format([len(i) for i in b2PushedBins_subsess]))

    print('Number of successful button pressing on box 1 in each session: {}'.format([len(i) for i in b1PushedBins_subsess_del]))
    print('Number of successful button pressing on box 2 in each session: {}'.format([len(i) for i in b2PushedBins_subsess_del]))

    pb_time = np.array(b1PushedBins_subsess[1])

    """
    plot the locations and pb times
    """
    # b1Pushed = np.zeros(len(bLocX))
    # b1Pushed[b1PushedBins] = 500
    # b2Pushed = np.zeros(len(bLocX))
    # b2Pushed[b2PushedBins] = 300
    # plt.plot(bLocX[:])
    # plt.plot(b1Pushed[:], 'r.')
    # plt.plot(b2Pushed[:], 'm.')
    # plt.plot(np.linspace(0, 65000), np.ones(np.linspace(0, 65000).shape) * np.nanmean(bLocX[list(b1PushedBins)]))
    # plt.plot(np.linspace(0, 65000), np.ones(np.linspace(0, 65000).shape) * (np.nanmean(bLocX[list(b1PushedBins)])
    #                                                                         + 3 *np.nanstd(bLocX[list(b1PushedBins)])))
    # plt.plot(np.linspace(0, 65000), np.ones(np.linspace(0, 65000).shape) * (np.nanmean(bLocX[list(b1PushedBins)])
    #                                                                         - 3 *np.nanstd(bLocX[list(b1PushedBins)])))
    # plt.show()

    """
    combine data with the same schedule from one session, 
    the list components indicate the starting and ending time of a timeframe,
    eliminate the first 10 trials after schedule change
    """
    bin_idx_range = []  # bin range of reward delivery
    removeFirst = 10
    start = 0
    trialBins = mergeArrays(rew1DelBins, rew2DelBins)

    if len(sch.squeeze()[sess_use][0]) >= 3:
        if np.all(sch.squeeze()[sess_use][0, 0:2] == target_schedule):
            valid_DelBins = trialBins[range(start + removeFirst, start + sch.squeeze()[sess_use][0, 2])]
            bin_idx_range.append([valid_DelBins[0], valid_DelBins[-1]])
            # print(valid_DelBins)
        start += sch.squeeze()[sess_use][0, 2]

    if len(sch.squeeze()[sess_use][0]) >= 6:
        if np.all(sch.squeeze()[sess_use][0, 3:5] == target_schedule):
            valid_DelBins = trialBins[range(start + removeFirst, start + sch.squeeze()[sess_use][0, 5])]
            bin_idx_range.append([valid_DelBins[0], valid_DelBins[-1]])
            # print(valid_DelBins)
        start += sch.squeeze()[sess_use][0, 5]

    if len(sch.squeeze()[sess_use][0]) >= 9:
        if np.all(sch.squeeze()[sess_use][0, 6:8] == target_schedule):
            valid_DelBins = trialBins[range(start + removeFirst, start + sch.squeeze()[sess_use][0, 8])]
            bin_idx_range.append([valid_DelBins[0], valid_DelBins[-1]])
            # print(valid_DelBins)
        start += sch.squeeze()[sess_use][0, 8]

    if len(sch.squeeze()[sess_use][0]) >= 12:
        if np.all(sch.squeeze()[sess_use][0, 9:11] == target_schedule):
            valid_DelBins = trialBins[range(start + removeFirst, start + sch.squeeze()[sess_use][0, 11])]
            bin_idx_range.append([valid_DelBins[0], valid_DelBins[-1]])
            # print(valid_DelBins)
        start += sch.squeeze()[sess_use][0, 11]


    """
    BOX1 locations based on actions, filter out the values that are nan or outliers
    """
    box1push = bLocX[b1PushedBins]
    mask1  = ~np.isnan(box1push)  # the elements that are not nan
    mask1[mask1] &= (box1push[mask1] > (np.nanmean(box1push) - 3 * np.nanstd(box1push)))  # elements that are not outlier
    invalid_idx1 = np.where(mask1 == False)[0]
    bLocX[b1PushedBins[invalid_idx1]] = np.nan # mark the elements that are either nan or outlier as "nan"

    """
    BOX2 locations, filter out the values that are nan or outliers
    """
    box2push = bLocX[b2PushedBins]
    mask2  = ~np.isnan(box2push)  # the elements that are not nan
    mask2[mask2] &= (box2push[mask2] > (np.nanmean(box2push) - 3 * np.nanstd(box2push)))  # elements that are not outlier
    invalid_idx2 = np.where(mask2 == False)[0]
    bLocX[b2PushedBins[invalid_idx2]] = np.nan # mark the elements that are either nan or outlier as "nan"

    """
    nan elements are filled by a smoothing filter that takes average of the previous three elements 
    """
    win_wid = 3
    bLocX_temp = np.concatenate((np.ones(win_wid) * np.nanmean(bLocX), bLocX))
    for i in np.where(np.isnan(bLocX) == True)[0]:
        bLocX_temp[i + win_wid] = np.mean(bLocX_temp[i:i + win_wid])

    bLocX = bLocX_temp[win_wid:]

    box1_boud = min(bLocX[b1PushedBins] )
    box2_boud = max(bLocX[b2PushedBins] )

    """
    discretize location as center(loc = 0), box1 (loc = 1) and box2(loc = 2)
    
    """
    locT = np.zeros(np.shape(bLocX), dtype = int)        # if at center location, loc = 0
    locT[np.where(bLocX < box2_boud)] = 2   # at box2 (left side)
    locT[np.where(bLocX > box1_boud)] = 1   # at box1 (right side)

    """
    actions
    """

    a0 = 0  # a0 = do nothing
    g0 = 1  # g0 = go to location 0
    g1 = 2  # g1 = go toward box 1 (via location 0 if from 2)
    g2 = 3  # g2 = go toward box 2 (via location 0 if from 1
    pb = 4  # pb  = push button

    actT = np.zeros(np.shape(bLocX), dtype=int)
    actT[b1PushedBins] = pb
    actT[b2PushedBins] = pb
    # actT[rew1DelBins - 1] = pb
    # actT[rew2DelBins-1] = pb

    for t in np.where(locT == 0)[0]:
        if t == len(bLocX) - 1:
            break
        if locT[t + 1] == 1:
            actT[t] = g1
        elif locT[t + 1] == 2:
            actT[t] = g2

    for t in np.where(locT == 1)[0]:
        if t == len(bLocX) - 1:
            break
        if locT[t + 1] != 1:
            actT[t] = g2

    for t in np.where(locT == 2)[0]:
        if t == len(bLocX) - 1:
            break
        if locT[t + 1] != 2:
            actT[t] = g1

    rew1DelBins_modified = []
    for i in b1PushedBins:
        if i in rew1DelBins or i + 1 in rew1DelBins or i + 2 in rew1DelBins or i + 3 in rew1DelBins or i + 4 in rew1DelBins or i + 5 in rew1DelBins:
            rew1DelBins_modified.append(i + 1)

    rew2DelBins_modified = []
    for i in b2PushedBins:
        if i in rew2DelBins or i + 1 in rew2DelBins or i + 2 in rew2DelBins or i + 3 in rew2DelBins or i + 4 in rew2DelBins or i + 5 in rew2DelBins:
            rew2DelBins_modified.append(i + 1)

    """
    rewards
    """
    rewT = np.zeros(np.shape(bLocX), dtype = int)
    rewT[rew1DelBins_modified] = 1
    rewT[rew2DelBins_modified] = 1

    lower1 = bin_idx_range[0][0]
    upper1 = bin_idx_range[0][1]
    bin_idx1 = np.arange(lower1, upper1)

    fig1, (ax1, ax2) = plt.subplots(2,1, figsize = (10,6))
    ax1.plot(bin_idx1, bLocX[bin_idx1], 'b')
    ax1.plot(b1PushedBins[(b1PushedBins >= lower1) & (b1PushedBins <= upper1)],
             bLocX[b1PushedBins[(b1PushedBins >= lower1) & (b1PushedBins <= upper1)]], 'r.')
    ax1.plot(b2PushedBins[(b2PushedBins >= lower1) & (b2PushedBins <= upper1)],
             bLocX[b2PushedBins[(b2PushedBins >= lower1) & (b2PushedBins <= upper1)]], 'm.')
    #ax1.plot(bin_idx1, locT[bin_idx1] * 200, 'g*')
    ax1.plot(np.linspace(lower1, upper1), np.ones(np.linspace(lower1, upper1).shape) * box1_boud)
    ax1.plot(np.linspace(lower1, upper1), np.ones(np.linspace(lower1, upper1).shape) * box2_boud)
    ax1.yaxis.tick_right()
    ax1.set_yticks([box2_boud, box1_boud])
    labels = [item.get_text() for item in ax1.get_yticklabels()]
    labels[1] = 'box1'
    labels[0] = 'box2'
    ax1.set_yticklabels(labels)
    ax1.set_title('behavior in the 1st sub-session')
    ax1.set_ylabel('location (along the cage)')

    lower2 = bin_idx_range[1][0]
    upper2 = bin_idx_range[1][1]
    bin_idx2 = np.arange(lower2, upper2)

    ax2.plot(bin_idx2, bLocX[bin_idx2], 'b', label = 'location' )
    ax2.plot(b1PushedBins[(b1PushedBins >= lower2) & (b1PushedBins <= upper2)],
             bLocX[b1PushedBins[(b1PushedBins >= lower2) & (b1PushedBins <= upper2)]], 'r.', label = 'open box 1')
    ax2.plot(b2PushedBins[(b2PushedBins >= lower2) & (b2PushedBins <= upper2)],
             bLocX[b2PushedBins[(b2PushedBins >= lower2) & (b2PushedBins <= upper2)]], 'm.', label = 'open box 2')
    ax2.plot(np.linspace(lower2, upper2), np.ones(np.linspace(lower2, upper2).shape) * box1_boud)
    ax2.plot(np.linspace(lower2, upper2), np.ones(np.linspace(lower2, upper2).shape) * box2_boud)
    ax2.yaxis.tick_right()
    ax2.set_yticks([box2_boud, box1_boud])
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    labels[1] = 'box1'
    labels[0] = 'box2'
    ax2.set_yticklabels(labels)
    ax2.legend(loc = 'upper right')

    ax2.set_title('behavior in the 2nd sub-session')
    ax2.set_ylabel('location (along the cage)')
    ax2.set_xlabel('time')
    plt.show()

else:
    path = os.getcwd()
    dataN_pkl_file = open(path + '/twoboxTask/Data/monkey_twobox_preprocessed.pkl', 'rb')
    dataN_pkl = pickle.load(dataN_pkl_file)
    dataN_pkl_file.close()

    actT = dataN_pkl['actions']
    rewT = dataN_pkl['rewards']
    locT = dataN_pkl['locations']
    bin_idx_range = dataN_pkl['bin_idx_range']

actTT = np.hstack((actT[range(bin_idx_range[1][0], 48000)]))
rewTT = np.hstack((rewT[range(bin_idx_range[1][0], 48000)]))
locTT = np.hstack((locT[range(bin_idx_range[1][0], 48000)]))
obsTT = (np.vstack([actTT, rewTT, locTT]).T)

T = obsTT.shape[0]
obs = obsTT[:T, :]
act = obs[:, 0]
rew = obs[:, 1]
loc = obs[:, 2]

obsN = np.expand_dims(obs, axis=0)
"""
IRC
"""
nq = 10
nr = 2
nl = 3
na = 5
discount = torch.tensor([.99], dtype=torch.float64)

sample_length = len(obs)
sample_number = 1

# app_rate1_ini = 0.0342 #0.10385418
# disapp_rate1_ini = 0.0187 #.01
# app_rate2_ini = 0.0356 #0.19612859
# disapp_rate2_ini =  0.0028 #.01
# food_missed_ini = 0.1085 #.1  # 0
# food_consumed_ini = 0.8391 #.9  # 1
# belief_diffusion_ini = 0.0325 #.1 #0.03827322 # .1
# policy_temperature_ini = 0.0792 #.06  #0.15768841  # .06
# push_button_cost_ini = 0.3566 #.2 #0.40363519  # .2
# grooming_reward_ini = -0.0052 #.3 #0.20094441  # .3
# travel_cost_ini = 0.3523 #.1 #0.3033027  # .1
# trip_prob_ini = 0.0447 #.05
# direct_prob_ini = 0.2186 #0.19538837  # .1
app_rate1_ini = 0.0577
disapp_rate1_ini = .0232
app_rate2_ini = 0.0859
disapp_rate2_ini = .0182
food_missed_ini = .1193  # 0
food_consumed_ini = .9331  # 1
belief_diffusion_ini = .1380  # 0.03827322 # .1
policy_temperature_ini = .2034  # 0.15768841  # .06
push_button_cost_ini = .3244  # 0.40363519  # .2
grooming_reward_ini = .0130  # 0.20094441  # .3
travel_cost_ini = .3951  # 0.3033027  # .1
trip_prob_ini = .0453
direct_prob_ini = 0.1954  # .1

app_rate1_ini = torch.autograd.Variable(torch.tensor([app_rate1_ini], dtype=torch.float64), requires_grad=True)
disapp_rate1_ini = torch.autograd.Variable(torch.tensor([disapp_rate1_ini], dtype=torch.float64), requires_grad=True)
app_rate2_ini = torch.autograd.Variable(torch.tensor([app_rate2_ini], dtype=torch.float64), requires_grad=True)
disapp_rate2_ini = torch.autograd.Variable(torch.tensor([disapp_rate2_ini], dtype=torch.float64), requires_grad=True)
food_missed_ini = torch.autograd.Variable(torch.tensor([food_missed_ini], dtype=torch.float64), requires_grad=True)  # 0
food_consumed_ini = torch.autograd.Variable(torch.tensor([food_consumed_ini], dtype=torch.float64), requires_grad=True)  # .99 #1
belief_diffusion_ini = torch.autograd.Variable(torch.tensor([belief_diffusion_ini], dtype=torch.float64), requires_grad=True)  # .1
policy_temperature_ini = torch.autograd.Variable(torch.tensor([policy_temperature_ini], dtype=torch.float64), requires_grad=True)  # .061
push_button_cost_ini = torch.autograd.Variable(torch.tensor([push_button_cost_ini],dtype=torch.float64),  requires_grad=True)  # .3
grooming_reward_ini = torch.autograd.Variable(torch.tensor([grooming_reward_ini],dtype=torch.float64), requires_grad=True)  # .3
travel_cost_ini = torch.autograd.Variable(torch.tensor([travel_cost_ini], dtype=torch.float64), requires_grad=True)  # .3
direct_prob_ini = torch.autograd.Variable(torch.tensor([direct_prob_ini],dtype=torch.float64),  requires_grad=True)  # .3
trip_prob_ini = torch.autograd.Variable(torch.tensor([trip_prob_ini], dtype=torch.float64), requires_grad=True)  # .3

point_ini = collections.OrderedDict()
point_ini['food_missed'] = food_missed_ini
point_ini['app_rate1'] = app_rate1_ini
point_ini['disapp_rate1'] = disapp_rate1_ini
point_ini['app_rate2'] = app_rate2_ini
point_ini['disapp_rate2'] = disapp_rate2_ini
point_ini['food_consumed'] = food_consumed_ini
point_ini['push_button_cost'] = push_button_cost_ini
point_ini['belief_diffusion'] = belief_diffusion_ini
point_ini['policy_temperature'] = policy_temperature_ini
point_ini['direct_prob'] = direct_prob_ini
point_ini['trip_prob'] = trip_prob_ini
point_ini['grooming_reward'] = grooming_reward_ini
point_ini['travel_cost'] = travel_cost_ini

LR = 10**-6*1
EPS = .1 #0.1
BATCH_SIZE = 1

# point_traj_set = []
# log_likelihood_traj_set = []
# for _ in range(10):
#     point_ini_p = point_ini.copy()
#     for k, v in point_ini_p.items():
#         point_ini_p[k] = v.clone().detach()
#         point_ini_p[k] = point_ini_p[k] * ( (2 * torch.rand(1) - 1) / 5 + 1)
#         point_ini_p[k].requires_grad = True
#     IRC_monkey = twobox_IRC_torch(discount, nq, nr, na, nl, point_ini_p)
#     IRC_monkey.IRC_batch(obsN[:, :1000, :], lr=LR, eps=EPS, batch_size=BATCH_SIZE, shuffle=True)
#     point_traj_set.append(IRC_monkey.point_traj)
#     log_likelihood_traj_set.append(IRC_monkey.log_likelihood_traj)

IRC_monkey = twobox_IRC_torch(discount, nq, nr, na, nl, point_ini)
IRC_monkey.IRC_batch(obsN[:, :1000, :], lr=LR, eps=EPS, batch_size=BATCH_SIZE, shuffle=True)
IRC_monkey.contour_LL(obsN)
IRC_monkey.plot_contour_LL()
print(1)
