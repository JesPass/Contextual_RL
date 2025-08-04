import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import copy
import pingouin as pg
import statsmodels.api as sm
from sklearn import metrics
from matplotlib.legend_handler import HandlerTuple

mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'Myriad Pro'
mpl.rcParams['font.sans-serif'] = 'Myriad Pro'


def anova(stats_me, statistic):
    stats_out = pg.anova(data=stats_me, dv=statistic, between='alg_type')
    post_hoc_out = pg.pairwise_tukey(data=stats_me, dv=statistic, between='alg_type')
    return stats_out, post_hoc_out


def t_test(stats_me, statistic):
    x_int = stats_me[stats_me['alg_type'] == 'SR']
    y_int = stats_me[stats_me['alg_type'] == 'joint_inf_priors']
    x = x_int[statistic]
    y = y_int[statistic]

    post_hoc_out = pg.ttest(x, y)
    return post_hoc_out


def anova_mixed_effects(stats_me, statistic):
    stats_out = pg.mixed_anova(dv=statistic, between='alg_type', within='len', subject='id_here', data=stats_me)
    post_hoc_out = pg.pairwise_tests(dv=statistic, between='alg_type', within='len', subject='id_here', data=stats_me,
                                     padjust='bonf', parametric=True)
    return stats_out, post_hoc_out


def non_param_anova(stats_me, statistic):
    stats_out = pg.welch_anova(data=stats_me, dv=statistic, between='alg_type')
    post_hoc_out = pg.pairwise_gameshowell(data=stats_me, dv=statistic, between='alg_type')
    return stats_out, post_hoc_out


def regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    return results


def do_maintenance(data, len_here, alg_type, task1_dfs, task2_dfs, noise, id_here):
    mean_probs_task2 = []
    mean_probs_task1 = []

    loc = 'cue'
    typ = 'rand'
    correct_runs = True

    if noise:
        len_here[0] = 2

    if typ == 'start':
        num_agents = len(data[32])

    if typ == 'end':
        num_agents = len(data[35])

    if typ == 'rand':
        num_agents = len(data[28])

    for i in np.arange(num_agents):
        # select one agent implementation

        if typ == 'start':
            num_attempts = data[32][i]
            probs = np.array(data[33][i])
            states = np.array(data[34][i])

        if typ == 'end':
            num_attempts = data[35][i]
            probs = np.array(data[36][i])
            states = np.array(data[37][i])

        if typ == 'rand':
            num_attempts = data[28][i]
            probs = np.array(data[30][i])
            states = np.array(data[31][i])

        task_1 = []
        task_2 = []

        # select correct trials
        if correct_runs:
            cor_trials = np.where(num_attempts == 1)[0] + 1
            probs_corr = probs[cor_trials]
            state_corr = states[cor_trials]

            for i in np.arange(len(cor_trials)):
                if state_corr[i][-1][-1] == 5:
                    task_1.append(i)
                elif state_corr[i][-1][-1] == 8:
                    task_2.append(i)
        else:
            cor_trials = np.where(num_attempts != 1)[0] + 1  # select the trials that contain incorrect trials
            probs_corr = probs[cor_trials]
            state_corr = states[cor_trials]

            for i in np.arange(len(cor_trials)):
                if state_corr[i][-1][-1] == 8:
                    task_1.append(i)
                elif state_corr[i][-1][-1] == 5:
                    task_2.append(i)

        # select state after cue on each correct trial
        prob_agent = []
        for j in task_1:
            if len_here[0] == 2:
                sel = 2
                states = [2]

                if not correct_runs:
                    # select index of states between cue and end on last trial
                    goal_chunks = len(state_corr[j]) - 1
                    for goal_ind in np.arange(goal_chunks):
                        goal_probs = probs_corr[j][goal_ind][:]
                        goal_states = state_corr[j][goal_ind][:]

                        second_last = np.where(np.array(goal_states) == sel)[0][-1]
                        states_in_q = goal_states[second_last:]
                        prob_in_q = np.array(goal_probs)[second_last:]

                        task_id = np.argmax(prob_in_q[0])  # task id is maximum probability after cue is seen

                        state_probs = []  # state_probs is average probability in state on trial
                        for sel_state in states:
                            loc = np.where(np.array(states_in_q) == sel_state)[0]
                            looper = []
                            for cur in loc:
                                looper.append(prob_in_q[cur][task_id])
                            prob = np.mean(looper)
                            state_probs.append(prob)

                        prob_agent.append(state_probs)
                else:
                    second_last = np.where(np.array(state_corr[j][-1][:]) == sel)[0][-1]
                    states_in_q = state_corr[j][-1][second_last:]
                    prob_in_q = np.array(probs_corr[j][-1])[second_last:]

                    task_id = np.argmax(prob_in_q[0])  # task id is maximum probability after cue is seen

                    state_probs = []  # state_probs is average probability in state on trial
                    for sel_state in states:
                        loc = np.where(np.array(states_in_q) == sel_state)[0]
                        looper = []
                        for cur in loc:
                            looper.append(prob_in_q[cur][task_id])
                        prob = np.mean(looper)
                        state_probs.append(prob)

                    prob_agent.append(state_probs)

            else:
                sel = 9
                list_len = []
                for i in np.arange(len_here[0] - 2):
                    dist = 9 + i
                    list_len.append(dist)
                states = list_len + [2]

                if not correct_runs:
                    # select index of states between cue and end on last trial
                    goal_chunks = len(state_corr[j]) - 1
                    for goal_ind in np.arange(goal_chunks):
                        goal_probs = probs_corr[j][goal_ind][:]
                        goal_states = state_corr[j][goal_ind][:]

                        second_last = np.where(np.array(goal_states) == sel)[0][-1]
                        states_in_q = goal_states[second_last:]
                        prob_in_q = np.array(goal_probs)[second_last:]

                        task_id = np.argmax(prob_in_q[0])  # task id is maximum probability after cue is seen

                        state_probs = []  # state_probs is average probability in state on trial
                        for sel_state in states:
                            loc = np.where(np.array(states_in_q) == sel_state)[0]
                            looper = []
                            for cur in loc:
                                looper.append(prob_in_q[cur][task_id])
                            prob = np.mean(looper)
                            state_probs.append(prob)

                        prob_agent.append(state_probs)

                else:
                    # select index of states between cue and end on last trial
                    second_last = np.where(np.array(state_corr[j][-1][:]) == sel)[0][-1]
                    states_in_q = state_corr[j][-1][second_last:]
                    prob_in_q = np.array(probs_corr[j][-1])[second_last:]

                    task_id = np.argmax(prob_in_q[0])  # task id is maximum probability after cue is seen

                    state_probs = []  # state_probs is average probability in state on trial
                    for sel_state in states:
                        loc = np.where(np.array(states_in_q) == sel_state)[0]
                        looper = []
                        for cur in loc:
                            looper.append(prob_in_q[cur][task_id])
                        prob = np.mean(looper)
                        state_probs.append(prob)

                    prob_agent.append(state_probs)

        past_mp1 = copy.deepcopy(mean_probs_task1)
        mini_mean = np.mean(prob_agent, axis=0)
        if (np.isnan([mini_mean]).any()):
            # NA values occur because there isn't any incorrect trials in that agent
            mini_mean = np.array([np.nan])
        mean_probs_task1.append(mini_mean)

        prob_agent = []
        for j in task_2:
            if len_here[0] == 2:
                sel = 2
                states = [2]

                if not correct_runs:
                    # select index of states between cue and end on last trial
                    goal_chunks = len(state_corr[j]) - 1
                    for goal_ind in np.arange(goal_chunks):
                        goal_probs = probs_corr[j][goal_ind][:]
                        goal_states = state_corr[j][goal_ind][:]

                        second_last = np.where(np.array(goal_states) == sel)[0][-1]
                        states_in_q = goal_states[second_last:]
                        prob_in_q = np.array(goal_probs)[second_last:]

                        task_id = np.argmax(prob_in_q[0])  # task id is maximum probability after cue is seen

                        state_probs = []  # state_probs is average probability in state on trial
                        for sel_state in states:
                            loc = np.where(np.array(states_in_q) == sel_state)[0]
                            looper = []
                            for cur in loc:
                                looper.append(prob_in_q[cur][task_id])
                            prob = np.mean(looper)
                            state_probs.append(prob)

                        prob_agent.append(state_probs)
                else:
                    second_last = np.where(np.array(state_corr[j][-1][:]) == sel)[0][-1]
                    states_in_q = state_corr[j][-1][second_last:]
                    prob_in_q = np.array(probs_corr[j][-1])[second_last:]

                    task_id = np.argmax(prob_in_q[0])  # task id is maximum probability after cue is seen

                    state_probs = []  # state_probs is average probability in state on trial
                    for sel_state in states:
                        loc = np.where(np.array(states_in_q) == sel_state)[0]
                        looper = []
                        for cur in loc:
                            looper.append(prob_in_q[cur][task_id])
                        prob = np.mean(looper)
                        state_probs.append(prob)

                    prob_agent.append(state_probs)

            else:
                sel = 9
                list_len = []
                for i in np.arange(len_here[0] - 2):
                    dist = 9 + i
                    list_len.append(dist)
                states = list_len + [2]
                if not correct_runs:
                    # select index of states between cue and end on last trial
                    goal_chunks = len(state_corr[j]) - 1
                    for goal_ind in np.arange(goal_chunks):
                        goal_probs = probs_corr[j][goal_ind][:]
                        goal_states = state_corr[j][goal_ind][:]

                        second_last = np.where(np.array(goal_states) == sel)[0][-1]
                        states_in_q = goal_states[second_last:]
                        prob_in_q = np.array(goal_probs)[second_last:]

                        task_id = np.argmax(prob_in_q[0])  # task id is maximum probability after cue is seen

                        state_probs = []  # state_probs is average probability in state on trial
                        for sel_state in states:
                            loc = np.where(np.array(states_in_q) == sel_state)[0]
                            looper = []
                            for cur in loc:
                                looper.append(prob_in_q[cur][task_id])
                            prob = np.mean(looper)
                            state_probs.append(prob)

                        prob_agent.append(state_probs)
                else:
                    # select index of states between cue and end on last trial
                    second_last = np.where(np.array(state_corr[j][-1][:]) == sel)[0][-1]
                    states_in_q = state_corr[j][-1][second_last:]
                    prob_in_q = np.array(probs_corr[j][-1])[second_last:]

                    task_id = np.argmax(prob_in_q[0])  # task id is maximum probability after cue is seen

                    state_probs = []  # state_probs is average probability in state on trial
                    for sel_state in states:
                        loc = np.where(np.array(states_in_q) == sel_state)[0]
                        looper = []
                        for cur in loc:
                            looper.append(prob_in_q[cur][task_id])
                        prob = np.mean(looper)
                        state_probs.append(prob)

                    prob_agent.append(state_probs)

        mini_mean = np.mean(prob_agent, axis=0)
        if (np.isnan([mini_mean]).any()):
            # NA values occur because there isn't any incorrect trials in that agent
            mini_mean = np.array([np.nan])
        mean_probs_task2.append(mini_mean)

    col_names_1 = []
    col_names_2 = []
    for i in np.arange(len_here[0] - 1):
        name = i
        name_2 = i
        col_names_1.append(name)
        col_names_2.append(name_2)

    len_here[0] = len_here[1]
    mean_vals = (np.array(mean_probs_task1) + np.array(mean_probs_task2)) / 2
    task1_df = pd.DataFrame(mean_vals, columns=col_names_1)
    task1_df['alg_type'] = alg_type
    task1_df['len'] = len_here

    task1_df['id_here'] = np.array(id_here)

    task2_df = pd.DataFrame(mean_probs_task2, columns=col_names_2)
    task2_df['alg_type'] = alg_type
    task2_df['len'] = len_here

    task2_df['id_here'] = np.array(id_here) * 2

    task1_dfs.append(task1_df)
    task2_dfs.append(task2_df)

    return task1_dfs, task2_dfs


def pickout_mean_SR(trials, cell, loc_inf, states, probs, task, task1_id, SRs):
    if task == 0:
        prob_agent = []
        for j in trials:
            # pick out probabilities on last visit to loc_inf
            loc = np.where(np.array(states[j][-1]) == loc_inf)[0]
            loc_1 = loc[-1]
            prob = np.array(probs[j][-1])[loc_1]
            prob_agent.append(prob)
        # find task_id
        mean_probs = np.mean(prob_agent, axis=0)
        task1_id = np.argmax(mean_probs)

    # find average SR for that task
    rel_SRs = []
    rel_SR = []
    for j in trials:
        for i in np.arange(len(SRs[j])):  # attempt per trial
            for k in np.arange(len(SRs[j][i])):  # at every step
                rel_SR = SRs[j][i][k][task1_id][:, cell]
                rel_SRs.append(np.copy(rel_SR))
    mean_SR = np.mean(rel_SRs, axis=0)

    return mean_SR, task1_id


def pull_out_learning_SRs(data):
    typ = 'start'
    all_SRs_1 = []
    all_SRs_2 = []
    loc_inf = 2  # for infered task id is choice
    cell = 0  # look at predictions given start of map

    if typ == 'start':
        num_agents = len(data[32])

    for i in np.arange(num_agents):
        # select one agent implementation

        if typ == 'start':
            num_attempts = data[32][i]
            probs = np.array(data[33][i])
            states = np.array(data[34][i])
            SRs = np.array(data[38][i])

        # last 10 trials first block
        task = 0
        task1_id = []
        trials = np.arange(39, 50)
        mean_SR, task1_id = pickout_mean_SR(trials, cell, loc_inf, states, probs, task, task1_id, SRs)
        all_SRs_1.append(np.copy(mean_SR))

        # last 10 trials second block
        task = 1
        trials = np.arange(89, 100)
        mean_SR_2, task1_id = pickout_mean_SR(trials, cell, loc_inf, states, probs, task, task1_id, SRs)
        all_SRs_2.append(np.copy(mean_SR_2))

    return all_SRs_1, all_SRs_2


def pull_out_learning(data):
    # number of trials on first, second reversal
    # number of incorrect updates following first v second reversal

    revs_1_list = []
    revs_2_list = []

    all_steps_1 = []
    all_steps_2 = []

    typ = 'start'

    if typ == 'start':
        num_agents = len(data[32])

    if typ == 'end':
        num_agents = len(data[35])

    if typ == 'rand':
        num_agents = len(data[28])

    for i in np.arange(num_agents):
        # select one agent implementation

        if typ == 'start':
            num_attempts = data[32][i]
            probs = np.array(data[33][i])
            states = np.array(data[34][i])

        if typ == 'end':
            num_attempts = data[35][i]
            probs = np.array(data[36][i])
            states = np.array(data[37][i])

        if typ == 'rand':
            num_attempts = data[28][i]
            probs = np.array(data[30][i])
            states = np.array(data[31][i])

        revs_1 = sum(num_attempts[50:60])  # from task 1 to task 2
        revs_2 = num_attempts[50]  # from task 2 to task 1

        revs_1_list.append(revs_1)
        revs_2_list.append(revs_2)

        # pick out most inferred task at choice in last 10 trials on first block
        prob_agent = []
        for j in np.arange(39, 50):
            loc = np.where(np.array(states[j][-1]) == 2)[0]
            loc_1 = loc[-1]
            prob = np.array(probs[j][-1])[loc_1]
            prob_agent.append(prob)

        mean_probs = np.mean(prob_agent, axis=0)
        task1_id = np.argmax(mean_probs)

        # amount of updates of that map on next set of trials
        steps_1 = 0
        for j in np.arange(50, 100):
            all_probs_trial = probs[j]
            for i in np.arange(len(all_probs_trial)):
                trial_probs = all_probs_trial[i]
                for k in np.arange(len(trial_probs)):
                    prob_at_state = trial_probs[k]
                    if np.argmax(prob_at_state) == task1_id:
                        steps_1 = steps_1 + 1

        all_steps_1.append(steps_1)

        prob_agent = []
        for j in np.arange(39 + 50, 50 + 50):
            loc = np.where(np.array(states[j][-1]) == 2)[0]
            loc_1 = loc[-1]
            prob = np.array(probs[j][-1])[loc_1]
            prob_agent.append(prob)

        mean_probs = np.mean(prob_agent, axis=0)
        task1_id = np.argmax(mean_probs)

        # amount of updates of that map on next set of trials
        steps_2 = 0
        for j in np.arange(50 + 50, 100 + 50):
            all_probs_trial = probs[j]
            for i in np.arange(len(all_probs_trial)):
                trial_probs = all_probs_trial[i]
                for k in np.arange(len(trial_probs)):
                    prob_at_state = trial_probs[k]
                    if np.argmax(prob_at_state) == task1_id:
                        steps_2 = steps_2 + 1

        all_steps_2.append(steps_2)

    return all_steps_1, all_steps_2, revs_1_list, revs_2_list


def distance_perf(data, limit, all_count_1, all_count_2, len_here):
    # number of trials on first, second reversal
    # number of incorrect updates following first v second reversal

    revs_1_list = []
    revs_2_list = []

    num_agents = len(data[39])
    for i in np.arange(num_agents):
        reversals = data[39][i]

        per_pre = len(np.where(np.array(reversals[(limit - 100):(limit)]) == 1)[0])
        per_post = len(np.where(np.array(reversals[(limit):(limit + 100)]) == 1)[0])

        revs_1_list.append(per_pre)
        revs_2_list.append(per_post)

    control_reversals = []
    control_reversals_post = []

    num_agents = len(data[39])
    for i in np.arange(num_agents):
        revs_3_list = []
        revs_4_list = []
        reversals = data[39][i]

        for limit_here in [100, 200, 500, 750, 1000, 1250, 1500]:
            per_pre = len(np.where(np.array(reversals[(limit_here - 100):(limit_here)]) == 1)[0])
            per_post = len(np.where(np.array(reversals[(limit_here):(limit_here + 100)]) == 1)[0])

            revs_3_list.append(per_pre)
            revs_4_list.append(per_post)
        control_reversals.append(revs_3_list)
        control_reversals_post.append(revs_4_list)

    control_1 = pd.DataFrame(control_reversals, columns=[100, 200, 500, 750, 1000, 1250, 1500])
    control_2 = pd.DataFrame(control_reversals_post, columns=[100, 200, 500, 750, 1000, 1250, 1500])
    control_1['len'] = len_here
    control_2['len'] = len_here
    control_1['limit'] = limit
    control_2['limit'] = limit
    all_count_1.append(control_1)
    all_count_2.append(control_2)

    return revs_1_list, revs_2_list, all_count_1, all_count_2


def distance_perf_out(data, limit, all_count_1, all_count_2, len_here):
    # number of trials on first, second reversal
    # number of incorrect updates following first v second reversal

    revs_1_list = []
    revs_2_list = []

    num_agents = len(data[39])
    for i in np.arange(num_agents):
        reversals = np.array(data[39][i])

        per_pre = (reversals[limit - 50] + reversals[limit - 100]) / 2
        per_post = (reversals[limit] + reversals[limit + 50]) / 2

        revs_1_list.append(per_pre)
        revs_2_list.append(per_post)

    control_reversals = []
    control_reversals_post = []

    num_agents = len(data[39])
    for i in np.arange(num_agents):
        revs_3_list = []
        revs_4_list = []
        reversals = data[39][i]

        for limit_here in [100, 200, 500, 750, 1000]:
            per_pre = (reversals[limit_here - 50] + reversals[limit_here - 100]) / 2
            per_post = (reversals[limit_here + 0] + reversals[limit_here + 50]) / 2

            revs_3_list.append(per_pre)
            revs_4_list.append(per_post)
        control_reversals.append(revs_3_list)
        control_reversals_post.append(revs_4_list)

    control_1 = pd.DataFrame(control_reversals, columns=[100, 200, 500, 750, 1000])
    control_2 = pd.DataFrame(control_reversals_post, columns=[100, 200, 500, 750, 1000])
    control_1['len'] = len_here
    control_2['len'] = len_here
    control_1['limit'] = limit
    control_2['limit'] = limit
    all_count_1.append(control_1)
    all_count_2.append(control_2)

    return revs_1_list, revs_2_list, all_count_1, all_count_2


def split_maintain(data, len_here, alg_type, task1_dfs, task2_dfs, id_here):
    list_maint = []
    for listed in np.arange(len_here[0] - 2):
        list_maint.append(9 + listed)
    list_maint.append(2)

    split_1 = []
    split_2 = []
    num_agents = len(data[28])
    for i in np.arange(len(data[28])):
        # select one agent implementation
        num_attempts = data[28][i]
        SRs = np.array(data[29][i])
        probs = np.array(data[30][i])
        states = np.array(data[31][i])

        # select correct trials
        cor_trials = np.where(num_attempts == 1)[0] + 1
        SRs_corr = SRs[cor_trials]
        probs_corr = probs[cor_trials]
        state_corr = states[cor_trials]

        task_1 = []
        task_2 = []
        mean_probs_task1 = []
        mean_probs_task2 = []

        per_agent_rep_1 = []
        per_agent_rep_2 = []

        for i in np.arange(len(cor_trials)):
            if state_corr[i][-1][-1] == 5:
                task_1.append(i)
            elif state_corr[i][-1][-1] == 8:
                task_2.append(i)

        for loc in list_maint:
            prob_agent = []
            for j in task_1:
                loc_int = np.where(np.array(state_corr[j][-1]) == loc)[0]
                loc_1 = loc_int[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                prob_agent.append(prob)
            mean_probs_task1 = np.mean(prob_agent, axis=0)

            prob_agent = []
            for j in task_2:
                loc_int = np.where(np.array(state_corr[j][-1]) == loc)[0]
                loc_1 = loc_int[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                prob_agent.append(prob)
            mean_probs_task2 = np.mean(prob_agent, axis=0)

            # probabilities at choice
            task_id_1 = np.argmax(mean_probs_task1)
            task_id_2 = np.argmax(mean_probs_task2)
            per_agent_rep_1.append(
                mean_probs_task1[task_id_1] - mean_probs_task2[task_id_1])  # appends splitter probs to a list
            per_agent_rep_2.append(mean_probs_task2[task_id_2] - mean_probs_task1[task_id_2])

        split_1.append(per_agent_rep_1)
        split_2.append(per_agent_rep_2)

    col_names_1 = []
    col_names_2 = []
    for i in np.arange(len_here[0] - 1):
        name = i
        name_2 = i
        col_names_1.append(name)
        col_names_2.append(name_2)

    task1_df = pd.DataFrame((np.array(split_1) + np.array(split_2)) / 2, columns=col_names_1)
    task1_df['alg_type'] = alg_type
    task1_df['len'] = len_here
    task1_df['id_here'] = id_here

    task2_df = pd.DataFrame(split_2, columns=col_names_2)
    task2_df['alg_type'] = alg_type
    task2_df['len'] = len_here
    task1_df['id_here'] = id_here

    task1_dfs.append(task1_df)
    task2_dfs.append(task2_df)
    return task1_dfs, task2_dfs


def do_splitting_fancy(data, len_here, loc, cell, noise, task1_dfs, task2_dfs, task3_dfs, task4_dfs, filename, keys,
                       alg):
    # pulls out the trial-averaged activity for each agent
    split = True

    mean_probs_task2 = []
    mean_cell_task2 = []

    mean_probs_task1 = []
    mean_cell_task1 = []

    if noise:
        len_here[0] = 2

    if loc == 'post_cue':  # post-cue for splitter cell cells
        if len_here[0] == 2:
            sel = 2
        else:
            sel = 9
    if loc == 'choice':
        sel = 2
    if cell == 'start':
        sel_cell = 0
    if cell == 'post_cue':
        if len_here[0] == 2:
            sel_cell = 2
        else:
            sel_cell = 9
    if cell == 'choice':
        sel_cell = 2
    if cell == 'at_cue':  # for over/underrepresentation of cue
        sel_cell = 1

    for i in np.arange(len(data[28])):
        # select one agent implementation
        num_attempts = data[28][i]
        SRs = np.array(data[29][i])
        probs = np.array(data[30][i])
        states = np.array(data[31][i])

        # select correct trials
        cor_trials = np.where(num_attempts == 1)[0] + 1
        SRs_corr = SRs[cor_trials]
        probs_corr = probs[cor_trials]
        state_corr = states[cor_trials]

        task_1 = []
        task_2 = []

        for i in np.arange(len(cor_trials)):
            if state_corr[i][-1][-1] == 5:
                task_1.append(i)
            elif state_corr[i][-1][-1] == 8:
                task_2.append(i)

        # select state after cue on each correct trial
        prob_agent = []
        SR_across_trials = []
        for j in task_1:
            if len_here[0] == 2:
                loc = np.where(np.array(state_corr[j][-1]) == 2)[0]
                loc_1 = loc[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                SR_list = []
                for task in np.arange(10):
                    SR_per_task = []
                    for cur in [loc_1]:
                        SR = SRs_corr[j][-1][cur][task][:, sel_cell]
                        chosen_SR = SRs_corr[j][-1][cur][task]

                        n_sts = len(SR)
                        state0 = np.zeros((n_sts, 1))
                        state1 = np.zeros((n_sts, 1))
                        state2 = np.zeros((n_sts, 1))
                        state0[0] = 1
                        state1[1] = 1
                        state1[-2] = 4
                        state2[2] = 1

                        if noise:
                            for lengths_in_way in np.arange(len_here[1] - 2):
                                state1[9 + lengths_in_way] = 1

                        # (current_SR @ self.state - self.discount_factor * (current_SR @ state_new)
                        state = state0
                        state_new = state1
                        predicted_state1 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        state = state1
                        state_new = state2
                        predicted_state2 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        if cell == 'at_cue':
                            SR_per_task.append(predicted_state1)
                        else:
                            SR_per_task.append(SR)

                    trial_ave_SR = np.mean(SR_per_task, axis=0)
                    SR_list.append(trial_ave_SR)
            else:
                loc = np.where(np.array(state_corr[j][-1]) == sel)[0]
                loc_1 = loc[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                SR_list = []
                for task in np.arange(10):
                    SR_per_task = []
                    for cur in [loc_1]:
                        SR = SRs_corr[j][-1][cur][task][:, sel_cell]
                        chosen_SR = SRs_corr[j][-1][cur][task]

                        n_sts = len(SR)
                        state0 = np.zeros((n_sts, 1))
                        state1 = np.zeros((n_sts, 1))
                        state2 = np.zeros((n_sts, 1))
                        state0[0] = 1
                        state1[1] = 1
                        state1[-2] = 4
                        state2[9] = 1

                        if noise:
                            for lengths_in_way in np.arange(len_here[1] - 2):
                                state1[9 + lengths_in_way] = 1

                        state = state0
                        state_new = state1
                        predicted_state1 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        state = state1
                        state_new = state2
                        predicted_state2 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        if cell == 'at_cue':
                            SR_per_task.append(predicted_state1)
                        else:
                            SR_per_task.append(SR)
                    trial_ave_SR = np.mean(SR_per_task, axis=0)
                    SR_list.append(trial_ave_SR)

            prob_agent.append(prob)
            SR_across_trials.append(SR_list)

        past_mp1 = copy.deepcopy(mean_probs_task1)
        past_mc1 = copy.deepcopy(mean_cell_task1)

        # prob_agent_1 = prob_agent/np.max(prob_agent)
        mean_probs_task1.append(np.mean(prob_agent, axis=0))
        mean_prob_task1 = np.mean(prob_agent, axis=0)
        mean_cell_task1.append(np.mean(SR_across_trials, axis=0))

        if split:
            rel_SR_1 = []
            for probs_it in np.arange(len(prob_agent)):
                maps = []
                for iterate in np.arange(10):  # per map
                    maps.append(
                        SR_across_trials[probs_it][iterate] * prob_agent[probs_it][iterate])  # task SRs times task prob
                rel_SR_1.append(maps)

        prob_agent = []
        SR_across_trials = []
        for j in task_2:
            if len_here[0] == 2:
                loc = np.where(np.array(state_corr[j][-1]) == 2)[0]
                loc_1 = loc[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                SR_list = []
                for task in np.arange(10):
                    SR_per_task = []
                    for cur in [loc_1]:
                        SR = SRs_corr[j][-1][cur][task][:, sel_cell]
                        # SR_per_task.append(SR)
                        chosen_SR = SRs_corr[j][-1][cur][task]

                        n_sts = len(SR)
                        state0 = np.zeros((n_sts, 1))
                        state1 = np.zeros((n_sts, 1))
                        state2 = np.zeros((n_sts, 1))
                        state0[0] = 1
                        state1[1] = 1
                        state1[-1] = 4
                        state2[2] = 1

                        if noise:
                            for lengths_in_way in np.arange(len_here[1] - 2):
                                state1[9 + lengths_in_way] = 1

                        state = state0
                        state_new = state1
                        predicted_state1 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        state = state1
                        state_new = state2
                        predicted_state2 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        if cell == 'at_cue':
                            SR_per_task.append(predicted_state1)
                        else:
                            SR_per_task.append(SR)

                    trial_ave_SR = np.mean(SR_per_task, axis=0)
                    SR_list.append(trial_ave_SR)
            else:
                loc = np.where(np.array(state_corr[j][-1]) == sel)[0]
                loc_1 = loc[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                SR_list = []
                for task in np.arange(10):
                    SR_per_task = []
                    for cur in [loc_1]:
                        SR = SRs_corr[j][-1][cur][task][:, sel_cell]
                        chosen_SR = SRs_corr[j][-1][cur][task]

                        n_sts = len(SR)
                        state0 = np.zeros((n_sts, 1))
                        state1 = np.zeros((n_sts, 1))
                        state2 = np.zeros((n_sts, 1))
                        state0[0] = 1
                        state1[1] = 1
                        state1[-1] = 4
                        state2[9] = 1

                        if noise:
                            for lengths_in_way in np.arange(len_here[1] - 2):
                                state1[9 + lengths_in_way] = 1

                        state = state0
                        state_new = state1
                        predicted_state1 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        state = state1
                        state_new = state2
                        predicted_state2 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        if cell == 'at_cue':
                            SR_per_task.append(predicted_state1)
                        else:
                            SR_per_task.append(SR)

                    trial_ave_SR = np.mean(SR_per_task, axis=0)
                    SR_list.append(trial_ave_SR)

            prob_agent.append(prob)
            SR_across_trials.append(SR_list)

        # prob_agent_2 = prob_agent / np.max(prob_agent)
        past_mp2 = copy.deepcopy(mean_probs_task2)
        past_mc2 = copy.deepcopy(mean_cell_task2)

        mean_probs_task2.append(np.mean(prob_agent, axis=0))
        mean_prob_task2 = np.mean(prob_agent, axis=0)
        mean_cell_task2.append(np.mean(SR_across_trials, axis=0))
        if split:
            rel_SR_2 = []
            for probs_it in np.arange(len(prob_agent)):  # for each trial
                maps = []
                for iterate in np.arange(10):  # per map
                    maps.append(
                        SR_across_trials[probs_it][iterate] * prob_agent[probs_it][iterate])  # task SRs times task prob
                rel_SR_2.append(maps)

            task_id_1 = np.argmax(mean_prob_task1)
            task_id_2 = np.argmax(mean_prob_task2)

            split_1 = []
            split_2 = []
            split_1_2 = []
            split_2_2 = []
            for relative in np.arange(len(rel_SR_1)):
                split_1.append(rel_SR_1[relative][task_id_1])  # list of lists
                split_2_2.append(rel_SR_1[relative][task_id_2])
            for relative in np.arange(len(rel_SR_2)):
                split_1_2.append(rel_SR_2[relative][task_id_1])  # list of lists
                split_2.append(rel_SR_2[relative][task_id_2])

            # make 1 df per agent
            col_names = []
            for i in np.arange(len_here[1] + 9):
                name = i
                col_names.append(name)

            task1_df = pd.DataFrame(np.array(split_1), columns=col_names)
            task1_df['alg_type'] = [alg[0]] * len(split_1)
            task1_df['len'] = [len_here[1]] * len(split_1)

            task2_df = pd.DataFrame(np.array(split_2), columns=col_names)
            task2_df['alg_type'] = [alg[0]] * len(split_2)
            task2_df['len'] = [len_here[1]] * len(split_2)

            task3_df = pd.DataFrame(np.array(split_1_2), columns=col_names)
            task3_df['alg_type'] = [alg[0]] * len(split_1_2)
            task3_df['len'] = [len_here[1]] * len(split_1_2)

            task4_df = pd.DataFrame(np.array(split_2_2), columns=col_names)
            task4_df['alg_type'] = [alg[0]] * len(split_2_2)
            task4_df['len'] = len_here[1] * len(split_2_2)

            task1_dfs.append(task1_df)
            task2_dfs.append(task2_df)
            task3_dfs.append(task3_df)
            task4_dfs.append(task4_df)
            keys = filename + str(i)

        if (np.isnan([mean_probs_task2[-1]]).any()) or (np.isnan([mean_probs_task1[-1]]).any()):
            mean_probs_task2 = past_mp2
            mean_cell_task2 = past_mc2
            mean_probs_task1 = past_mp1
            mean_cell_task1 = past_mc1
            print('err')

    len_here[0] = len_here[1]

    return mean_probs_task1, mean_cell_task1, mean_probs_task2, mean_cell_task2, task1_dfs, task2_dfs, task3_dfs, task4_dfs, keys


def find_max_map(sel_cell, state_corr, probs_corr, task):
    prob_agent = []
    for j in task:
        # pick out probabilities on last visit to loc_inf
        loc = np.where(np.array(state_corr[j][-1]) == sel_cell)[0]
        loc_1 = loc[-1]
        prob = np.array(probs_corr[j][-1])[loc_1]
        prob_agent.append(prob)
        # find task_id
    mean_probs = np.mean(prob_agent, axis=0)
    task_id = np.argmax(mean_probs)
    return task_id


def actual_splitter_cells(data, len_here):
    # splitter cells for choice spot
    # pulls out the trial-averaged activity for each agent

    mean_probs_task2 = []
    mean_cell_task2 = []

    mean_probs_task1 = []
    mean_cell_task1 = []

    max_cell = len_here[0] + 7
    sel_cell = 11  # which location are we predicting?

    max1_peragent = []
    min2_peragent = []

    max2_peragent = []
    min1_peragent = []
    for i in np.arange(len(data[28])):
        # select one agent implementation
        num_attempts = data[28][i]
        SRs = np.array(data[29][i])
        probs = np.array(data[30][i])
        states = np.array(data[31][i])

        # select correct trials
        cor_trials = np.where(num_attempts == 1)[0] + 1
        SRs_corr = SRs[cor_trials]
        probs_corr = probs[cor_trials]
        state_corr = states[cor_trials]

        task_1 = []
        task_2 = []

        for i in np.arange(len(cor_trials)):
            if state_corr[i][-1][-1] == 5:
                task_1.append(i)
            elif state_corr[i][-1][-1] == 8:
                task_2.append(i)

        # find max task at location we're trying to predict, ie set of cells we should be looking at
        task1_id = find_max_map(sel_cell, state_corr, probs_corr, task_1)
        task2_id = find_max_map(sel_cell, state_corr, probs_corr, task_2)

        # these are predicted future occupancies of sel_cell given each state
        # Firing of cells in task 1
        SR_max1_pertrial = []  # preffered trial type
        SR_min2_pertrial = []  # non-preffered

        # in task 2
        SR_max2_pertrial = []
        SR_min1_pertrial = []
        # splitter cells are SR_max1_peragent - SR_min1_peragent

        for sel in np.arange(max_cell):  # pick out different locations of agent
            SR_per_task_1 = []  # list of predicted future occupancies across trials
            SR_per_task_2 = []
            for j in task_1:
                task = task1_id
                loc = np.where(np.array(state_corr[j][-1]) == sel)[0]  # last visit of said location
                if len(loc) == 0:
                    SR1 = 0
                    SR2 = 0
                else:
                    loc_1 = loc[-1]
                    prob = np.array(probs_corr[j][-1])[loc_1][task1_id]  # probabilities of the map preferred in task 1
                    prob2 = np.array(probs_corr[j][-1])[loc_1][task2_id]  # probability of map preffered in task 2
                    # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                    cur = loc_1
                    #
                    if sel == 1:
                        SR1 = SRs_corr[j][-1][cur][task][sel_cell, sel] + SRs_corr[j][-1][cur][task][
                            sel_cell, -2] * 4 * prob
                    else:
                        SR1 = SRs_corr[j][-1][cur][task][sel_cell, sel] * prob
                    if sel == 1:
                        SR2 = SRs_corr[j][-1][cur][task][sel_cell, sel] + SRs_corr[j][-1][cur][task][
                            sel_cell, -2] * 4 * prob2
                    else:
                        SR2 = SRs_corr[j][-1][cur][task][sel_cell, sel] * prob2
                SR_per_task_1.append(SR1)  # across trials for current loc
                SR_per_task_2.append(SR2)

            SR_max1_pertrial.append(np.mean(SR_per_task_1))  # across trials, mean for all locations
            SR_min2_pertrial.append(np.mean(SR_per_task_2))

            SR_per_task_1 = []  # list of predicted future occupancies across trials
            SR_per_task_2 = []
            for j in task_2:
                task = task1_id
                loc = np.where(np.array(state_corr[j][-1]) == sel)[0]  # last visit of said location
                if len(loc) == 0:
                    SR1 = 0
                    SR2 = 0
                else:
                    loc_1 = loc[-1]
                    prob = np.array(probs_corr[j][-1])[loc_1][task2_id]  # map preffered in task 2 large
                    prob2 = np.array(probs_corr[j][-1])[loc_1][task1_id]  # map preffered in task 1 small
                    # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                    cur = loc_1
                    if sel == 1:
                        SR1 = SRs_corr[j][-1][cur][task][sel_cell, sel] + SRs_corr[j][-1][cur][task][
                            sel_cell, -1] * 4 * prob
                    else:
                        SR1 = SRs_corr[j][-1][cur][task][sel_cell, sel] * prob
                    if sel == 1:
                        SR2 = SRs_corr[j][-1][cur][task][sel_cell, sel] + SRs_corr[j][-1][cur][task][
                            sel_cell, -2] * 4 * prob2
                    else:
                        SR2 = SRs_corr[j][-1][cur][task][sel_cell, sel] * prob2
                SR_per_task_1.append(SR1)  # across trials for current loc
                SR_per_task_2.append(SR2)

            SR_max2_pertrial.append(np.mean(SR_per_task_1))
            SR_min1_pertrial.append(np.mean(SR_per_task_2))

        # for splitter plots
        max1_peragent.append(SR_max1_pertrial)  # SR for each agent
        min2_peragent.append(SR_min2_pertrial)

        max2_peragent.append(SR_max2_pertrial)  # SR for each agent
        min1_peragent.append(SR_min1_pertrial)

        # for summary plot
    preferred_task = (np.array(max1_peragent) + np.array(max2_peragent)) / 2
    non_preferred_task = (np.array(min1_peragent) + np.array(min2_peragent)) / 2

    return max1_peragent, min2_peragent, max2_peragent, min1_peragent, preferred_task, non_preferred_task


def do_splitting(data, len_here, loc, cell, noise):
    # pulls out the trial-averaged activity for each agent
    mean_probs_task2 = []
    mean_cell_task2 = []

    mean_probs_task1 = []
    mean_cell_task1 = []

    if noise:
        len_here[0] = 2

    if loc == 'post_cue':  # post-cue for splitter cell cells
        if len_here[0] == 2:
            sel = 2
        else:
            sel = 9
    if loc == 'choice':
        sel = 2
    if cell == 'start':
        sel_cell = 0
    if cell == 'post_cue':
        if len_here[0] == 2:
            sel_cell = 2
        else:
            sel_cell = 9
    if cell == 'choice':
        sel_cell = 2
    if cell == 'at_cue':  # for over/underrepresentation of cue
        sel_cell = 1

    for i in np.arange(len(data[28])):
        # select one agent implementation
        num_attempts = data[28][i]
        SRs = np.array(data[29][i])
        probs = np.array(data[30][i])
        states = np.array(data[31][i])

        # select correct trials
        cor_trials = np.where(num_attempts == 1)[0] + 1
        SRs_corr = SRs[cor_trials]
        probs_corr = probs[cor_trials]
        state_corr = states[cor_trials]

        task_1 = []
        task_2 = []

        for i in np.arange(len(cor_trials)):
            if state_corr[i][-1][-1] == 5:
                task_1.append(i)
            elif state_corr[i][-1][-1] == 8:
                task_2.append(i)

        # select state after cue on each correct trial
        prob_agent = []
        SR_across_trials = []
        for j in task_1:
            if len_here[0] == 2:
                loc = np.where(np.array(state_corr[j][-1]) == 2)[0]
                loc_1 = loc[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                SR_list = []
                for task in np.arange(10):
                    SR_per_task = []
                    for cur in [loc_1]:
                        SR = SRs_corr[j][-1][cur][task][:, sel_cell]
                        chosen_SR = SRs_corr[j][-1][cur][task]

                        n_sts = len(SR)
                        state0 = np.zeros((n_sts, 1))
                        state1 = np.zeros((n_sts, 1))
                        state2 = np.zeros((n_sts, 1))
                        state0[0] = 1
                        state1[1] = 1
                        state1[-2] = 4
                        state2[2] = 1

                        if noise:
                            for lengths_in_way in np.arange(len_here[1] - 2):
                                state1[9 + lengths_in_way] = 1

                        # (current_SR @ self.state - self.discount_factor * (current_SR @ state_new)
                        state = state0
                        state_new = state1
                        predicted_state1 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        state = state1
                        state_new = state2
                        predicted_state2 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        if cell == 'at_cue':
                            SR_per_task.append(predicted_state1)
                        else:
                            SR_per_task.append(SR)

                    trial_ave_SR = np.mean(SR_per_task, axis=0)
                    SR_list.append(trial_ave_SR)
            else:
                loc = np.where(np.array(state_corr[j][-1]) == sel)[0]
                loc_1 = loc[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                SR_list = []
                for task in np.arange(10):
                    SR_per_task = []
                    for cur in [loc_1]:
                        SR = SRs_corr[j][-1][cur][task][:, sel_cell]
                        chosen_SR = SRs_corr[j][-1][cur][task]

                        n_sts = len(SR)
                        state0 = np.zeros((n_sts, 1))
                        state1 = np.zeros((n_sts, 1))
                        state2 = np.zeros((n_sts, 1))
                        state0[0] = 1
                        state1[1] = 1
                        state1[-2] = 4
                        state2[9] = 1

                        if noise:
                            for lengths_in_way in np.arange(len_here[1] - 2):
                                state1[9 + lengths_in_way] = 1

                        state = state0
                        state_new = state1
                        predicted_state1 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        state = state1
                        state_new = state2
                        predicted_state2 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        if cell == 'at_cue':
                            SR_per_task.append(predicted_state1)
                        else:
                            SR_per_task.append(SR)
                    trial_ave_SR = np.mean(SR_per_task, axis=0)
                    SR_list.append(trial_ave_SR)

            prob_agent.append(prob)
            SR_across_trials.append(SR_list)

        past_mp1 = copy.deepcopy(mean_probs_task1)
        past_mc1 = copy.deepcopy(mean_cell_task1)

        # prob_agent_1 = prob_agent/np.max(prob_agent)
        mean_probs_task1.append(np.mean(prob_agent, axis=0))
        mean_cell_task1.append(np.mean(SR_across_trials, axis=0))

        prob_agent = []
        SR_across_trials = []
        for j in task_2:
            if len_here[0] == 2:
                loc = np.where(np.array(state_corr[j][-1]) == 2)[0]
                loc_1 = loc[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                SR_list = []
                for task in np.arange(10):
                    SR_per_task = []
                    for cur in [loc_1]:
                        SR = SRs_corr[j][-1][cur][task][:, sel_cell]
                        # SR_per_task.append(SR)
                        chosen_SR = SRs_corr[j][-1][cur][task]

                        n_sts = len(SR)
                        state0 = np.zeros((n_sts, 1))
                        state1 = np.zeros((n_sts, 1))
                        state2 = np.zeros((n_sts, 1))
                        state0[0] = 1
                        state1[1] = 1
                        state1[-1] = 4
                        state2[2] = 1

                        if noise:
                            for lengths_in_way in np.arange(len_here[1] - 2):
                                state1[9 + lengths_in_way] = 1

                        state = state0
                        state_new = state1
                        predicted_state1 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        state = state1
                        state_new = state2
                        predicted_state2 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        if cell == 'at_cue':
                            SR_per_task.append(predicted_state1)
                        else:
                            SR_per_task.append(SR)

                    trial_ave_SR = np.mean(SR_per_task, axis=0)
                    SR_list.append(trial_ave_SR)
            else:
                loc = np.where(np.array(state_corr[j][-1]) == sel)[0]
                loc_1 = loc[-1]
                prob = np.array(probs_corr[j][-1])[loc_1]
                # prob = np.mean(np.array(probs_corr[j][0])[loc], axis=0)
                SR_list = []
                for task in np.arange(10):
                    SR_per_task = []
                    for cur in [loc_1]:
                        SR = SRs_corr[j][-1][cur][task][:, sel_cell]
                        chosen_SR = SRs_corr[j][-1][cur][task]

                        n_sts = len(SR)
                        state0 = np.zeros((n_sts, 1))
                        state1 = np.zeros((n_sts, 1))
                        state2 = np.zeros((n_sts, 1))
                        state0[0] = 1
                        state1[1] = 1
                        state1[-1] = 4
                        state2[9] = 1

                        if noise:
                            for lengths_in_way in np.arange(len_here[1] - 2):
                                state1[9 + lengths_in_way] = 1

                        state = state0
                        state_new = state1
                        predicted_state1 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        state = state1
                        state_new = state2
                        predicted_state2 = (chosen_SR @ state - 0.9 * (chosen_SR @ state_new))[:, 0]

                        if cell == 'at_cue':
                            SR_per_task.append(predicted_state1)
                        else:
                            SR_per_task.append(SR)

                    trial_ave_SR = np.mean(SR_per_task, axis=0)
                    SR_list.append(trial_ave_SR)

            prob_agent.append(prob)
            SR_across_trials.append(SR_list)

        # prob_agent_2 = prob_agent / np.max(prob_agent)
        past_mp2 = copy.deepcopy(mean_probs_task2)
        past_mc2 = copy.deepcopy(mean_cell_task2)

        mean_probs_task2.append(np.mean(prob_agent, axis=0))
        mean_cell_task2.append(np.mean(SR_across_trials, axis=0))

        if (np.isnan([mean_probs_task2[-1]]).any()) or (np.isnan([mean_probs_task1[-1]]).any()):
            mean_probs_task2 = past_mp2
            mean_cell_task2 = past_mc2
            mean_probs_task1 = past_mp1
            mean_cell_task1 = past_mc1
            print('err')

    len_here[0] = len_here[1]

    return mean_probs_task1, mean_cell_task1, mean_probs_task2, mean_cell_task2


def import_dat(data_type, saved, import_folder):
    imp_str = data_type

    noise = False
    dist = False
    if 'noise' in imp_str:
        noise = True
    folder = [import_folder + imp_str + "/"]

    if saved:
        with open(folder[0] + 'saved_import', "rb") as fp:  # Unpickling# print(file)
            data = pickle.load(fp)
        df3 = data[0]
        df_prob1 = data[1]
        df_prob2 = data[2]
        noise = data[3]
        df_prob_split1 = data[4]
        df_prob_split2 = data[5]
        df_split1 = data[6]
        df_split2 = data[7]
        df_split3 = data[8]
        df_split4 = data[9]
        df_control1 = []
        df_control2 = []

    else:
        all_data = []
        key_names = []
        keys_fancy = []
        means = []
        dist_perf1 = []
        dist_perf2 = []

        task1_dfs = []
        task2_dfs = []
        keys_prob = []
        keys_prob2 = []

        split1_dfs = []
        split2_dfs = []
        split3_dfs = []
        split4_dfs = []

        all_steps_1 = []
        all_steps_2 = []

        revs_1_list = []
        revs_2_list = []

        split_1_dfs = []
        split_2_dfs = []
        all_count_1 = []
        all_count_2 = []

        max1_peragent = []
        min2_peragent = []
        max2_peragent = []
        min1_peragent = []
        preferred_task = []
        non_preferred_task = []

        col_names = ['steps_record_500', 'steps_record_400', 'steps_record_300', 'steps_record_200', 'steps_record_100',
                     'trials_on_reversal', 'percent_correct_100', 'percent_correct_200', 'percent_correct_300',
                     'percent_correct_400', 'percent_correct_500', 'n_switch_100', 'n_switch_200', 'n_switch_300',
                     'n_switch_400', 'n_switch_500', "trials_on_reversal_task1", "trials_on_reversal_task2",
                     'percent_correct_100_task1', 'percent_correct_100_task2', 'percent_correct_500_task1',
                     'percent_correct_500_task2', 'inc_switch_100', 'inc_switch_100_task1', 'inc_switch_100_task2',
                     'inc_switch_500', 'inc_switch_500_task1', 'inc_switch_500_task2', 'cue', 'len', 'limit',
                     'alg_type',
                     'lim_type', 'mean_probs_task1', 'mean_cell_task1', 'mean_probs_task2', 'mean_cell_task2',
                     'steps_1', 'steps_2', 'revs_1', 'revs_2', 'SRs_start1', 'SRs_start2', 'mean_probs_task1_rep',
                     'mean_cell_task1_rep', 'mean_probs_task2_rep', 'mean_cell_task2_rep', 'mean_probs_task1_choice',
                     'mean_cell_task1_choice', 'mean_probs_task2_choice', 'mean_cell_task2_choice', 'dist_perf1',
                     'dist_perf2', 'id_here', 'max1', 'min2', 'max2', 'min1', 'pref', 'non-pref']

        for loc in folder:
            file_names = os.listdir(loc)
            for file in file_names:
                if file == '.DS_Store':
                    # do nothing
                    imp = 'no'
                elif file in ['saved_import', 'post_hoc_block', 'stats_block']:
                    # do nothing
                    imp = 'no'
                elif '._' in file:
                    # do nothing
                    imp = 'no'

                else:
                    print(file)
                    with open(loc + file, "rb") as fp:  # Unpickling# print(file)
                        data = pickle.load(fp)
                        means.append(np.mean(data[6]))
                    data = pd.DataFrame(np.asarray(data).T)  # rows are trials
                    num = np.shape(np.asarray(data).T)[1]
                    outcome = False

                    pot_cues = [3, 3.5, 4, 4.5, 5]
                    for qt in pot_cues:
                        if "_" + str(qt) + 'j' in file:
                            cue_here = [qt] * num
                        elif "_" + str(qt) + 's' in file:
                            cue_here = [qt] * num

                    cue_here = [4] * num

                    pot_len = [2, 5, 7, 10, 12, 15, 17, 20, 25]
                    for l in pot_len:
                        if str(l) + "_" in file:
                            len_here = [l] * num

                    if 'reward_feature' in file:
                        alg_type = ['reward_feature'] * num
                        unid = 1000
                    elif 'followed_by_outcome' in file:
                        alg_type = ['joint_inf_priors'] * num
                        lim_type = ['joint_inf_priors'] * num
                        unid = 1000 * 2
                    elif 'outcome' in file:
                        alg_type = ['outcome'] * num
                        lim_type = ['outcome'] * num
                        outcome = True
                        unid = 1000 * 3
                    elif 'joint_inference' in file:
                        alg_type = ['joint_inference'] * num
                        lim_type = ['joint_inference'] * num
                        unid = 1000 * 4
                    elif 'joint_prior' in file:
                        alg_type = ['joint_prior'] * num
                        lim_type = ['joint_prior'] * num
                        unid = 1000 * 5
                    elif 'joint_inf_priors' in file:
                        alg_type = ['joint_inf_priors'] * num
                        lim_type = ['joint_inf_priors'] * num
                        unid = 1000 * 6


                    elif 'switching_SR' or 'just_SR' in file:
                        alg_type = ['SR'] * num
                        lim_type = ['SR'] * num
                        unid = 1000 * 7

                    if 'replay_fake' in file:
                        alg_type = ['replay_fake'] * num
                        lim_type = ['replay'] * num
                    elif 'replay' in file:
                        unid = 1000 * 8
                        alg_type = ['replay'] * num
                        if '500' in file:
                            lim_type = ['replay500'] * num
                        elif '1000' in file:
                            lim_type = ['replay1000'] * num

                    if 'outcome_push' in file:
                        alg_type = ['outcome_push'] * num
                        lim_type = ['outcome_push500'] * num
                        if '500' in file:
                            lim_type = ['outcome_push500'] * num
                        elif '1000' in file:
                            lim_type = ['outcome_push1000'] * num
                    if 'fc' in file:
                        lim_type = ['forced_choice'] * num
                        alg_type = ['forced_choice'] * num
                        unid = 1000 * 9

                    if 'fake_out1' in file:
                        lim_type = ['fake'] * num
                        alg_type = ['fake'] * num
                        unid = 1000 * 10

                    if 'single_SR_unlearning' in file:
                        lim_type = ['single_SR_unlearning'] * num
                        alg_type = ['single_SR_unlearning'] * num
                        unid = 1000 * 11
                    elif 'single_SR' in file:
                        lim_type = ['single_SR'] * num
                        alg_type = ['single_SR'] * num
                        unid = 1000 * 12
                    if 'single_TD' in file:
                        lim_type = ['single_TD'] * num
                        alg_type = ['single_TD'] * num
                        unid = 1000 * 13

                    if 'lim200' in file:
                        limits = [200] * num
                    elif 'lim-1' in file:
                        limits = [-1] * num
                    elif 'lim500' in file:
                        limits = [500] * num
                    elif 'lim50' in file:
                        limits = [50] * num
                    elif 'lim51' in file:
                        limits = [51] * num
                    elif 'lim1000' in file:
                        limits = [1000] * num
                        unid = 1000 * 19
                    elif 'lim100' in file:
                        limits = [100] * num
                    elif 'lim750' in file:
                        limits = [750] * num
                    elif 'lim1500' in file:
                        limits = [1500] * num
                    elif 'lim1250' in file:
                        limits = [1250] * num
                    else:
                        limits = [200] * num

                    if '110' in file:
                        iterator_here = 10
                    elif '115' in file:
                        iterator_here = 15
                    elif '120' in file:
                        iterator_here = 20
                    elif '125' in file:
                        iterator_here = 25
                    elif '130' in file:
                        iterator_here = 30
                    elif '135' in file:
                        iterator_here = 35
                    elif '140' in file:
                        iterator_here = 40
                    elif '145' in file:
                        iterator_here = 45
                    elif '150' in file:
                        iterator_here = 50

                    id_here = np.arange(0, num) + 10 + iterator_here + unid

                    if not outcome:
                        # location = 'choice'
                        # cell = 'post_cue'
                        mean_probs_task1 = []
                        mean_cell_task1 = []
                        mean_probs_task2 = []
                        mean_cell_task2 = [[]]
                        mean_probs_task1_rep = []
                        mean_cell_task1_rep = []
                        mean_probs_task2_rep = []
                        mean_cell_task2_rep = [[]]
                        mean_probs_task1_choice = []
                        mean_cell_task1_choice = []
                        mean_probs_task2_choice = []
                        mean_cell_task2_choice = [[]]
                        all_SRs_1 = []
                        all_SRs_2 = []
                        if imp_str != 'fig_1':
                            task1_dfs, task2_dfs = do_maintenance(data, len_here, alg_type, task1_dfs, task2_dfs, noise,
                                                                  id_here)
                            if 'SRstart' in imp_str:
                                # splitter cell stuff
                                cell = 'post_cue'
                                location = 'post_cue'
                                mean_probs_task1, mean_cell_task1, mean_probs_task2, mean_cell_task2, split1_dfs, split2_dfs, split3_dfs, split4_dfs, keys_fancy = do_splitting_fancy(
                                    data, len_here, location, cell, noise, split1_dfs, split2_dfs, split3_dfs,
                                    split4_dfs, file, keys_fancy, alg_type)
                                # over/underrepresentation of cue
                                cell = 'at_cue'
                                location = 'choice'  # to pick task
                                mean_probs_task1_rep, mean_cell_task1_rep, mean_probs_task2_rep, mean_cell_task2_rep = do_splitting(
                                    data, len_here, location, cell, noise)
                                # figure 1e overall representation of choice maps
                                cell = 'start'
                                location = 'choice'
                                mean_probs_task1_choice, mean_cell_task1_choice, mean_probs_task2_choice, mean_cell_task2_choice = do_splitting(
                                    data, len_here, location, cell, noise)
                                if len_here[0] == 20:
                                    max1_peragent, min2_peragent, max2_peragent, min1_peragent, preferred_task, non_preferred_task = actual_splitter_cells(
                                        data, len_here)

                                # maintenance of splitter cell probs
                                if not noise:
                                    split_1_dfs, split_2_dfs = split_maintain(data, len_here, alg_type, split_1_dfs,
                                                                              split_2_dfs, id_here)
                                    keys_prob2 = keys_prob2 + [loc + file]

                                all_SRs_1, all_SRs_2 = pull_out_learning_SRs(data)
                            all_steps_1, all_steps_2, revs_1_list, revs_2_list = pull_out_learning(data)
                            keys_prob = keys_prob + [loc + file]

                            if 'dist' in imp_str:
                                if 'out' in imp_str:
                                    dist_perf1, dist_perf2, all_count_1, all_count_2 = distance_perf_out(data,
                                                                                                         limits[0],
                                                                                                         all_count_1,
                                                                                                         all_count_2,
                                                                                                         len_here)

                                else:
                                    dist_perf1, dist_perf2, all_count_1, all_count_2 = distance_perf(data, limits[0],
                                                                                                     all_count_1,
                                                                                                     all_count_2,
                                                                                                     len_here)

                    else:
                        mean_probs_task1 = []
                        mean_cell_task1 = []
                        mean_probs_task2 = []
                        mean_cell_task2 = [[]]
                        mean_probs_task1_rep = []
                        mean_cell_task1_rep = []
                        mean_probs_task2_rep = []
                        mean_cell_task2_rep = [[]]
                        mean_probs_task1_choice = []
                        mean_cell_task1_choice = []
                        mean_probs_task2_choice = []
                        mean_cell_task2_choice = [[]]
                        all_SRs_1 = []
                        all_SRs_2 = []
                        if imp_str != 'fig_1':
                            if len_here[0] in [2, 20]:
                                task1_dfs, task2_dfs = do_maintenance(data, len_here, alg_type, task1_dfs, task2_dfs,
                                                                      noise, id_here)
                            all_steps_1, all_steps_2, revs_1_list, revs_2_list = pull_out_learning(data)

                    data = data.iloc[:, 0:28]
                    data = pd.concat([data, pd.DataFrame(
                        [cue_here, len_here, limits, alg_type, lim_type, mean_probs_task1, mean_cell_task1,
                         mean_probs_task2,
                         mean_cell_task2, all_steps_1, all_steps_2, revs_1_list, revs_2_list, all_SRs_1, all_SRs_2,
                         mean_probs_task1_rep, mean_cell_task1_rep, mean_probs_task2_rep, mean_cell_task2_rep,
                         mean_probs_task1_choice, mean_cell_task1_choice, mean_probs_task2_choice,
                         mean_cell_task2_choice, dist_perf1, dist_perf2, id_here, max1_peragent, min2_peragent,
                         max2_peragent, min1_peragent, preferred_task, non_preferred_task]).T], axis=1)
                    data.columns = col_names  # [0:num + 1]
                    all_data.append(data)
                    key_names = key_names + [loc + file]

        data_dict = dict(zip(key_names, all_data))
        data_frame = pd.concat(data_dict, sort=False)
        data_frame.index.names = ['type', 'trial']
        df = data_frame.reset_index(['type', 'trial'])

        df_prob1 = []
        df_prob2 = []
        if imp_str != 'fig_1':
            data_dict_prob = dict(zip(keys_prob, task1_dfs))
            data_prob1 = pd.concat(data_dict_prob, sort=False)
            data_prob1.index.names = ['type', 'trial']
            df_prob1 = data_prob1.reset_index(['type', 'trial'])

            data_dict_prob = dict(zip(keys_prob, task2_dfs))
            data_prob2 = pd.concat(data_dict_prob, sort=False)
            data_prob2.index.names = ['type', 'trial']
            df_prob2 = data_prob2.reset_index(['type', 'trial'])

        df_prob_split1 = []
        df_prob_split2 = []
        df_split1 = []
        df_split2 = []
        df_split3 = []
        df_split4 = []
        df_control1 = []
        df_control2 = []

        if 'SRstart' in imp_str:
            if not noise:
                data_dict_prob = dict(zip(keys_prob2, split_1_dfs))
                data_prob1 = pd.concat(data_dict_prob, sort=False)
                data_prob1.index.names = ['type', 'trial']
                df_prob_split1 = data_prob1.reset_index(['type', 'trial'])

                data_dict_prob = dict(zip(keys_prob2, split_2_dfs))
                data_prob2 = pd.concat(data_dict_prob, sort=False)
                data_prob2.index.names = ['type', 'trial']
                df_prob_split2 = data_prob2.reset_index(['type', 'trial'])

                data_dict_prob = dict(zip(keys_fancy, split1_dfs))
                data_prob1 = pd.concat(data_dict_prob, sort=False)
                data_prob1.index.names = ['type', 'trial']
                df_split1 = data_prob1.reset_index(['type', 'trial'])

                data_dict_prob = dict(zip(keys_fancy, split2_dfs))
                data_prob1 = pd.concat(data_dict_prob, sort=False)
                data_prob1.index.names = ['type', 'trial']
                df_split2 = data_prob1.reset_index(['type', 'trial'])

                data_dict_prob = dict(zip(keys_fancy, split3_dfs))
                data_prob1 = pd.concat(data_dict_prob, sort=False)
                data_prob1.index.names = ['type', 'trial']
                df_split3 = data_prob1.reset_index(['type', 'trial'])

                data_dict_prob = dict(zip(keys_fancy, split4_dfs))
                data_prob1 = pd.concat(data_dict_prob, sort=False)
                data_prob1.index.names = ['type', 'trial']
                df_split4 = data_prob1.reset_index(['type', 'trial'])

        if 'dist' in imp_str:
            key_names = np.arange(len(all_count_1))
            data_dict = dict(zip(key_names, all_count_1))
            data_frame = pd.concat(data_dict, sort=False)
            data_frame.index.names = ['type', 'trial']
            df_control1 = data_frame.reset_index(['type', 'trial'])

            data_dict = dict(zip(key_names, all_count_2))
            data_frame = pd.concat(data_dict, sort=False)
            data_frame.index.names = ['type', 'trial']
            df_control2 = data_frame.reset_index(['type', 'trial'])

        cue_h = 4
        df2 = df[df['cue'].isin([cue_h])]
        df3 = df2[df2['len'] != 25]

        df3.len = df.len.astype(float)
        df3.percent_correct_100 = df3.percent_correct_100.astype(float)
        df3.trials_on_reversal = df3.trials_on_reversal.astype(float)
        df3.n_switch_100 = df3.n_switch_100.astype(float)
        df3.steps_record_100 = df3.steps_record_100.astype(float)
        df3.percent_correct_500 = df3.percent_correct_500.astype(float)
        df3.trials_on_reversal_task1 = df3.trials_on_reversal_task1.astype(float)
        df3.trials_on_reversal_task2 = df3.trials_on_reversal_task2.astype(float)
        df3.percent_correct_100_task1 = df3.percent_correct_100_task1.astype(float)
        df3.percent_correct_100_task2 = df3.percent_correct_100_task2.astype(float)
        df3.inc_switch_100 = df3.inc_switch_100.astype(float)
        df3.inc_switch_100_task1 = df3.inc_switch_100_task1.astype(float)
        # df3.inc_switch_100_task2 = df3.inc_switch_100_task2.astype(float)  # switch from task 1 to task 2
        df3.limit = df.limit.astype(float)

    return df3, df_prob1, df_prob2, noise, df_prob_split1, df_prob_split2, df_split1, df_split2, df_split3, df_split4, df_control1, df_control2


def import_dat_simple(data_type, saved, import_folder):
    imp_str = data_type

    noise = False
    dist = False
    if 'noise' in imp_str:
        noise = True
    folder = [import_folder + imp_str + "/"]

    if saved:
        with open(folder[0] + 'saved_import', "rb") as fp:  # Unpickling# print(file)
            df3 = pickle.load(fp)

    else:
        all_data = []
        key_names = []
        keys_fancy = []
        means = []
        dist_perf1 = []
        dist_perf2 = []

        task1_dfs = []
        task2_dfs = []
        keys_prob = []
        keys_prob2 = []

        split1_dfs = []
        split2_dfs = []
        split3_dfs = []
        split4_dfs = []

        all_steps_1 = []
        all_steps_2 = []

        revs_1_list = []
        revs_2_list = []

        split_1_dfs = []
        split_2_dfs = []

        col_names = ['steps_record_500', 'steps_record_400', 'steps_record_300', 'steps_record_200', 'steps_record_100',
                     'trials_on_reversal', 'percent_correct_100', 'percent_correct_200', 'percent_correct_300',
                     'percent_correct_400', 'percent_correct_500', 'n_switch_100', 'n_switch_200', 'n_switch_300',
                     'n_switch_400', 'n_switch_500', "trials_on_reversal_task1", "trials_on_reversal_task2",
                     'percent_correct_100_task1', 'percent_correct_100_task2', 'percent_correct_500_task1',
                     'percent_correct_500_task2', 'inc_switch_100', 'inc_switch_100_task1', 'inc_switch_100_task2',
                     'inc_switch_500', 'inc_switch_500_task1', 'inc_switch_500_task2', 'cue', 'len', 'limit',
                     'alg_type',
                     'lim_type', 'id_here']

        for loc in folder:
            file_names = os.listdir(loc)
            ity = 0
            for file in file_names:
                # print(file)
                if file == '.DS_Store':
                    # do nothing
                    imp = 'no'
                elif file == 'saved_import':
                    # do nothing
                    imp = 'no'
                elif '._' in file:
                    # do nothing
                    imp = 'no'
                else:
                    ity = ity + 1
                    with open(loc + file, "rb") as fp:  # Unpickling# print(file)
                        data = pickle.load(fp)
                        means.append(np.mean(data[6]))
                    data = pd.DataFrame(np.asarray(data).T)  # rows are trials
                    num = np.shape(np.asarray(data).T)[1]
                    outcome = False

                    pot_cues = [3, 3.5, 4, 4.5, 5]
                    for qt in pot_cues:
                        if "_" + str(qt) + 'j' in file:
                            cue_here = [qt] * num
                        elif "_" + str(qt) + 's' in file:
                            cue_here = [qt] * num

                    cue_here = [4] * num

                    pot_len = [2, 3, 4, 5, 7, 10, 12, 15, 17, 20, 25, 100]
                    for l in pot_len:
                        if str(l) + "_" in file:
                            len_here = [l] * num

                    if 'default' in file:
                        alg_type = ['explore'] * num
                        lim_type = ['explore'] * num
                        unid = 0
                    elif 'explore' in file:
                        alg_type = ['explore'] * num
                        lim_type = ['explore'] * num
                        unid = 0
                    else:
                        if 'reward_feature' in file:
                            alg_type = ['reward_feature'] * num
                            lim_type = ['reward_feature'] * num
                            unid = 1000
                        elif 'rew_feat1' in file:
                            alg_type = ['reward_feature'] * num
                            lim_type = ['reward_feature'] * num
                            unid = 2000
                        elif 'outcome' in file:
                            alg_type = ['outcome'] * num
                            lim_type = ['outcome'] * num
                            unid = 3000
                            outcome = True
                        elif 'joint_inf_cov' in file:
                            alg_type = ['joint_inference'] * num
                            lim_type = ['joint_inference'] * num
                            unid = 4000
                        elif 'joint_prior' in file:
                            alg_type = ['joint_prior'] * num
                            lim_type = ['joint_prior'] * num
                            unid = 5000
                        elif 'joint_inf_priors' in file:
                            unid = 6000
                            alg_type = ['joint_inf_priors'] * num
                            lim_type = ['joint_inf_priors'] * num
                        elif 'switching_SR' or 'just_SR' in file:
                            alg_type = ['SR'] * num
                            lim_type = ['SR'] * num
                            unid = 7000

                        if 'replay_fake' in file:
                            alg_type = ['replay_fake'] * num
                            lim_type = ['replay'] * num
                        elif 'replay' in file:
                            alg_type = ['replay'] * num
                            unid = 8000
                            if '500' in file:
                                lim_type = ['replay500'] * num
                            elif '1000' in file:
                                lim_type = ['replay1000'] * num

                        if 'outcome_push' in file:
                            alg_type = ['outcome_push'] * num
                            lim_type = ['outcome_push500'] * num
                            if '500' in file:
                                lim_type = ['outcome_push500'] * num
                            elif '1000' in file:
                                lim_type = ['outcome_push1000'] * num
                        if 'fc' in file:
                            lim_type = ['forced_choice'] * num
                            alg_type = ['forced_choice'] * num
                            unid = 9000

                        if 'fake_out1' in file:
                            lim_type = ['ideal_observer'] * num
                            alg_type = ['ideal_observer'] * num
                            unid = 10000

                        if 'single_SR_unlearning' in file:
                            lim_type = ['single_SR_unlearning'] * num
                            alg_type = ['single_SR_unlearning'] * num
                            unid = 1100
                        elif 'single_SR' in file:
                            lim_type = ['single_SR'] * num
                            alg_type = ['single_SR'] * num
                            unid = 1200
                        if 'single_TD' in file:
                            lim_type = ['single_TD'] * num
                            alg_type = ['single_TD'] * num
                            unid = 1300

                    if 'lim200' in file:
                        limits = [200] * num
                    elif 'lim-1' in file:
                        limits = [-1] * num
                    elif 'lim500' in file:
                        limits = [500] * num
                    elif 'lim50' in file:
                        limits = [50] * num
                    elif 'lim51' in file:
                        limits = [51] * num
                    elif 'lim1000' in file:
                        limits = [1000] * num
                    elif 'lim100' in file:
                        limits = [100] * num
                    elif 'lim750' in file:
                        limits = [750] * num
                    elif 'lim1500' in file:
                        limits = [1500] * num
                    elif 'lim1250' in file:
                        limits = [1250] * num
                    else:
                        limits = [200] * num

                    if '110' in file:
                        iterator_here = 10
                    elif '115' in file:
                        iterator_here = 15
                    elif '120' in file:
                        iterator_here = 20
                    elif '125' in file:
                        iterator_here = 25
                    elif '130' in file:
                        iterator_here = 30
                    elif '135' in file:
                        iterator_here = 35
                    elif '140' in file:
                        iterator_here = 40
                    elif '145' in file:
                        iterator_here = 45
                    elif '150' in file:
                        iterator_here = 50

                    if data_type in ['struct_disc', 'DNMS']:
                        iterator_here = 100
                        id_here = np.arange(0, num) + 10 + iterator_here + unid + ity * num
                    else:
                        id_here = np.arange(0, num) + 10 + iterator_here + unid

                    data = data.iloc[:, 0:28]
                    data = pd.concat([data, pd.DataFrame(
                        [cue_here, len_here, limits, alg_type, lim_type, id_here]).T], axis=1)
                    data.columns = col_names  # [0:num + 1]
                    all_data.append(data)
                    key_names = key_names + [loc + file]

        data_dict = dict(zip(key_names, all_data))
        data_frame = pd.concat(data_dict, sort=False)
        data_frame.index.names = ['type', 'trial']
        df = data_frame.reset_index(['type', 'trial'])

        cue_h = 4
        df2 = df[df['cue'].isin([cue_h])]
        df3 = df2[df2['len'] != 25]

        df3.len = df.len.astype(float)
        df3.percent_correct_100 = df3.percent_correct_100.astype(float)
        df3.trials_on_reversal = df3.trials_on_reversal.astype(float)
        df3.n_switch_100 = df3.n_switch_100.astype(float)
        df3.steps_record_100 = df3.steps_record_100.astype(float)
        df3.percent_correct_500 = df3.percent_correct_500.astype(float)
        df3.trials_on_reversal_task1 = df3.trials_on_reversal_task1.astype(float)
        df3.trials_on_reversal_task2 = df3.trials_on_reversal_task2.astype(float)
        df3.percent_correct_100_task1 = df3.percent_correct_100_task1.astype(float)
        df3.percent_correct_100_task2 = df3.percent_correct_100_task2.astype(float)
        df3.inc_switch_100 = df3.inc_switch_100.astype(float)
        df3.inc_switch_100_task1 = df3.inc_switch_100_task1.astype(float)
        # df3.inc_switch_100_task2 = df3.inc_switch_100_task2.astype(float)  # switch from task 1 to task 2
        df3.limit = df.limit.astype(float)

    return df3


def save_import(df3, df_prob1, df_prob2, noise, df_prob_split1, df_prob_split2, df_split1, df_split2, df_split3,
                df_split4, df_control1, df_control2, data_type, import_folder):
    imp_str = data_type

    noise = False
    dist = False
    if 'noise' in imp_str:
        noise = True
    folder = import_folder + imp_str + "/"

    with open(folder + 'saved_import', "wb") as fp:  # Pickling
        pickle.dump(
            [df3, df_prob1, df_prob2, noise, df_prob_split1, df_prob_split2, df_split1, df_split2, df_split3,
             df_split4], fp)


def save_import_simple(df3, import_folder):
    imp_str = data_type

    noise = False
    dist = False
    folder = import_folder + imp_str + "/"

    with open(folder + 'saved_import', "wb") as fp:  # Pickling
        pickle.dump(df3, fp)


def format_plot():
    sns.set(rc={'figure.figsize': (4.5, 5)})
    sns.set(font_scale=1.8)
    sns.set_style(style='white')
    palette_plots = {'joint_inference': '#35C4B9', 'joint_prior': '#35C4B9', 'outcome': 'blue', 'SR': '#C5499B',
                     'joint_inf_priors': 'purple', 'fake': 'black', 2: '#F7C1AD', 20: '#35C4B9', 'single_SR': '#F7C1AD',
                     'single_TD': '#666666', 'single_SR_unlearning': '#35C4B9', 'new': '#35C4B9',
                     'reward_feature': '#35C4B9', 'forced_choice': '#35C4B9', 'explore': '#35C4B9', 'replay': '#35C4B9',
                     'ideal_observer': '#35C4B9'}
    return palette_plots


def plot_perf_metrics(df, noise, type, sub):
    if type == 'blocks':
        metric = 'trials_on_reversal'
    elif type == 'random':
        metric = 'percent_correct_100'
    elif type == 'random_rev':
        metric = 'inc_switch_100'

    palette_plots = format_plot()

    if sub in ['b', 'c']:
        df = df[df['lim_type'].isin(['SR', 'outcome'])]
    if sub in ['e', 'f']:
        palette_plots = {'outcome': '#9E9EEF', 'SR': '#F2C9E8', 'joint_inf_priors': 'purple'}

    sns.lineplot(x='len', y=metric, hue='lim_type', data=df, marker='o', legend=False,
                 palette=palette_plots)

    if type == 'blocks':
        plt.ylabel('Attempts on block reversals')
        if noise:
            plt.ylim(0, 10)
        else:
            plt.ylim(0, 17)
            ymarks = range(0, 20, 5)
            plt.yticks(ymarks)
    elif type == 'random':
        plt.ylim(40, 100)
        plt.ylabel('Performance on random trials')
    elif type == 'random_rev':
        plt.ylabel('Steps on first block')
        if noise:
            plt.ylim(0, 10)
        else:
            plt.ylim(0, 17)

    if noise:
        plt.xlabel('Distractors around cue')
    else:
        plt.xlabel('Distance from cue to choice')


def fig_3(df, noise):
    # try rolling average: urgh its going to be picking out columns and then doing things with it annoying
    # df['moving'] = df.groupby(['type', 'trial',]).rolling(2)['value'].mean().reset_index(drop=True)

    # random performance only
    add = ''
    if noise:
        add = 'noise'

    # fig3b/e
    sub = 'b'
    type = 'blocks'
    plot_perf_metrics(df, noise, type, sub)
    plt.savefig('figures/fig_3b' + add + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    # fig3c/f
    sub = 'c'
    type = 'random'
    plot_perf_metrics(df, noise, type, sub)
    plt.savefig('figures/fig_3c' + add + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    # fig 4c/d
    sub = 'e'
    type = 'blocks'
    plot_perf_metrics(df, noise, type, sub)
    plt.savefig('figures/fig_4c' + add + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    # fig4e/f
    sub = 'f'
    type = 'random'
    plot_perf_metrics(df, noise, type, sub)
    plt.savefig('figures/fig_4e' + add + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()
    stats_me = df[['id_here', 'len', 'alg_type', 'trials_on_reversal']]
    stats_out, post_hoc_out = anova_mixed_effects(stats_me, 'trials_on_reversal')

    return stats_out, post_hoc_out


def plot_maintenance(df_prob1, df_prob2, fig):
    # plotting confidence drops
    # all_maint = pd.concat([df_prob1, df_prob2])
    all_maint = df_prob1
    add_on = ''
    if noise:
        add_on = 'noise'

    if fig == '5d':
        df5 = all_maint[all_maint['alg_type'].isin(['SR', 'joint_inf_priors'])]
        for j in [20]:
            df6 = df5[df5['len'].isin([j])]
            plt_me = pd.melt(df6, id_vars=['type', 'trial', 'alg_type', 'len', 'id_here'])
            stats_me = plt_me[['id_here', 'alg_type', 'variable', 'value']]
            stats_me.columns = ['id_here', 'alg_type', 'len', 'value']
            statistic = 'value'
            palette_plots = format_plot()
            plt_me['variable'] = plt_me['variable'] + 1
            if noise:
                ax = sns.barplot(x='variable', y='value', hue='alg_type', data=plt_me, palette=palette_plots)
                ax.get_legend().remove()
            else:
                ax = sns.lineplot(x='variable', y='value', hue='alg_type', data=plt_me, marker='o', legend=False,
                                  palette=palette_plots)
            plt.xlabel('len')
            plt.ylim(0, 1)
            plt.ylabel('Probability of inferred task after cue')
            if noise:
                plt.xlabel('Noise ' + str(j))
                stats_out = []
                post_hoc_out = t_test(stats_me, statistic)
            else:
                plt.xlabel('Distance after cue')
                stats_out, post_hoc_out = anova_mixed_effects(stats_me, statistic)
            plt.savefig('figures/fig_' + fig + str(j) + add_on + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
            plt.show()
    if fig == '5b':
        df5 = all_maint[all_maint['alg_type'].isin(['outcome', 'joint_inf_priors'])]
        for j in [2, 20]:
            df6 = df5[df5['len'].isin([j])]
            plt_me = pd.melt(df6, id_vars=['type', 'trial', 'alg_type', 'len'])
            # picking out at choice data in cue dist:
            if not noise:
                if j == 2:
                    plt_me = plt_me[plt_me['variable'] == 0]
                if j == 20:
                    plt_me = plt_me[plt_me['variable'] == 18]
            palette_plots = format_plot()
            ax = sns.barplot(x='variable', y='value', hue='alg_type', data=plt_me, palette=palette_plots)
            ax.get_legend().remove()
            plt.ylim(0, 1)
            plt.ylabel('Probability of inferred task after cue')
            plt.xlabel('Noise ' + str(j))
            plt.savefig('figures/fig_' + fig + str(j) + add_on + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
            plt.show()

    return stats_out, post_hoc_out


def plot_steps(df, noise):
    # steps
    sns.set(font_scale=1.5)
    sns.set_style(style='white')
    sns.lineplot(x='len', y='steps_2', hue='lim_type', data=df, marker='o', legend=False,
                 palette={'joint_inference': 'purple', 'outcome': 'blue', 'forced_choice': 'grey', 'SR': '#C5499B',
                          'replay1000': 'green', 'replay500': 'grey', 'joint_inf_priors': 'purple',
                          'joint_inference': 'purple', 'fake': 'black'})
    plt.xlabel('len')
    # plt.ylim(0,1000)
    # plt.title('Task 1')
    plt.ylabel('Performance on random trials')
    plt.xlabel('Distance from cue to choice')
    plt.ylabel('Performance on random trials task2')
    plt.ylabel('Attempts on reversals')
    plt.ylabel('Steps on first block')
    plt.show()
    # plt.legend(loc='lower left', labels=['Joint', 'HPC', 'PFC'])
    # plt.show()
    # plt.savefig('rev_TD_2_t2.pdf', dpi=500, bbox_inches='tight')

    # palette={'replay' : 'blue','replay_fake': 'orange', 'SR': '#C5499B', '500': '#98D7DB', 'joint_inf_priors': 'purple',  '200': 'orange'}

    add = ''
    if noise:
        add = 'noise'
        estimate = np.mean
    else:
        estimate = np.median

    palette_plots = format_plot()
    sns.lineplot(x='len', y='steps_1', hue='lim_type', data=df, marker='o', legend=False, palette=palette_plots,
                 estimator=estimate)
    plt.ylabel('Updates to incorrect map')
    if noise:
        plt.xlabel('Distractors around cue')
    else:
        plt.xlabel('Distance from cue to choice')
    plt.savefig('figures/fig_5aupdates' + add + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    stats_me = df[['id_here', 'len', 'alg_type', 'steps_1']]
    stats_me.steps_1 = stats_me.steps_1.astype(float)
    statistic = 'steps_1'
    stats_out, post_hoc_out = anova_mixed_effects(stats_me, statistic)

    return stats_out, post_hoc_out


def plot_fig2(df):
    df = df[df['alg_type'].isin(['single_SR', 'single_TD', 'single_SR_unlearning', 'SR', 'outcome'])]
    plt_me = df[['trials_on_reversal', 'alg_type']]
    palette_plots = format_plot()

    f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2,
                                 sharex=True)
    ax1 = sns.barplot(x='alg_type', y='trials_on_reversal', data=plt_me, palette=palette_plots,
                      order=['single_SR', 'single_SR_unlearning', 'single_TD', 'outcome', 'SR'], ax=ax1)
    ax2 = sns.barplot(x='alg_type', y='trials_on_reversal', data=plt_me, palette=palette_plots,
                      order=['single_SR', 'single_SR_unlearning', 'single_TD', 'outcome', 'SR'], ax=ax2)
    ax1.set_ylim(250, 350)
    ax2.set_ylim(0, 20)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    f.text(0, 0.5, 'Attempts on block reversals', va="center", rotation="vertical")
    ax1.xaxis.tick_top()
    ax2.xaxis.tick_bottom()
    f.subplots_adjust(left=0.18)
    d = .01
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    ax2.set_xlabel("")
    ax2.set_xticklabels(['SR', 'SR1', 'TD', 'OI', 'FI'])
    # plt.xticks(rotation=45, ha='right')
    plt.savefig('figures/fig_2c' + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    stats_me = df[['id_here', 'len', 'alg_type', 'trials_on_reversal']]
    stats_out, post_hoc_out = non_param_anova(stats_me, statistic='trials_on_reversal')

    plt_me = df[['percent_correct_100', 'alg_type']]
    palette_plots = format_plot()
    ax = sns.barplot(x='alg_type', y='percent_correct_100', data=plt_me, palette=palette_plots,
                     order=['single_SR', 'single_SR_unlearning', 'single_TD', 'outcome', 'SR'])
    ax.set_xticklabels(['SR', 'SR1', 'TD', 'OI', 'FI'])
    ax.set_xlabel('')
    plt.ylim(40, 100)
    # plt.xticks(rotation=45, ha='right')
    plt.ylabel('Performance on random trials')
    plt.savefig('figures/fig_2d' + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    stats_me = df[['id_here', 'len', 'alg_type', 'percent_correct_100']]
    stats_out_2, post_hoc_out_2 = anova(stats_me, statistic='percent_correct_100')
    return stats_out, post_hoc_out, stats_out_2, post_hoc_out_2


def plot_mean_SR(df, noise):
    add = ''
    for id in ['SRs_start1', 'SRs_start2']:
        if id == 'SRs_start1':
            algs = ['SR']
        else:
            algs = ['SR', 'joint_inf_priors']
        for alg in algs:
            df_int = df[df['lim_type'] == alg]
            for pot_len in [20]:
                df_int2 = df_int[df_int['len'] == pot_len]
                df_int3 = df_int2[id]
                max_cell_list = np.mean(df_int3.values, axis=0)

                if noise:
                    add = 'noise'
                    top_add = round(pot_len / 2) - 3
                    if top_add < 0:
                        top_add = 0
                        grid = np.zeros((7, 3))
                    else:
                        grid = np.zeros((pot_len + 1, 3))
                    # arms
                    grid[0 + top_add, 2] = max_cell_list[5]
                    grid[1 + top_add, 2] = max_cell_list[4]
                    grid[2 + top_add, 2] = max_cell_list[3]
                    grid[3 + top_add, 2] = max_cell_list[2]
                    grid[4 + top_add, 2] = max_cell_list[6]
                    grid[5 + top_add, 2] = max_cell_list[7]
                    grid[6 + top_add, 2] = max_cell_list[8]
                    for gr in np.arange(2):  # 0,1
                        grid[3 + top_add, gr] = max_cell_list[gr]

                    # add things gradually around cues stacking on one top, one on bottom

                    top_cue_loc = 3 + top_add - 1
                    bottom_cue_loc = 3 + top_add + 1

                    # add on top of cue
                    val = 0
                    for gr in np.arange(1, round((pot_len - 2) / 2) + 1):
                        fill_me = top_cue_loc - gr
                        grid[fill_me, 1] = max_cell_list[9 + val]
                        val = val + 2
                    val = 1
                    for gr in np.arange(1, int((pot_len - 2) / 2) + 1):
                        fill_me = bottom_cue_loc + gr
                        grid[fill_me, 1] = max_cell_list[9 + val]
                        val = val + 2
                    grid[top_cue_loc, 1] = max_cell_list[-2]  # cues
                    grid[bottom_cue_loc, 1] = max_cell_list[-1]  # cues


                else:
                    grid = np.zeros((7, pot_len + 1))
                    grid[0, pot_len] = max_cell_list[5]
                    grid[1, pot_len] = max_cell_list[4]
                    grid[2, pot_len] = max_cell_list[3]
                    grid[4, pot_len] = max_cell_list[6]
                    grid[5, pot_len] = max_cell_list[7]
                    grid[6, pot_len] = max_cell_list[8]

                    for gr in np.arange(2):
                        grid[3, gr] = max_cell_list[gr]
                    if pot_len > 2:
                        for gr in np.arange(pot_len - 1):
                            grid[3, 2 + gr] = max_cell_list[9 + gr]

                    grid[3, pot_len] = max_cell_list[2]
                    grid[2, 1] = max_cell_list[-2]
                    grid[4, 1] = max_cell_list[-1]

                plt.imshow(np.log10(grid), cmap='RdPu', vmin=-4, vmax=0.5)  # vmin=0, vmax=1
                plt.margins(x=0)
                plt.axis('off')
                plt.savefig('figures/' + id + alg + str(pot_len) + add + '.pdf', dpi=500, bbox_inches='tight',
                            format='pdf')
                plt.show()


def plot_cell(df3, type, task, pot_len):
    nom = 'mean_probs_task' + str(task) + '_' + type
    list_of_lists = list(df3[nom].values)
    indices = np.argmax(list_of_lists, axis=1)
    nom = 'mean_cell_task' + str(task) + '_' + type
    mean_cell_task1 = list(df3[nom].values)
    max_cell_list = []

    for lit in np.arange(len(mean_cell_task1)):
        # for every agent lit is agent num
        ind = indices[lit]
        max_cell = mean_cell_task1[lit][
            ind]
        max_cell_list.append(max_cell)
    max_cell_list = np.mean(max_cell_list,
                            axis=0)

    grid = np.zeros((7, pot_len + 1))
    grid[0, pot_len] = max_cell_list[5]
    grid[1, pot_len] = max_cell_list[4]
    grid[2, pot_len] = max_cell_list[3]
    grid[4, pot_len] = max_cell_list[6]
    grid[5, pot_len] = max_cell_list[7]
    grid[6, pot_len] = max_cell_list[8]

    for gr in np.arange(2):
        grid[3, gr] = max_cell_list[gr]
    if pot_len > 2:
        for gr in np.arange(pot_len - 1):
            grid[3, 2 + gr] = max_cell_list[9 + gr]
    grid[3, pot_len] = max_cell_list[2]
    grid[2, 1] = max_cell_list[-2]
    grid[4, 1] = max_cell_list[-1]

    # Plot task 1 max cells
    if task == 1:
        colors = 'RdPu'
    if task == 2:
        colors = 'Oranges'
    plt.imshow(np.log10(grid), cmap=colors, vmin=-3, vmax=0.5)
    plt.margins(x=0)
    plt.axis('off')
    # plt.colorbar()
    plt.savefig('figures/fig_2e' + str(task) + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    # plt.savefig('figures/fig_1e'+str(task)+'colorbar-3jet', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()


def plot_fig2_SRs(df_split):
    # starter cell plots
    for alg in ['SR']:
        int = df_split[df_split['lim_type'] == alg]
        for pot_len in [2]:
            df3 = int[int['len'].isin([pot_len])]
            plot_cell(df3=df3, type='choice', task=1, pot_len=pot_len)
            plot_cell(df3=df3, type='choice', task=2, pot_len=pot_len)


def per_task(df3):
    over_rep = []
    under_rep = []

    for task in [1, 2]:
        nom = 'mean_probs_task' + str(task) + '_rep'
        list_of_lists = list(df3[nom].values)
        indices = np.argmax(list_of_lists, axis=1)

        nom = 'mean_cell_task' + str(task) + '_rep'
        mean_cell_task = list(df3[nom].values)
        diff_1 = []
        diff_2 = []

        for lit in np.arange(len(mean_cell_task)):
            ind = indices[lit]
            max_cell = mean_cell_task[lit][ind]
            diff_1.append(max_cell[-2])
            diff_2.append(max_cell[-1])

        if task == 1:
            under_rep.append(np.array(diff_1))  # underrepresentation of correct cue
            over_rep.append(np.array(diff_2))
        elif task == 2:
            under_rep.append(np.array(diff_2))
            over_rep.append((diff_1))

    split_1 = sum(over_rep)
    split_2 = sum(under_rep)

    return split_1, split_2


def make_dataframes(split, data_all, count, alg, pot_len, typ, key_names, id_here):
    d1 = pd.DataFrame(split)

    num = np.shape(np.asarray(d1).T)[1]
    algie = [alg] * num
    lenie = [pot_len] * num
    typie = [typ] * num

    d1_i = pd.concat([d1, pd.DataFrame([algie, lenie, typie, id_here]).T], axis=1)
    key_names = key_names + [str(count) + 'hi']
    count = count + 1

    d1_i.columns = ['data', 'alg', 'len', 'typ', 'id_here']
    data_all.append(d1_i)

    return data_all, count, key_names


def cue_plots(typ_here, df):
    d_1 = df[df['typ'] == typ_here]
    add = ''
    if noise:
        add = 'noise'

    palette_here = format_plot()
    sns.lineplot(x='len', y='data', hue='alg', data=d_1, marker='o', legend=False, palette=palette_here)

    stats_me = d_1[['id_here', 'len', 'alg', 'data']]
    stats_me.columns = ['id_here', 'len', 'alg_type', 'data']
    stats_out, post_hoc_out = anova_mixed_effects(stats_me, 'data')

    if typ_here == 'over':
        if noise:
            plt.ylim(-0.25, 3)
        else:
            plt.ylim(-0.25, 2.5)
            ymarks = range(0, 3, 1)
            plt.yticks(ymarks)
        plt.ylabel('Representation of incorrect cue')
    elif typ_here == 'under':
        if noise:
            plt.ylim(-5.5, 0.25)
        else:
            plt.ylim(-6, 0.25)

        plt.ylabel('Representation of correct cue')
    if noise:
        plt.xlabel('Distractors around cue')
    else:

        plt.xlabel('Distance from cue to choice')
    plt.savefig('figures/5c' + typ_here + add + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()
    return stats_out, post_hoc_out


def cue_preds(df_split, noise):
    # plottings average splitter probabilities and over/underprediction of cue
    data_all = []
    count = 0
    key_names = []

    for alg in ['SR', 'joint_inf_priors']:
        df2 = df_split[df_split['lim_type'] == alg]
        for pot_len in [2, 5, 7, 10, 12, 15, 17, 20]:
            df3 = df2[df2['len'].isin([pot_len])]
            id_here = df3['id_here']

            split_1, split_2 = per_task(df3=df3)

            typ = 'over'
            data_all, count, key_names = make_dataframes(split_1, data_all, count, alg, pot_len, typ, key_names,
                                                         id_here)
            typ = 'under'
            data_all, count, key_names = make_dataframes(split_2, data_all, count, alg, pot_len, typ, key_names,
                                                         id_here)

    data_dict = dict(zip(key_names, data_all))
    data_frame_here = pd.concat(data_dict, sort=False)
    data_frame_here.index.names = ['type', 'trial']
    df_new = data_frame_here.reset_index(['type', 'trial'])

    stats = []
    post_hoc = []

    for typ_here in ['over', 'under']:
        stats_out, post_hoc_out = cue_plots(typ_here, df_new)
        stats.append(stats_out)
        post_hoc.append(post_hoc_out)

    return stats, post_hoc


def plot_splitter(pot_len, max_cell_list, task, alg, typ):
    grid = np.zeros((7, pot_len + 1))
    grid[0, pot_len] = max_cell_list[5]
    grid[1, pot_len] = max_cell_list[4]
    grid[2, pot_len] = max_cell_list[3]
    grid[4, pot_len] = max_cell_list[6]
    grid[5, pot_len] = max_cell_list[7]
    grid[6, pot_len] = max_cell_list[8]

    for gr in np.arange(2):
        grid[3, gr] = max_cell_list[gr]
    if pot_len > 2:
        for gr in np.arange(pot_len - 1):
            grid[3, 2 + gr] = max_cell_list[9 + gr]
    grid[3, pot_len] = max_cell_list[2]
    grid[2, 1] = max_cell_list[-2]
    grid[4, 1] = max_cell_list[-1]

    plt.imshow(grid, cmap='jet', vmin=0, vmax=1.2)
    plt.margins(x=0)
    plt.axis('off')
    # plt.savefig('figures/fig_5b' + str(task) + alg + typ, dpi=500, bbox_inches='tight') #reactivate
    plt.show()


def plot_splitter_rl(df):
    stats_out = []
    post_hoc_out = []
    aucs = []
    int1 = df[df['len'] == 20]
    pot_len = 20
    for alg in ['joint_inf_priors', 'SR']:
        auc_list = []
        int2 = int1[int1['alg_type'] == alg]
        int2 = int2.reset_index()
        for task in [1, 2]:
            for typ in ['min', 'max']:
                max_cell_list = np.mean(np.array(int2[typ + str(task)].values.tolist()), axis=0)

                grid = np.zeros((7, pot_len + 1))
                grid[0, pot_len] = max_cell_list[5]
                grid[1, pot_len] = max_cell_list[4]
                grid[2, pot_len] = max_cell_list[3]
                grid[4, pot_len] = max_cell_list[6]
                grid[5, pot_len] = max_cell_list[7]
                grid[6, pot_len] = max_cell_list[8]

                for gr in np.arange(2):
                    grid[3, gr] = max_cell_list[gr]
                if pot_len > 2:
                    for gr in np.arange(pot_len - 2):
                        grid[3, 2 + gr] = max_cell_list[9 + gr]
                grid[3, pot_len] = max_cell_list[2]

                plt.imshow(grid, cmap='jet', vmin=0, vmax=1.2)
                plt.margins(x=0)
                plt.axis('off')
                plt.savefig('figures/fig_6a' + str(task) + alg + typ + '.pdf', dpi=500, bbox_inches='tight',
                            format='pdf')
                plt.show()

        # figure 6
        dfs = []
        for preference in ['pref', 'non-pref']:
            maze_index = [0, 1] + list(np.arange(20, 27)) + list(np.arange(2, 20))
            pref_df = pd.DataFrame(int2[preference].to_list(), columns=maze_index)
            pref_df['alg_type'] = int2['alg_type']
            if 'pref':
                pref_df['id_here'] = int2['id_here'] + 10000
            else:
                pref_df['id_here'] = int2['id_here']
            pref_df['typ'] = [preference] * len(pref_df)
            pref_df = pref_df.drop(columns=np.arange(21, 27).tolist())

            analysis_auc = pref_df
            auc_list = []
            for index, row in analysis_auc.iterrows():
                # stats 2 on differences between algs
                y = row[np.arange(0, 21).tolist()]
                x = np.arange(0, 21)
                auc_here = metrics.auc(x, y)
                auc_list.append(auc_here)
            aucs.append(np.array(auc_list))

            pref_df = pd.melt(pref_df, id_vars=['alg_type', 'id_here', 'typ'])
            dfs.append(pref_df)
        preference_df = pd.concat(dfs)

        palette_h = format_plot()
        palette_h = {'non-pref': '#F7C1AD', 'pref': '#35C4B9'}
        sns.lineplot(x='variable', y='value', hue='typ', data=preference_df, marker='o', legend=False,
                     palette=palette_h)
        plt.ylim(0, 1.5)
        ymarks = [0, 0.5, 1, 1.5]
        plt.yticks(ymarks)
        plt.ylabel('Simulated firing rate')
        plt.xlabel('Distance from start')
        plt.savefig('figures/fig6b' + alg + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
        plt.show()

    # t-test on difference between cell FR
    joint_inf_diff = pg.ttest(aucs[0], aucs[1], paired=True)
    SR_diff = pg.ttest(aucs[2], aucs[3], paired=True)

    joint_inf_aucs = aucs[0] - aucs[1]
    SR_aucs = aucs[2] - aucs[3]
    alg_diff = pg.ttest(joint_inf_aucs, SR_aucs)

    return joint_inf_diff, SR_diff, alg_diff


def split_cells(df3, task, pot_len, alg):
    nom = 'mean_probs_task' + str(task)
    list_of_lists = list(df3[nom].values)
    maxes = np.max(list_of_lists, axis=1)
    indices = np.argmax(list_of_lists, axis=1)
    nom = 'mean_cell_task' + str(task)
    mean_cell_task1 = list(df3[nom].values)
    max_cell_list = []
    max_cell_list_int = []

    for lit in np.arange(len(mean_cell_task1)):
        # for every agent lit is agent num
        ind = indices[lit]
        max_cell = mean_cell_task1[lit][ind] * maxes[lit]
        max_cell_list_int.append(max_cell)
    max_cell_list_1 = np.mean(max_cell_list_int, axis=0)

    # plotting splitter cell
    if pot_len == 20:
        if not noise:
            typ = 'max'
            plot_splitter(pot_len, max_cell_list_1, task, alg, typ)

    return list_of_lists, indices, max_cell_list_1, mean_cell_task1, maxes, max_cell_list_int


def record_splitter_cell(max_cell_list, pot_len):
    # record splitter cell
    split_cells = []
    for j in np.arange(len(max_cell_list)):
        split_cell = np.zeros(pot_len + 1)
        list_len = []
        for picked in np.arange(pot_len - 2):
            dist = 9 + picked
            list_len.append(dist)
        states = [0] + [1] + list_len + [2]
        counter = -1
        for state in states:
            counter = counter + 1
            split_cell[counter] = max_cell_list[j][state]
        split_cells.append(split_cell)
    return split_cells


def splitter_probabilities_plots(df_split, noise):
    # splitter cell actual average cell plots
    # splitter cell probabilities

    data_all = []
    task1_dfs = []
    task2_dfs = []
    key_names = []
    count = 0
    correlate_split_perf = []
    correlate_keys = []

    for alg in ['SR', 'joint_inf_priors']:
        df2 = df_split[df_split['lim_type'] == alg]
        for pot_len in [2, 5, 7, 10, 12, 15, 17, 20]:
            df3 = df2[df2['len'].isin([pot_len])]

            # find and plot max cells for task
            task = 1
            list_of_lists_1, indices_1, max_cell_list_1, mean_cell_task1, maxes_1, max_cell_list_all = split_cells(df3,
                                                                                                                   task,
                                                                                                                   pot_len,
                                                                                                                   alg)
            split_cell_1 = record_splitter_cell(max_cell_list_all, pot_len)
            task = 2
            list_of_lists_2, indices_2, max_cell_list_2, mean_cell_task2, maxes_2, max_cell_list_all = split_cells(df3,
                                                                                                                   task,
                                                                                                                   pot_len,
                                                                                                                   alg)
            split_cell_2 = record_splitter_cell(max_cell_list_all, pot_len)  # record splitter cell

            # plot firing of those cells on opposite task
            mins_2 = []
            min_cell_list = []
            for lit in np.arange(len(list_of_lists_1)):  # mean_prob_task1
                ind = indices_2[lit]
                mins_2.append(list_of_lists_1[lit][ind])
            for lit in np.arange(len(mean_cell_task1)):
                # for every agent lit is agent num
                ind = indices_2[lit]
                min_cell = mean_cell_task1[lit][ind] * mins_2[lit]
                min_cell_list.append(min_cell)
            min_cell_list_2 = np.mean(min_cell_list, axis=0)

            split_cell_mins_2 = record_splitter_cell(min_cell_list, pot_len)  # record splitter cell
            # plotting splitter cell
            if pot_len == 20:
                if not noise:
                    typ = 'min'
                    task = 2
                    plot_splitter(pot_len, min_cell_list_2, task, alg, typ)

            mins_1 = []
            min_cell_list = []
            for lit in np.arange(len(list_of_lists_2)):
                ind = indices_1[lit]
                mins_1.append(list_of_lists_2[lit][ind])
            for lit in np.arange(len(mean_cell_task1)):
                # for every agent lit is agent num
                ind = indices_1[lit]
                min_cell = mean_cell_task2[lit][ind] * mins_1[lit]
                min_cell_list.append(min_cell)
            min_cell_list_1 = np.mean(min_cell_list, axis=0)

            split_cell_mins_1 = record_splitter_cell(min_cell_list, pot_len)  # record splitter cell
            # plotting splitter cell
            if pot_len == 20:
                if not noise:
                    typ = 'min'
                    task = 1
                    plot_splitter(pot_len, min_cell_list_1, task, alg, typ)

            # Splitter cells
            split_1 = maxes_1 - mins_1
            split_2 = maxes_2 - mins_2

            d1 = pd.DataFrame(split_1)
            d2 = pd.DataFrame(split_2)

            num = np.shape(np.asarray(d1).T)[1]
            algie = [alg] * num
            lenie = [pot_len] * num
            taskie = [1] * num
            d1_i = pd.concat([d1, pd.DataFrame([algie, lenie, taskie]).T], axis=1)
            key_names = key_names + [str(count) + 'hi']
            count = count + 1

            d1_i.columns = ['data', 'alg', 'len', 'task']
            data_all.append(d1_i)

            id_here = df3['id_here']

            correlate_split_perf.append(
                pd.DataFrame(np.array([(split_1 + split_2) / 2, df3['percent_correct_100'], algie, lenie, id_here]).T,
                             columns=['split', 'perf', 'alg', 'len', 'id_here']))
            correlate_keys = correlate_keys + [str(count) + 'hi']

            num = np.shape(np.asarray(d2).T)[1]

            algie = [alg] * num
            lenie = [pot_len] * num
            taskie = [2] * num

            d2_i = pd.concat([d2, pd.DataFrame([algie, lenie, taskie]).T], axis=1)
            d2_i.columns = ['data', 'alg', 'len', 'task']

            data_all.append(d2_i)
            key_names = key_names + [str(count) + 'hi']
            count = count + 1

            col_names_1 = []
            col_names_2 = []
            for picked in np.arange(pot_len + 1):
                name = picked
                name_2 = picked
                col_names_1.append(name)
                col_names_2.append(name_2)

            task1_df = pd.DataFrame(np.array(split_cell_mins_1))
            task1_df.columns = col_names_1
            task1_df['alg'] = [alg] * num
            task1_df['len'] = [pot_len] * num
            task1_df['id'] = ['min'] * num
            task1_df['t'] = [1] * num
            task1_dfs.append(task1_df)

            task1_df = pd.DataFrame(np.array(split_cell_1))
            task1_df.columns = col_names_1
            task1_df['alg'] = [alg] * num
            task1_df['len'] = [pot_len] * num
            task1_df['id'] = ['max'] * num
            task1_df['t'] = [1] * num
            task1_dfs.append(task1_df)

            task2_df = pd.DataFrame(np.array(split_cell_mins_2))
            task2_df.columns = col_names_2
            task2_df['alg'] = [alg] * num
            task2_df['len'] = [pot_len] * num
            task2_df['id'] = ['min'] * num
            task2_df['t'] = [2] * num
            task2_dfs.append(task2_df)

            task2_df = pd.DataFrame(np.array(split_cell_2))
            task2_df.columns = col_names_2
            task2_df['alg'] = [alg] * num
            task2_df['len'] = [pot_len] * num
            task2_df['id'] = ['max'] * num
            task2_df['t'] = [2] * num
            task2_dfs.append(task2_df)

    data_dict = dict(zip(key_names, data_all))
    data_frame_here = pd.concat(data_dict, sort=False)
    data_frame_here.index.names = ['type', 'trial']
    df_new = data_frame_here.reset_index(['type', 'trial'])

    data_dict = dict(zip(correlate_keys, correlate_split_perf))
    data_frame_here = pd.concat(data_dict, sort=False)
    data_frame_here.index.names = ['type', 'trial']
    df_corr = data_frame_here.reset_index(['type', 'trial'])

    keys_task2_dfs = np.arange(len(task2_dfs))
    keys_task1_dfs = np.arange(len(task1_dfs))

    data_dict = dict(zip(keys_task2_dfs, task2_dfs))
    data_frame_here = pd.concat(data_dict, sort=False)
    data_frame_here.index.names = ['type', 'trial']
    df_new_2 = data_frame_here.reset_index(['type', 'trial'])

    data_dict = dict(zip(keys_task1_dfs, task1_dfs))
    data_frame_here = pd.concat(data_dict, sort=False)
    data_frame_here.index.names = ['type', 'trial']
    df_new_1 = data_frame_here.reset_index(['type', 'trial'])

    df_new_new = pd.concat([df_new_1, df_new_2])
    plotted = df_new_new[df_new_new['len'] == 20]

    # plotting fig 6d/e
    add = ''
    if noise:
        add = 'noise'
    palette_here = format_plot()
    sns.lineplot(x='len', y='data', hue='alg', data=df_new, marker='o', legend=False, palette=palette_here)
    plt.ylabel('Difference in splitter probabilities')
    plt.ylim(0, 1)
    if noise:
        plt.xlabel('Distractors around cue')
    else:
        plt.xlabel('Distance from cue to choice')
    plt.savefig('figures/fig6d' + add + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    # plotting correlation plots, fig 5e/f
    palette_here = format_plot()

    df_corr.perf = df_corr.perf.astype(float)
    df_corr.split = df_corr.split.astype(float)
    df_corr.len = df_corr.len.astype(float)
    alg_plot = df_corr[df_corr['alg'] == 'SR']
    sns.lmplot(x='perf', y='split', data=alg_plot, hue='alg', palette=palette_here, legend=False)
    plt.xlabel('Performance on random trials')
    plt.ylabel('Difference in splitter probabilities')
    plt.xlim(25, 100)
    plt.ylim(-0.05, 0.9)
    plt.savefig('figures/fig6corr' + add + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    # Statistics for 5d/e left
    stats_me = df_corr[['id_here', 'len', 'alg', 'split']]
    stats_me.columns = ['id_here', 'len', 'alg_type', 'split']
    stats_out, post_hoc_out = anova_mixed_effects(stats_me, 'split')

    # Statisticss for 5d/e right
    x = np.array(alg_plot['perf']).reshape((-1, 1))
    y = np.array(alg_plot['split'])
    regression_output = regression(x, y)

    return regression_output, stats_out, post_hoc_out


def plot_splitter_ev(df_prob_split1, df_prob_split2):
    df = df_prob_split1
    df_new = df[df['len'] == 20]
    df_new2 = df_new.loc[:, df_new.columns != 18]
    plt_me = pd.melt(df_new2, id_vars=['type', 'trial', 'alg_type', 'len', 'id_here'])
    plt_me['variable'] = plt_me['variable'] + 1
    plt.ylim(0, 1)
    palette_here = format_plot()
    sns.lineplot(x='variable', y='value', hue='alg_type', data=plt_me, marker='o', legend=False, palette=palette_here)
    plt.ylabel('Difference in splitter probabilities')
    plt.xlabel('Distance from cue')
    plt.savefig('figures/fig6csplitcellloss' + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    stats_me = plt_me[['id_here', 'variable', 'value', 'alg_type']]
    stats_me.columns = ['id_here', 'len', 'value', 'alg_type']
    stats_out, post_hoc_out = anova_mixed_effects(stats_me, 'value')
    return stats_out, post_hoc_out


def odour_analysis(df, data_type):
    palette_plots = format_plot()
    df2 = df[df['len'] == 3]

    # Block plots
    ax = sns.barplot(x='lim_type', y='trials_on_reversal', data=df2, errorbar='ci', palette=palette_plots, order=['SR', 'joint_inf_priors'])
    ax.set_xticklabels(['feature', 'joint'])
    plt.ylabel('Attempts on block switches')
    ax.get_legend().set_visible(False)
    plt.xlabel('Trial number')
    plt.tight_layout()
    plt.savefig('figures/fig7_odour_task_block' + data_type + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    #stats
    df3 = df2
    stats_me = df3[['id_here', 'len', 'alg_type', 'trials_on_reversal']]
    x_int = stats_me[stats_me['alg_type'] == 'SR']
    y_int = stats_me[stats_me['alg_type'] == 'joint_inf_priors']
    x = x_int['trials_on_reversal'].values.tolist()
    y = y_int['trials_on_reversal'].values.tolist()
    if data_type=='DNMS':
        block_stats = pg.ttest(x, y)
    else:
        block_stats = pg.mwu(x, y)


    # Random plots
    df_rands = df2[
        ['lim_type', 'id_here', 'percent_correct_500', 'percent_correct_400', 'percent_correct_300',
         'percent_correct_200',
         'percent_correct_100']]
    melty = df_rands.melt(['lim_type', 'id_here'])

    melty['variable'] = melty['variable'].map(
        {'percent_correct_500': 0, 'percent_correct_400': 100, 'percent_correct_300': 200,
         'percent_correct_200': 300,
         'percent_correct_100': 400})

    ax = sns.lineplot(x='variable', y='value', hue='lim_type', data=melty, errorbar='ci', palette=palette_plots)
    plt.ylabel('Percent correct random trials')
    plt.ylim(45, 80)
    ax.get_legend().set_visible(False)
    plt.xlabel('Trial number')
    plt.tight_layout()
    plt.savefig('figures/fig7_odour_task_rand' + data_type + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
    plt.show()

    # stats
    melty = melty.reset_index()
    stats_me = melty
    stats_me['value'] = stats_me['value'].astype('float')

    stats_out = pg.mixed_anova(dv='value', between='lim_type', within='variable', subject='id_here', data=stats_me)
    post_hoc_out = pg.pairwise_tests(dv='value', between='lim_type', within='variable', subject='id_here',
                                     data=stats_me,
                                     padjust='fdr_bh', parametric=True)

    return block_stats, stats_out, post_hoc_out


def regress_SR(df):
    df1 = df[df['alg_type'] == 'SR']
    x = np.array(df1['len']).reshape((-1, 1))
    y = np.array(df1['percent_correct_100'])
    regression_output = regression(x, y)
    return regression_output


if __name__ == '__main__':

    # To generate plots enter data_type and import folder location:
    data_type = 'struct_disc'  # in cue_SRstart, noise_SRstart, noise, cue, fig_1, other_noise, other_cue, struct_disc, DNMS
    import_folder = 'Data/'  # folder/file management is setup for mac

    # each data_type uses a different set of output files from the algorithm to run, see comments within for which figures will be generated
    if data_type == 'cue_SRstart':
        # this dataset has all of the data in it apart from outcome
        df, df_prob1, df_prob2, noise, df_prob_split1, df_prob_split2, df_split1, df_split2, df_split3, df_split4, df_control1, df_control2 = import_dat(
            data_type='cue_SRstart', saved=True, import_folder=import_folder)
        # fig 5b
        plot_mean_SR(df, noise)
        # fig 5c
        stats, post_hoc = cue_preds(df, noise)

        # fig 6a/b
        joint_inf_diff, SR_diff, alg_diff = plot_splitter_rl(df)
        # print(stats_out['p-unc'])

        # fig 6c
        stats_out, post_hoc_out = plot_splitter_ev(df_prob_split1, df_prob_split2)

        # fig 6d
        regression_output, stats_out, post_hoc_out = splitter_probabilities_plots(df, noise)
        # print(regression_output.summary())
        # print(regression_output.pvalues[1])
        # print(stats_out['p-unc'])

    if data_type == 'noise_SRstart':
        # this dataset has all of the data in it apart from outcome
        df, df_prob1, df_prob2, noise, df_prob_split1, df_prob_split2, df_split1, df_split2, df_split3, df_split4, df_control1, df_control2 = import_dat(
            data_type='noise_SRstart', saved=True, import_folder=import_folder)
        # generate plots
        # figure 2e
        plot_fig2_SRs(df)
        # fig 5b
        plot_mean_SR(df, noise)
        # fig 5c
        stats, post_hoc = cue_preds(df, noise)
        # fig 6e
        regression_output, stats_out, post_hoc_out = splitter_probabilities_plots(df, noise)
        # print(regression_output.summary())
        # print(regression_output.pvalues[1])
        # print(stats_out['p-unc'])

    if data_type == 'noise':
        df, df_prob1, df_prob2, noise, df_prob_split1, df_prob_split2, df_split1, df_split2, df_split3, df_split4, df_control1, df_control2 = import_dat(
            data_type='noise', saved=True, import_folder=import_folder)
        # fig 3/4
        stats_out, post_hoc_out = fig_3(df, noise)
        # fig 3 regression
        regression_output = regress_SR(df)
        # fig 5a
        stats_out, post_hoc_out = plot_steps(df, noise)
        # fig 5d
        fig = '5d'
        stats_out, post_hoc_out = plot_maintenance(df_prob1, df_prob2, fig)

        # Save processed data on first pass:
        # save_import(df, df_prob1, df_prob2, noise, df_prob_split1, df_prob_split2, df_split1, df_split2, df_split3,df_split4, df_control1, df_control2, data_type, import_folder)

    if data_type == 'cue':
        # process cue data
        df, df_prob1, df_prob2, noise, df_prob_split1, df_prob_split2, df_split1, df_split2, df_split3, df_split4, df_control1, df_control2 = import_dat(
            data_type='cue', saved=True, import_folder=import_folder)
        # fig 3/4
        stats_out, post_hoc_out = fig_3(df, noise)
        # fig 3 regression
        regression_output = regress_SR(df)
        # fig 5a
        stats_out, post_hoc_out = plot_steps(df, noise)
        # fig 5d
        fig = '5d'
        stats_out, post_hoc_out = plot_maintenance(df_prob1, df_prob2, fig)

        # save_import(df, df_prob1, df_prob2, noise, df_prob_split1, df_prob_split2, df_split1, df_split2, df_split3,df_split4, df_control1, df_control2, data_type, import_folder)

    if data_type == 'fig_1':
        # plots fig 2c,d
        df = import_dat_simple(
            data_type='fig_1', saved=True, import_folder=import_folder)
        stats_out, post_hoc_out, stats_out_2, post_hoc_out_2 = plot_fig2(df)

    if data_type == 'other_noise':
        # supplementary figures 1/2
        df = import_dat_simple(data_type='other_noise', saved=True, import_folder=import_folder)
        stats_me = df[['id_here', 'len', 'alg_type', 'percent_correct_100']]
        stats_out, post_hoc_out = anova_mixed_effects(stats_me, 'percent_correct_100')
        palette_plots = format_plot()
        for plot_me in ['reward_feature', 'forced_choice', 'explore', 'ideal_observer', 'replay']:
            df2 = df[df['alg_type'].isin(['SR', 'joint_inf_priors', plot_me])]
            sns.lineplot(x='len', y='percent_correct_100', hue='alg_type', data=df2, marker='o', legend=False,
                         palette=palette_plots)
            plt.ylim(40, 100)
            plt.xlabel('Distractors around cue')  # 'Distractors around cue'
            plt.ylabel('Performance on random trials')
            plt.savefig('figures/supp_noise' + plot_me + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
            plt.show()
        # save_import_simple(df, import_folder)

    if data_type == 'other_cue':
        # supplementary figures 1/2
        df = import_dat_simple(data_type='other_cue', saved=True, import_folder=import_folder)
        stats_me = df[['id_here', 'len', 'alg_type', 'percent_correct_100']]
        stats_out, post_hoc_out = anova_mixed_effects(stats_me, 'percent_correct_100')
        palette_plots = format_plot()
        for plot_me in ['reward_feature', 'forced_choice', 'explore', 'ideal_observer', 'replay']:
            df2 = df[df['alg_type'].isin(['SR', 'joint_inf_priors', plot_me])]
            sns.lineplot(x='len', y='percent_correct_100', hue='alg_type', data=df2, marker='o', legend=False,
                         palette=palette_plots)
            plt.ylim(40, 100)
            plt.xlabel('Distance from cue to choice')
            plt.ylabel('Performance on random trials')
            plt.savefig('figures/supp_cue' + plot_me + '.pdf', dpi=500, bbox_inches='tight', format='pdf')
            plt.show()
        # save_import_simple(df, import_folder)

    if data_type == 'DNMS':
        # import
        import_folder = '/Users/jpasslack/Documents/transfer_to_cluster/odour_paper/'
        df = import_dat_simple(data_type='struct_disc', saved=False, import_folder=import_folder)
        #import_folder = 'Data/'
        #save_import_simple(df, import_folder)

        # fig 7b/c
        block_stats, stats_out, post_hoc_out = odour_analysis(df, block, data_type)

    if data_type == 'struct_disc':
        # import
        import_folder = "/Users/jpasslack/Documents/transfer_to_cluster/odour_task_rerun/"  # (lim=1350) or is it "/Users/jpasslack/Documents/transfer_to_cluster/O5/"? (lim=1400)
        df = import_dat_simple(data_type='struct_disc', saved=False, import_folder=import_folder)
        #import_folder = 'Data/'
        #save_import_simple(df, import_folder)

        # fig 7e/f
        block_stats, stats_out, post_hoc_out = odour_analysis(df, block, data_type)
