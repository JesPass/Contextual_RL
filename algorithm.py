import numpy as np
import random
from scipy.stats import norm
from scipy.stats import multivariate_normal
import pickle
import itertools


def agent_run(learner, rewards, world):
    state_new, reward_new = world.next_state(learner, rewards)
    if not world.passive:
        feature_iw = 0
        learner.update_estimates(world, reward_new, state_new, feature_iw)  # task inference, based on new observations
        learner.calc_state_action_values()
        action_new = learner.pick_action(learner.state_action_values,
                                         world)  # one-step lookahead, needed to update with TD rules
    elif world.passive:
        learner.update_estimates(world, reward_new, state_new,
                                 action_new=0)  # task inference, based on new observations
    learner.update_lists(action_new)  # record all the updated estimates for the next run
    return learner


def episode_run(world, learner, ret_n_trials, ret_perf, CR_record):
    task_tot = 0

    prob_record_start = []
    state_record_start = []
    prob_record_end = []
    state_record_end = []
    prob_record_rand = []
    state_record_rand = []
    CR_recorded = []
    prob_record_CR = []
    state_record_all = []

    for task_num in world.tasks:
        rewards, starting_state = world.create_task(task_num)
        learner.starting_state = starting_state

        prob_record_start_int = []
        state_record_start_int = []
        prob_record_end_int = []
        state_record_end_int = []
        prob_record_rand_int = []
        state_record_rand_int = []
        CR_map_int = []
        prob_record_all_int = []
        state_record_all_int = []

        performance = 0
        n_trial = 0
        perform_ave = 0
        task_tot = task_tot + 1
        if world.one_shot:
            while perform_ave < world.trials:
                prob_record_start_int2 = []
                state_record_start_int2 = []
                prob_record_end_int2 = []
                state_record_end_int2 = []
                state_record_rand_int2 = []
                prob_record_rand_int2 = []
                CR_map_int2 = []
                prob_record_all_int2 = []
                state_record_all_int2 = []

                rewards, starting_state = world.create_task(task_num)
                learner.reset(world)
                n_trial = n_trial + 1
                # print(n_trial)
                # exploration phase without rewards
                learner.counter = 0
                if task_num == 100000000:
                    while learner.counter < 10000:
                        learner = agent_run(learner, rewards, world)
                    learner.default_SR = learner.SR
                    default_SR = np.copy(learner.default_SR)
                    learner.SR_all = [np.copy(default_SR), np.copy(default_SR), np.copy(default_SR),
                                      np.copy(default_SR), np.copy(default_SR), np.copy(default_SR),
                                      np.copy(default_SR), np.copy(default_SR), np.copy(default_SR),
                                      np.copy(default_SR)]
                    perform_ave = world.trials

                # phase with rewards
                else:
                    world.step_in_episode = 0
                    if world.passive:
                        while learner.counter < (np.shape(world.states)[1] - 1):
                            learner = agent_run(learner, rewards, world)
                    else:
                        flat_state = np.where(learner.state == 1)[0][0]
                        while flat_state not in world.goal_state:
                            learner = agent_run(learner, rewards, world)
                            flat_state = np.where(learner.state == 1)[0][0]
                            if task_tot < 200:
                                prob_record_start_int.append(learner.importance_weights)
                                state_record_start_int.append(flat_state)
                            if task_tot >= 800 & task_tot < 1000:
                                prob_record_end_int.append(learner.importance_weights)
                                state_record_end_int.append(flat_state)
                            if task_tot >= 1400:
                                # record data from last 100 random trials
                                state_record_rand_int.append(flat_state)  # location
                                prob_record_rand_int.append(learner.importance_weights)  # probability of tasks\
                            if (task_tot >= 500) & (task_tot < 1000):
                                if CR_record:
                                    CR_map_int.append(learner.CR_maps_all)
                            prob_record_all_int.append(learner.importance_weights)
                            state_record_all_int.append(flat_state)

                        if n_trial >= world.trials:
                            perform_ave = sum(learner.correct_visits[-world.trials:])
                        performance = sum(learner.correct_visits[-n_trial:])

                        ret_perf.append(learner.correct_visits[-1])

                        if task_tot < 200:
                            prob_record_start_int2.append(prob_record_start_int)
                            state_record_start_int2.append(state_record_start_int)
                        if task_tot >= 800 & task_tot < 1000:
                            prob_record_end_int2.append(prob_record_end_int)
                            state_record_end_int2.append(state_record_end_int)
                        if task_tot >= 1400:
                            # record data from last 100 random trials
                            state_record_rand_int2.append(state_record_rand_int)  # location
                            prob_record_rand_int2.append(prob_record_rand_int)  # probability of tasks
                        if (task_tot >= 500) & (task_tot < 1000):
                            if CR_record:
                                CR_map_int2.append(CR_map_int)
                        prob_record_all_int2.append(prob_record_all_int)
                        state_record_all_int2.append(state_record_all_int)

            ret_n_trials.append(n_trial)
        else:
            for i in np.arange(world.trials):
                # print(i)
                learner.reset(world)
                # exploration phase without rewards
                learner.counter = 0
                if task_num == 100000000:
                    while learner.counter < 1000:
                        learner = agent_run(learner, rewards, world)
                # phase with rewards
                else:
                    world.step_in_episode = 0
                    if world.passive:
                        while learner.counter < (np.shape(world.states)[1] - 1):
                            learner = agent_run(learner, rewards, world)
                    else:
                        flat_state = np.where(learner.state == 1)[0][0]
                        while flat_state not in world.goal_state:
                            learner = agent_run(learner, rewards, world)
                            flat_state = np.where(learner.state == 1)[0][0]

        if task_tot < 200:
            prob_record_start.append(prob_record_start_int2)
            state_record_start.append(state_record_start_int2)
        if task_tot >= 800 & task_tot < 1000:
            prob_record_end.append(prob_record_end_int2)
            state_record_end.append(state_record_end_int2)
        if task_tot >= 1400:
            # record data from last 100 random trials
            state_record_rand.append(state_record_rand_int2)  # location
            prob_record_rand.append(prob_record_rand_int2)  # probability of tasks
        if (task_tot >= 500) & (task_tot < 1000):
            if CR_record:
                CR_recorded.append(CR_map_int2)  # CR maps on last 100 random trials

        prob_record_CR.append(prob_record_all_int2)  # records probabilities
        state_record_all.append(state_record_all_int2)

    learner.analyze_me(world)
    return ret_perf, ret_n_trials, prob_record_start, state_record_start, prob_record_end, state_record_end, state_record_rand, prob_record_rand, CR_recorded, prob_record_CR, state_record_all


def dual_agent_run(learner, learner_feature, rewards, world, SR_error, n_trial, task_type):
    if learner.task_tot > world.limit:
        world.forced_choice = False

    state_new, reward_new = world.next_state(learner_feature,
                                             rewards)  # make sure the action picked corresponds to feature action vals

    # Do feature inference
    learner_feature.update_estimates(world, reward_new, state_new)  # task inference, based on new observations
    if world.joint_prior:
        if learner.task_tot <= world.limit:
            # if world.do_joint:
            if task_type == 'struct_disc':
                if learner.task_tot >= 250 + 150:
                    learner.particles = np.copy(learner_feature.particles)
            elif task_type == 'DNMS':
                if learner.task_tot >= 550:
                    learner.particles = np.copy(learner_feature.particles)  # sharing of particles
            else:
                learner.particles = np.copy(learner_feature.particles)  # sharing of particles

    # #Pass feature inference to outcome inference  algorithm
    learner.update_estimates(world, reward_new, state_new,
                             learner_feature.importance_weights)  # task inference, based on new observations
    if world.joint_prior:
        if learner.task_tot <= world.limit:
            if task_type == 'struct_disc':
                if learner.task_tot >= 250 + 150:
                    learner_feature.particles = np.copy(learner.particles)  # sharing of particles
            elif task_type == 'DNMS':
                if learner.task_tot >= 550:
                    learner_feature.particles = np.copy(learner.particles)
            else:
                learner_feature.particles = np.copy(learner.particles)  # sharing of particles

    if world.joint_inf:
        # if world.do_joint:
        if learner.task_tot <= world.limit:
            if task_type == 'struct_disc':
                if learner.task_tot >= 250 + 150:
                    learner_feature.current_task = learner.current_task  # this is the joint inferred task from end of prev ep
                    learner_feature.confidence = learner.confidence
            elif task_type == 'DNMS':
                if learner.task_tot >= 550:
                    learner_feature.current_task = learner.current_task  # this is the joint inferred task from end of prev ep
                    learner_feature.confidence = learner.confidence
            else:
                learner_feature.current_task = learner.current_task  # this is the joint inferred task from end of prev ep
                learner_feature.confidence = learner.confidence
        # basically until you are confident in a map use outcomee inference to drive hpc learning and actions, as soon as there is a map associated with an outcome map use it

    # Update SR maps and choose correct one for action selection

    learner_feature.update_SR(state_new, reward_new, world)
    learner.update_SR(state_new, reward_new, world)  # transfer updated SR and reward to outcome model

    # pick next action
    learner_feature.calc_state_action_values()
    action_new = learner_feature.pick_action(learner_feature.state_action_values,
                                             world)  # one-step lookahead, needed to update with TD rules

    # update lists of info; not sure this records anything important atm tbh
    learner.update_lists(action_new)  # record all the updated estimates for the next run
    learner_feature.update_lists(action_new)

    return learner, learner_feature


def dual_episode_run(world, learner, learner_feature, ret_n_trials, ret_perf, record_SRs):
    task_tot = 0
    SR_error = 1
    err = []
    out_conf_list = []
    feat_conf_list = []

    SR_record = []
    state_record = []
    prob_record = []

    prob_record_start = []
    state_record_start = []
    SR_record_start = []

    prob_record_end = []
    state_record_end = []

    state_record_all = []
    prob_record_SR = []
    prob_record_CR = []

    world.current_act_task = 1

    default_counter = 0

    for task_num in world.tasks:

        if 100000000 in world.tasks:
            # transfer default map for second task
            if task_num == 2 & default_counter == 0:
                learner_feature.SR_all[1] = np.copy(default_SR)
                learner_feature.covariance_all[1] = np.copy(co)
                default_counter = 1

        rewards, starting_state = world.create_task(task_num)
        learner.starting_state = starting_state
        learner_feature.starting_state = starting_state
        performance = 0
        n_trial = 0
        perform_ave = 0
        task_tot = task_tot + 1
        world.task_tot = task_tot
        learner.task_tot = task_tot

        # exploration phase without rewards

        if task_num == 100000000:
            learner_default = SR_SF_Agent(world, starting_state=starting_state, kalman_SR=True, kalman=False, TD=False,
                                          kalman_TD=False)
            learner_default.counter = 0
            learner_default.reset(world)
            while learner_default.counter < 10000:
                # print(learner_default.counter)
                learner_default = agent_run(learner_default, rewards, world)
            learner_default.default_SR = learner_default.SR
            default_SR = np.copy(learner_default.default_SR)
            co = np.copy(learner_default.covariance)
            # set covariance and SR for task 1
            learner_feature.SR_all[0] = np.copy(default_SR)
            learner_feature.covariance_all[0] = np.copy(co)
            continue

        SR_record_int2 = []
        state_record_int2 = []
        prob_record_int2 = []

        prob_record_start_int2 = []
        state_record_start_int2 = []
        SR_record_start_int2 = []

        prob_record_end_int2 = []
        state_record_end_int2 = []

        state_record_all_int2 = []
        prob_record_SR_int2 = []
        prob_record_CR_int2 = []

        while perform_ave < world.trials:
            learner.reset(world)
            learner_feature.reset(world)
            n_trial = n_trial + 1
            # print(n_trial)
            # print(np.max(learner_feature.raw_prob))
            # exploration phase without rewards
            learner.counter = 0

            world.step_in_episode = 0
            flat_state = np.where(learner.state == 1)[0][0]

            SR_record_int = []
            state_record_int = []
            prob_record_int = []

            prob_record_start_int = []
            state_record_start_int = []
            SR_record_start_int = []

            prob_record_end_int = []
            state_record_end_int = []

            state_record_all_int = []
            prob_record_SR_int = []
            prob_record_CR_int = []

            while flat_state not in world.goal_state:
                learner, learner_feature = dual_agent_run(learner, learner_feature, rewards, world, SR_error, n_trial,
                                                          world.inf_task)
                flat_state = np.where(learner.state == 1)[0][0]
                if task_tot >= 1400:
                    # record data from last 100 random trials
                    if record_SRs:
                        SR_record_int.append(learner_feature.SR_all.copy())  # SRs
                    else:
                        SR_record_int.append([])
                    state_record_int.append(flat_state)  # location
                    prob_record_int.append(learner_feature.importance_weights)  # probability of tasks
                if task_tot < 200:
                    prob_record_start_int.append(learner_feature.importance_weights)
                    state_record_start_int.append(flat_state)
                    if record_SRs:
                        SR_record_start_int.append(learner_feature.SR_all.copy())  # SRs
                    else:
                        SR_record_start_int.append([])
                if task_tot >= 800 & task_tot < 1000:
                    prob_record_end_int.append(learner_feature.importance_weights)
                    state_record_end_int.append(flat_state)
                state_record_all_int.append(flat_state)  # locations
                prob_record_SR_int.append(learner_feature.importance_weights)  # records probabilities
                prob_record_CR_int.append(learner.importance_weights)  # records probabilities
            if n_trial >= world.trials:
                perform_ave = sum(learner.correct_visits[-world.trials:])
            performance = sum(learner.correct_visits[-n_trial:])
            ret_perf.append(learner.correct_visits[-1])

            if learner.correct_visits[-1] == 0:
                learner.feature_states_for_ret = []

            # attempt to lose the limit thing
            # learner_feature.ave_resid.append(np.mean(learner_feature.running_resid))
            # learner_feature.running_resid = []

            # if len(learner_feature.ave_resid) >= 5:
            #     # check over last 5 trials for accuracy of SR
            #     if np.mean(learner_feature.ave_resid[-5:]) < 1.05:
            #         # if error less than 5% then don't use joint
            #         world.do_joint = False
            #     else:
            #         world.do_joint = True
            # else:
            #     world.do_joint = True
            #
            # learner.do_joint = world.do_joint
            if task_tot >= 1400:
                # record data from last 100 random trials
                SR_record_int2.append(SR_record_int)  # SRs
                state_record_int2.append(state_record_int)  # location
                prob_record_int2.append(prob_record_int)  # probability of tasks
            if task_tot < 200:
                prob_record_start_int2.append(prob_record_start_int)
                state_record_start_int2.append(state_record_start_int)
                SR_record_start_int2.append(SR_record_start_int)
            if task_tot >= 800 & task_tot < 1000:
                prob_record_end_int2.append(prob_record_end_int)
                state_record_end_int2.append(state_record_end_int)
            state_record_all_int2.append(state_record_all_int)  # locations
            prob_record_SR_int2.append(prob_record_SR_int)  # records probabilities
            prob_record_CR_int2.append(prob_record_CR_int)  # records probabilities

        # #retrograde replay states and rewards at end of all episodes
        if world.replay:
            if learner.task_tot < world.limit:
                learner_feature.current_task = learner.current_task
                learner_feature.confidence = learner.confidence
                learner_feature.retrograde_update(learner_feature.states_for_ret, learner_feature.feat_conf_list,
                                                  learner.out_conf_list, world)

        if task_tot >= 1400:
            # record data from last 100 random trials
            SR_record.append(SR_record_int2)  # SRs
            state_record.append(state_record_int2)  # location
            prob_record.append(prob_record_int2)  # probability of tasks
        if task_tot < 200:
            prob_record_start.append(prob_record_start_int2)
            state_record_start.append(state_record_start_int2)
            SR_record_start.append(SR_record_start_int2)
        if task_tot >= 800 & task_tot < 1000:
            prob_record_end.append(prob_record_end_int2)
            state_record_end.append(state_record_end_int2)

        state_record_all.append(state_record_all_int2) # locations
        prob_record_SR.append(prob_record_SR_int2)  # records probabilities
        prob_record_CR.append(prob_record_CR_int2)  # records probabilities

        ret_n_trials.append(n_trial)
        world.current_act_task = task_num

    learner.analyze_me(world)
    return ret_perf, ret_n_trials, out_conf_list, feat_conf_list, err, SR_record, state_record, prob_record, prob_record_start, state_record_start, prob_record_end, state_record_end, SR_record_start, state_record_all, prob_record_SR, prob_record_CR


class Agent:
    def __init__(self, world, starting_state, kalman_SR, kalman, kalman_TD):
        self.states = []  # where have i been
        self.SA_list = []  # state action values
        self.reward_list = []  # expected reward list
        # self.SA_list = [[[] for i in range(self.n_actions)] for i in range(self.n_states)]  # state action values
        # self.reward_list = [[] for i in range(self.n_states)]  # learned rewards for states
        self.n_states = world.n_states
        self.n_actions = world.n_actions
        self.covariance = np.diag(np.ones(self.n_states))
        self.state = starting_state
        self.starting_state = starting_state
        if not world.passive:
            self.action = self.first_action()
        self.alpha_RW = 0.7  # learning rate for RW reward vector
        if kalman:  # unlearning vs relearning in 1 trial
            self.alpha_RW = 1.0
        self.alpha = 0.1  # learning rate for TD reward vector, SR matrix
        self.discount_factor = 0.9  # discount factor for TD reward vector, SR matrix
        self.epsilon = 0.2  # explore-exploit ratio # for Q-learning #0.2 from paper #changed
        #        if not world.passive:
        #            self.state_action_pair = int(self.state * self.n_actions + self.action)
        if not world.passive:
            self.state_action_values = np.zeros((self.n_states * self.n_actions))
        self.counter_list = []
        self.counter = 0
        self.default = False
        self.passive = False
        self.feature = False
        if world.passive:
            self.passive = True
        self.covariance_default = np.identity(self.n_states)
        self.covariance = np.copy(self.covariance_default)
        self.covariance_outcome = np.copy(self.covariance_default)
        self.kalman_gain_default = np.zeros((self.n_states, 1))
        self.kalman_gain = np.copy(self.kalman_gain_default)
        self.kalman_gain_outcome = np.copy(self.kalman_gain_default)
        self.kalman = kalman
        self.kalman_SR = kalman_SR
        self.task_tot = 0
        self.running_resid = []

        self.kalman_TD = kalman_TD
        if self.kalman_TD:
            self.covariance_TD = np.copy(self.covariance_default) * 0.6  # 0.6
            self.kalman_gain_TD = np.copy(self.kalman_gain_default)
            self.residual_covariance_TD = 1.6  # 1.6

    def update_lists(self):
        placeholder = 0

    def reset(self, world):
        self.state = self.starting_state  # teleport to starting state after each ep
        if not world.passive:
            self.action = self.pick_action(self.state_action_values,
                                           world)
        self.update_lists(self.action)

    def first_action(self):
        # action probabilities, given state
        global action
        prob_actions = np.zeros(self.n_actions)
        prob_actions[:] = 1 / self.n_actions
        rand_num = random.random()
        cum_prob = 0
        for i in np.arange(self.n_actions):
            cum_prob = prob_actions[i] + cum_prob
            if cum_prob > rand_num:
                action = i
                break
        return action

    def pick_action(self, state_action_values, world):
        rand_num = random.random()
        if rand_num <= self.epsilon:
            possible_actions = np.arange(self.n_actions)
            if world.inf_task in ['struct_disc', 'DNMS']:
                if self.state[2] == 1:
                    possible_actions = np.array([2, 3])
            action = random.choice(possible_actions)
        else:
            vals = state_action_values[(self.state * self.n_actions):(
                    self.state * self.n_actions + self.n_actions)]
            possible_actions = np.argwhere(vals == np.amax(vals))
            if world.inf_task in ['struct_disc', 'DNMS']:
                if self.state[2] == 1:
                    possible_actions = np.argwhere(vals == np.amax(vals[2:]))
                    possible_actions = possible_actions[possible_actions > 1]
            action = random.choice(possible_actions)

        return action

    def kalman_filter_CR(self, CR_state, CR_obs):
        inp = np.zeros((self.n_states, 1))
        inp[CR_state] = 1
        diffusion_variance = 0
        outcome_noise_variance = 1
        drifted_covariance = self.covariance_outcome + np.identity(self.n_states) * diffusion_variance
        self.residual_covariance_outcome = inp.T @ drifted_covariance @ inp + outcome_noise_variance
        self.kalman_gain_outcome = (drifted_covariance @ inp) / self.residual_covariance_outcome
        self.covariance_outcome = drifted_covariance - self.kalman_gain_outcome @ inp.T @ drifted_covariance

    def kalman_filter_SR(self, state_new, world):
        diffusion_variance = 0
        outcome_noise_variance = 1
        if world.inf_task in ['struct_disc', 'DNMS']:
            outcome_noise_variance = 0.3
        discounted_state_derivative = self.state - self.discount_factor * state_new
        drifted_covariance = self.covariance + np.identity(self.n_states) * diffusion_variance
        self.residual_covariance = \
            (discounted_state_derivative.T @ drifted_covariance @ discounted_state_derivative + outcome_noise_variance)[
                0][
                0]
        self.kalman_gain = (drifted_covariance @ discounted_state_derivative) / self.residual_covariance
        self.covariance = drifted_covariance - self.residual_covariance * self.kalman_gain @ self.kalman_gain.T
        # self.running_resid.append(np.copy(self.residual_covariance))

    def mini_SR_filter(self, state_new, j):
        diffusion_variance = 0
        outcome_noise_variance = 1
        discounted_state_derivative = self.state - self.discount_factor * state_new
        drifted_covariance = self.covariance_all[j] + np.identity(self.n_states) * diffusion_variance
        self.residual_covariance_all[j] = np.copy( \
            (discounted_state_derivative.T @ drifted_covariance @ discounted_state_derivative + outcome_noise_variance)[
                0][
                0])

    def kalman_filter_TD(self, state):
        inp = state
        diffusion_variance = 0
        outcome_noise_variance = 1
        drifted_covariance = self.covariance_TD + np.identity(self.n_states) * diffusion_variance
        self.residual_covariance_TD = inp.T @ drifted_covariance @ inp + outcome_noise_variance
        self.kalman_gain_TD = (drifted_covariance @ inp) / self.residual_covariance_TD
        self.covariance_TD = drifted_covariance - self.kalman_gain_TD @ inp.T @ drifted_covariance


class SR_SF_Agent(Agent):
    # learns with SR of state, with one step look-ahead
    # TO DO
    # make sure all inputs state, state_new ([[]]) double bracketed
    # make sure all np.zeros are (n,1) shape
    def __init__(self, world, starting_state, kalman_SR, kalman, TD, kalman_TD):
        super().__init__(world, starting_state, kalman_SR, kalman, kalman_TD)
        self.reward_weights = np.zeros((self.n_states, 1))
        self.SR_list = []  # learned SR maps
        self.SR = np.diag(np.ones(self.n_states))
        self.state_lookahead = np.zeros(self.n_actions)
        self.state_action_values = np.zeros((self.n_states, 1))
        self.correct_visits = [0]
        self.retro = False
        self.multi = False
        self.TD_arg = TD
        if self.TD_arg:
            self.TD = np.zeros((self.n_states, 1))
        self.importance_weights = []

    def pick_action(self, state_action_values, world):
        vals = np.zeros(self.n_actions)
        # explore_bonus = np.zeros(self.n_actions)
        for action in range(0, 4):
            state_flat = np.where(self.state == 1)[0][0]
            state_lookahead = world.one_step_lookahead(state_flat,
                                                       action)
            # explore_bonus[action] = np.diag(self.covariance) @ state_lookahead #explore actions that lead to states with higher variance ie less explored states
            if not self.TD_arg:
                vals[action] = self.state_action_values.T @ state_lookahead
            if self.TD_arg:
                vals[action] = (self.state_action_values.T @ state_lookahead)[0]
        rand_num = random.random()
        # explore_bonus = explore_bonus / np.sum(explore_bonus)  # normalize to sum to 1
        if rand_num <= self.epsilon:
            possible_actions = np.arange(self.n_actions)
            if world.inf_task in ['struct_disc', 'DNMS']:
                if self.state[2] == 1:
                    possible_actions = np.array([2, 3])
            action = random.choice(possible_actions)
        else:
            possible_actions = np.argwhere(vals == np.amax(vals))
            if world.inf_task in ['struct_disc', 'DNMS']:
                if self.state[2] == 1:
                    possible_actions = np.argwhere(vals == np.amax(vals[2:]))
                    possible_actions = possible_actions[possible_actions > 1]
            if len(possible_actions) == 0:
                possible_actions = np.arange(self.n_actions)
                if world.inf_task in ['struct_disc', 'DNMS']:
                    if self.state[2] == 1:
                        possible_actions = np.array([2, 3])
                action = random.choice(possible_actions)
            else:
                if world.inf_task in ['struct_disc', 'DNMS']:
                    if self.state[2] == 1:
                        action = random.choice(possible_actions)
                    else:
                        action = random.choice(possible_actions[:, 0])
                else:
                    action = random.choice(possible_actions[:, 0])
        return action

    def calc_state_action_values(self):
        self.state_action_values = self.SR.T @ self.reward_weights  # for r_w rewards and state SR, value for each feature
        if self.TD_arg:
            self.state_action_values = self.TD

    def learn_TD(self, state_new, reward_new):
        TD_error = 0 - self.TD.T @ self.state + self.discount_factor * self.TD.T @ state_new  # one step TD-error
        alpha = self.alpha
        if self.kalman_TD:
            if self.multi:
                self.covariance_TD = np.copy(self.covariance_all_TD[self.current_task])
                self.kalman_gain_TD = np.copy(self.kalman_gain_all_TD[self.current_task])
                self.residual_covariance_TD = np.copy(self.residual_covariance_all_TD[self.current_task])
                self.kalman_filter_TD(self.state)
                alpha = abs(self.kalman_gain_TD)
                self.covariance_all_TD[self.current_task] = np.copy(self.covariance_TD)
                self.residual_covariance_all_TD[self.current_task] = np.copy(self.residual_covariance_TD)
                # if experienced map before update to drifted_covariance = self.covariance + np.identity(self.n_states) * 0.01
                all_tasks = set(self.selected_map).difference([self.current_task])
                for i in all_tasks:
                    self.covariance_all_TD[i] = np.copy(
                        self.covariance_all_TD[i] + np.identity(self.n_states) * 0)
            else:
                self.kalman_filter_TD(self.state)
                alpha = abs(self.kalman_gain_TD)

        TD_new = self.TD + alpha * (TD_error @ self.state.T).T
        self.TD = np.copy(TD_new)

        state_flat_new = np.where(state_new == 1)[0][0]
        if state_flat_new in [5, 8]:  # at the end of ep update last state val
            TD_error = reward_new - self.TD.T @ state_new  # one step TD-error, there is no value at the next state, bc there is no next state
            alpha = self.alpha
            if self.kalman_TD:
                if self.multi:
                    self.covariance_TD = np.copy(self.covariance_all_TD[self.current_task])
                    self.kalman_gain_TD = np.copy(self.kalman_gain_all_TD[self.current_task])
                    self.residual_covariance_TD = np.copy(self.residual_covariance_all_TD[self.current_task])
                    self.kalman_filter_TD(state_new)
                    alpha = abs(self.kalman_gain_TD)
                    self.covariance_all_TD[self.current_task] = np.copy(self.covariance_TD)
                    self.residual_covariance_all_TD[self.current_task] = np.copy(self.residual_covariance_TD)
                    # if experienced map before update to drifted_covariance = self.covariance + np.identity(self.n_states) * 0.01
                    all_tasks = set(self.selected_map).difference([self.current_task])
                    for i in all_tasks:
                        self.covariance_all_TD[i] = np.copy(
                            self.covariance_all_TD[i] + np.identity(self.n_states) * 0)
                else:
                    self.kalman_filter_TD(state_new)
                    alpha = abs(self.kalman_gain_TD)
            TD_new = self.TD + alpha * (TD_error @ state_new.T).T
            self.TD = np.copy(TD_new)

        return TD_new

    def learn_successor_S(self, state_new, reward_new, world):
        state_obs = self.state
        TD_error = state_obs + self.discount_factor * self.SR @ state_new - self.SR @ self.state
        alpha = self.alpha
        if self.kalman_SR:
            if self.multi:
                self.covariance = np.copy(self.covariance_all[self.current_task])
                self.kalman_gain = np.copy(self.kalman_gain_all[self.current_task])
                self.residual_covariance = np.copy(self.residual_covariance_all[self.current_task])
                self.kalman_filter_SR(state_new, world)
                alpha = abs(self.kalman_gain)
                self.covariance_all[self.current_task] = np.copy(self.covariance)
                self.residual_covariance_all[self.current_task] = np.copy(self.residual_covariance)
                # if experienced map before update to drifted_covariance = self.covariance + np.identity(self.n_states) * 0.01
                all_tasks = set(self.selected_map).difference([self.current_task])
                for i in all_tasks:
                    self.covariance_all[i] = np.copy(
                        self.covariance_all[i] + np.identity(self.n_states) * 0)
            else:
                self.kalman_filter_SR(state_new, world)
                alpha = abs(self.kalman_gain)
        inds = np.where(self.state != 0)[0]
        state_norm = np.copy(self.state)
        state_norm[inds] = 1
        normer = sum(state_norm)
        state_norm[inds] = state_norm[inds] / self.state[
            inds]  # think about the flexible learning rate in whether this needs to happen
        state_norm = state_norm / normer

        SR_new = self.SR + (alpha * (TD_error @ state_norm.T).T).T  # this applies the relevant state LR
        self.SR = np.copy(SR_new)

        return SR_new

    def learn_reward(self, reward_new, state_new):
        r_expected = self.reward_weights.T @ state_new
        r_received = reward_new
        reward_weights_new = self.reward_weights + self.alpha_RW * (state_new @ (
                r_received - r_expected))
        self.reward_weights = np.copy(reward_weights_new)
        return reward_weights_new

    def update_estimates(self, world, reward_new, state_new, feature_iw):
        if not self.TD_arg:
            self.learn_successor_S(state_new, reward_new, world)
            self.learn_reward(reward_new, state_new)
        if self.TD_arg:
            self.learn_TD(state_new, reward_new)
        self.state = state_new
        self.counter = self.counter + 1
        state_flat_new = np.where(state_new == 1)[0][0]
        if world.inf_task in ['struct_disc', 'DNMS']:
            if (state_flat_new == 5 and world.task_num in [1, 4, 6]) or (
                    state_flat_new == 8 and world.task_num in [2, 3, 5]):
                self.correct_visits.append(1)
            elif (state_flat_new == 5 and world.task_num in [2, 3, 5]) or (
                    state_flat_new == 8 and world.task_num in [1, 4, 6]):
                self.correct_visits.append(0)
        else:
            if (state_flat_new == 5 and world.task_num == 1) or (state_flat_new == 8 and world.task_num == 2):
                self.correct_visits.append(1)
            elif (state_flat_new == 5 and world.task_num == 2) or (state_flat_new == 8 and world.task_num == 1):
                self.correct_visits.append(0)

    def update_lists(self, action_new):
        self.states.append(self.state)
        self.reward_list.append(np.copy(self.reward_weights))
        if not self.passive:
            self.SA_list.append(np.copy(self.state_action_values))
            self.action = action_new
        self.SR_list.append(np.copy(self.SR))
        self.counter_list.append(self.counter)

    def analyze_me(self, world):
        time_per_episode = np.where(np.asarray(self.counter_list) == 1)[0] - 1
        self.time_per_episode = time_per_episode[1:] - time_per_episode[0:-1]


class BSRAgent_SS(SR_SF_Agent):
    # learns with SR of state-action pairs, multiple maps
    def __init__(self, world, starting_state, filter, SR_learning_rate, stickiness, particle_full, kalman_SR, kalman,
                 multi, conc, covariance_tracking, TD, kalman_TD, fake_outcome):
        super().__init__(world, starting_state, kalman_SR, kalman, TD, kalman_TD)
        self.n_maps = 10

        self.kalman_SR = kalman_SR

        # multi variables
        self.covariance_tracking = covariance_tracking
        self.limit = world.limit
        self.joint_inf = world.joint_inf
        self.do_joint = True

        self.SA_list = [[] for i in range(self.n_maps)]
        self.CR_map_list = [[] for i in range(self.n_maps)]  # list of list of lists #x[0].append([1,2,3])
        self.SR_list = [[] for i in range(self.n_maps)]
        self.reward_list = [[] for i in range(self.n_maps)]  # list of all reward weight vectors
        self.selected_map = []  # list of selected maps
        self.selection_probabilities = []  # probability of being in each map
        self.SR = np.diag(np.ones(
            self.n_states))  # important to keep environmental transitions at 1, does it make sense to keep featrures at 0?
        self.filter = filter  # 3 in paper
        if self.filter == 0:
            self.kernel = [1]
        if self.filter == 1:
            self.kernel = [0.9, 1, 0.9]
        if self.filter == 2:
            self.kernel = [0.9 ** 2, 0.9, 1, 0.9, 0.9 ** 2]
        if self.filter == 3:
            self.kernel = [0.9 ** 3, 0.9 ** 2, 0.9, 1, 0.9, 0.9 ** 2, 0.9 ** 3]
        if self.filter == 4:
            self.kernel = [0.9 ** 4, 0.9 ** 3, 0.9 ** 2, 0.9, 1, 0.9, 0.9 ** 2, 0.9 ** 3, 0.9 ** 4]
        self.n_particles = 100
        self.n_context_history = 10
        self.particles = np.zeros((self.n_particles, self.n_context_history))  # 100 particles,  10 last contexts
        self.prob_tasks = np.zeros(self.n_maps)
        self.prob_tasks = np.array([1, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0])
        self.prob_tasks = self.prob_tasks / np.sum(self.prob_tasks)
        self.importance_weights = self.prob_tasks
        for j in np.arange(100):
            self.particles[j, :] = np.random.choice(self.n_maps, self.n_context_history, replace=True,
                                                    p=self.prob_tasks)
        self.CR_maps_all = [np.copy(np.zeros(self.n_states)), np.copy(np.zeros(self.n_states)),
                            np.copy(np.zeros(self.n_states)), np.copy(np.zeros(self.n_states)),
                            np.copy(np.zeros(self.n_states)), np.copy(np.zeros(self.n_states)),
                            np.copy(np.zeros(self.n_states)), np.copy(np.zeros(self.n_states)),
                            np.copy(np.zeros(self.n_states)), np.copy(np.zeros(self.n_states))]
        self.reward_weights_all = [np.copy(self.reward_weights), np.copy(self.reward_weights),
                                   np.copy(self.reward_weights), np.copy(self.reward_weights),
                                   np.copy(self.reward_weights), np.copy(self.reward_weights),
                                   np.copy(self.reward_weights), np.copy(self.reward_weights),
                                   np.copy(self.reward_weights), np.copy(self.reward_weights)]
        self.SR_all = [np.copy(self.SR), np.copy(self.SR), np.copy(self.SR), np.copy(self.SR), np.copy(self.SR),
                       np.copy(self.SR), np.copy(self.SR), np.copy(self.SR),
                       np.copy(self.SR), np.copy(self.SR)]
        self.task_mem = []
        self.state_memory = []
        self.reward_memory = [0] * (self.filter + 1)
        self.normalizer = [0] * (self.filter + 1)
        self.current_task = int(np.argmax(self.importance_weights))
        self.dirichlet_conc = conc  # higher ->more likely to switch distributions #2 in paper #<=1.6 or =>2.1 give split via 3 first odours, 2 gives semi split into rewarded/unrewarded
        # never splits into more than 3 groups
        self.CR_learning_rate = 0.2  # 0.2 in paper
        self.counter = 0
        self.alpha = SR_learning_rate
        self.sticky = True
        self.stickiness = stickiness
        self.trial_end_task = []
        self.trial_end_prob = []
        self.trial_begin_task = []
        self.trial_begin_prob = []
        self.particle_full = particle_full
        if particle_full:
            self.particles = np.zeros((self.n_particles, 1))
            self.importance_weights = np.array([1 / self.dirichlet_conc])
            self.current_task = 0
            self.possible_contexts = [0]
        if self.kalman:
            self.covariance_all_outcome = []
            self.kalman_gain_all_outcome = []
            self.residual_covariance_all_outcome = []
            for j in np.arange(self.n_maps):
                self.covariance_all_outcome.append(np.copy(self.covariance_default * 0.6))
                self.kalman_gain_all_outcome.append(np.copy(self.kalman_gain_default))
                self.residual_covariance_all_outcome.append(np.copy(1.6))

        self.kalman_SR = kalman_SR
        if self.kalman_SR:
            self.covariance_all = []
            self.kalman_gain_all = []
            self.residual_covariance_all = []
            for j in np.arange(self.n_maps):
                self.covariance_all.append(np.copy(self.covariance_default * 0.6))
                self.kalman_gain_all.append(np.copy(self.kalman_gain_default))
                self.residual_covariance_all.append(np.copy(1.6))

        self.multi = multi
        self.confidence = self.importance_weights[self.current_task]
        self.out_conf_list = []
        self.retro = False
        self.raw_prob = 0
        self.stored_weights = np.copy(self.importance_weights)

        self.TD_arg = TD
        if self.TD_arg:
            self.selected_map = []  # list of selected maps
            self.n_maps = 10
            self.TD_list = [[] for i in range(self.n_maps)]
            self.TD = np.zeros((self.n_states, 1))
            self.TD_all = [np.copy(self.TD), np.copy(self.TD), np.copy(self.TD), np.copy(self.TD), np.copy(self.TD),
                           np.copy(self.TD), np.copy(self.TD), np.copy(self.TD), np.copy(self.TD), np.copy(self.TD),
                           np.copy(self.TD),
                           np.copy(self.TD)]
            self.alpha = SR_learning_rate
            self.current_task = 0

            self.kalman_TD = kalman_TD
            if self.kalman_TD:
                self.covariance_all_TD = []
                self.kalman_gain_all_TD = []
                self.residual_covariance_all_TD = []
                for j in np.arange(self.n_maps):
                    self.covariance_all_TD.append(np.copy(self.covariance_default * 0.6))  # 0.6
                    self.kalman_gain_all_TD.append(np.copy(self.kalman_gain_default))
                    self.residual_covariance_all_TD.append(np.copy(1.6))  # 1.6
            self.fake_outcome = fake_outcome

    def particle_filter(self):
        # 1) define max number of contexts: self.n_maps
        # 2) define number of particles and number of context memories: self.n_particles, self.n_context_history
        alpha = self.dirichlet_conc  # concentration dirichlet
        new_contexts = np.zeros(self.n_particles)
        for i in np.arange(self.n_particles):  # for each particle
            # prob of being in new context
            # 3) recognize which contexts are in particle history: if this is less than the max number, add a context
            current_particle = self.particles[i, :].tolist()
            context_history = set(current_particle)  # what contexts has this particle been in
            num_contexts = len(context_history)
            counts_per_context = [current_particle.count(k) for k in
                                  context_history]  # how many times has it been in each context
            # 4) calculate probabilities
            prob = np.zeros(self.n_maps)  # list of probabilities for each context for particles next position
            if num_contexts < self.n_maps:
                prob[(np.array(list(context_history))).astype(int)] = np.array(counts_per_context) / (
                        self.n_context_history + alpha)
                new_prob = alpha / (self.n_context_history + alpha)  # probability of being in a new cluster
                if self.sticky:
                    # check whether the previous context fits the current one
                    prev_context = np.zeros(num_contexts)
                    prev_context[np.array(list(context_history)) == current_particle[-1]] = 1
                    prob[(np.array(list(context_history))).astype(int)] = (np.array(
                        counts_per_context) + self.stickiness * prev_context) / (
                                                                                  self.n_context_history + alpha + self.stickiness)
                    new_prob = alpha / (
                            self.n_context_history + alpha + self.stickiness)  # probability of being in a new cluster
                which_context = min(set(range(self.n_maps)).difference(context_history))
                prob[which_context] = new_prob  # new context added
            else:
                prob[(np.array(list(context_history))).astype(int)] = (np.array(
                    counts_per_context) + alpha / num_contexts) / (self.n_context_history + alpha)
                if self.sticky:
                    # check whether the previous context fits the current one
                    prev_context = np.zeros(num_contexts)
                    prev_context[np.array(list(context_history)) == current_particle[-1]] = 1
                    prob[(np.array(list(context_history))).astype(int)] = (np.array(
                        counts_per_context) + alpha / num_contexts + self.stickiness * prev_context) / (
                                                                                  self.n_context_history + alpha + self.stickiness)
            # 5) pick 1 new context based off of probabilities
            new_contexts[i] = int(np.random.choice(self.n_maps, 1, p=prob))
        self.particles = np.append(self.particles[:, 1:], np.asarray([new_contexts]).T, 1)
        return new_contexts

    def full_particle_filter(self):
        # 1) define max number of contexts: self.n_maps
        # 2) define number of particles and number of context memories: self.n_particles, self.n_context_history
        alpha = self.dirichlet_conc  # concentration dirichlet
        new_contexts = np.zeros(self.n_particles)
        for i in np.arange(self.n_particles):  # for each particle
            # prob of being in new context
            # 3) recognize which contexts are in particle history: if this is less than the max number, add a context
            current_particle = self.particles[i, :].tolist()
            context_history = set(current_particle)  # what contexts has this particle been in
            num_contexts = len(context_history)
            counts_per_context = [current_particle.count(k) for k in
                                  context_history]  # how many times has it been in each context
            n_context_history = len(current_particle)
            length_contexts = int(max(context_history) + 1)
            # 4) calculate probabilities
            prob = np.zeros(length_contexts)  # list of probabilities for each context for particles next position
            # check whether the previous context fits the current one
            prev_context = np.zeros(num_contexts)
            prev_context[np.array(list(context_history)) == current_particle[-1]] = 1
            # calculate probs
            prob[(np.array(list(context_history))).astype(int)] = (np.array(
                counts_per_context) + self.stickiness * prev_context) / (
                                                                          n_context_history + alpha + self.stickiness)
            new_prob = alpha / (
                    n_context_history + alpha + self.stickiness)  # probability of being in a new cluster
            prob = np.append(prob, new_prob)  # new context added
            possible_contexts = np.arange(len(prob))
            # 5) pick 1 new context based off of probabilities
            new_contexts[i] = int(np.random.choice(possible_contexts, 1, p=prob))
            if n_context_history < 100:
                self.particles = np.append(self.particles, np.asarray([new_contexts]).T, 1)
            else:
                self.particles = np.append(self.particles[:, 1:], np.asarray([new_contexts]).T, 1)
        return new_contexts

    def infer_task(self, CR_pred, reward_obs, CR_state, world):
        CR_sd = 1.6  # 1.6 in paper
        if self.particle_full:
            new_contexts = self.full_particle_filter()
            list_of_new_contexts = new_contexts.tolist()
            possible_contexts_outcome = set(new_contexts)
            length_contexts = int(max(possible_contexts_outcome) + 1)
            list_possible_conts = np.arange(length_contexts)
            particles_per_context = [list_of_new_contexts.count(k) for k in
                                     list_possible_conts]  # how many particles in each context
            # 1) calculate the probability for each context based off of observation vs expectation
            context_probs = np.zeros(length_contexts)
            CR_pred = np.zeros(length_contexts)
            for i in list_possible_conts:
                if i >= len(self.CR_maps_all):
                    self.CR_maps_all.append(np.copy(np.zeros(self.n_states)))
                    self.covariance_all_outcome.append(np.copy(self.covariance_default))
                i = int(i)
                CR_pred[i] = self.CR_maps_all[i][CR_state]
            for j in list_possible_conts:
                j = int(j)
                if self.kalman:
                    CR_sd = self.covariance_all_outcome[j][CR_state, CR_state]
                mean = CR_pred[j]
                context_probs[j] = norm.pdf(reward_obs, loc=mean, scale=CR_sd)
        else:
            new_contexts = self.particle_filter()
            context_probs = np.zeros(self.n_maps)
            list_of_new_contexts = new_contexts.tolist()
            particles_per_context = [list_of_new_contexts.count(k) for k in
                                     np.arange(self.n_maps)]  # how many particles in each context
            # 1) calculate the probability for each context based off of observation vs expectation
            for j in np.arange(self.n_maps):
                if self.kalman:
                    if self.covariance_tracking:
                        CR_sd = self.residual_covariance_all_outcome[j]
                mean = CR_pred[j]
                context_probs[j] = norm.pdf(reward_obs, loc=mean, scale=CR_sd)
        # 2) normalize each by the sum of all (weights_p)
        norm_context_probs = context_probs / sum(context_probs)
        # 3) multiply probs by number of particles in context (total_particle_weights)
        total_particle_weights = norm_context_probs * particles_per_context
        # 4) normalize by sum gives: (weights_t) THIS IS WHAT IS USED DOWNSTREAM AS IMPORTANCE WEIGHTS
        self.raw_prob = context_probs  # measure of how good the map is, not in relationship to anything
        self.importance_weights = total_particle_weights / sum(total_particle_weights)
        inferred_task = int(np.argmax(self.importance_weights))
        # 5) resample:
        # 5.1) particle weights: normalized probability for each new particles context (weights_p), normalized by sum of all particle weights (total particle weights)
        new = [int(x) for x in list(new_contexts)]
        particle_weights = norm_context_probs[new[:]] / sum(total_particle_weights)
        # 5.2) resample particles and histories based on their particle weights, ie how relevant they are given current observations
        inds = np.random.choice(100, 100, replace=True, p=particle_weights)
        self.particles = np.copy(self.particles[inds[:]])
        max_here = max(self.importance_weights)
        # self.out_conf_list.append(max_here)
        return inferred_task

    def conv_reward_map_learn(self, CR_obs, CR_state, reward_observed, world):
        CR_pred = np.zeros(self.n_maps)
        for i in np.arange(self.n_maps):
            CR_pred[i] = self.CR_maps_all[i][CR_state]
        self.current_task = self.infer_task(CR_pred, CR_obs, CR_state, world)

    def conv_reward_map_update(self, CR_obs, CR_state):
        CR_pred = np.zeros(self.n_maps)
        for i in np.arange(self.n_maps):
            CR_pred[i] = self.CR_maps_all[i][CR_state]
        alpha = self.CR_learning_rate
        pred_error = CR_obs - CR_pred[int(self.current_task)]
        if self.kalman:
            self.covariance_outcome = np.copy(self.covariance_all_outcome[self.current_task])
            self.kalman_gain_outcome = np.copy(self.kalman_gain_all_outcome[self.current_task])
            self.residual_covariance_outcome = np.copy(self.residual_covariance_all_outcome[self.current_task])
            self.kalman_filter_CR(CR_state, CR_obs)
            alpha = self.kalman_gain_outcome[
                        CR_state] * 2 / 3  # this is vital if not used to make choice, just alpha doesn't work,
            self.covariance_all_outcome[self.current_task] = np.copy(self.covariance_outcome)
            self.residual_covariance_all_outcome[self.current_task] = np.copy(self.residual_covariance_outcome)
            # if experienced map before update to drifted_covariance = self.covariance + np.identity(self.n_states) * 0.01
            all_tasks = set(self.selected_map).difference([self.current_task])
            for i in all_tasks:
                self.covariance_all_outcome[i] = np.copy(
                    self.covariance_all_outcome[i] + np.identity(self.n_states) * 0)
            # adding this in makes no sense, basically doubling up on impact of particle filter, basically covariance is meant to represent: confidence of observations given I am in the context I say I am in, losing experience in a context changes your memory of that context via particle filter, but not of things within that context
        new_CR = self.CR_maps_all[int(self.current_task)][CR_state] + alpha * pred_error
        self.CR_maps_all[self.current_task][CR_state] = new_CR.copy()

    def update_SR(self, state_new, reward_new, world):
        # pick relevant components
        if not self.multi:
            if not self.TD_arg:
                self.SR = np.copy(self.SR_all[self.current_task])
                self.reward_weights = np.copy(self.reward_weights_all[self.current_task])
                new_SR = self.learn_successor_S(state_new, reward_new, world)
                new_reward = self.learn_reward(reward_new, state_new)
                self.SR_all[self.current_task] = np.copy(new_SR)
                self.reward_weights_all[self.current_task] = np.copy(new_reward)
            if self.TD_arg:
                self.TD = np.copy(self.TD_all[self.current_task])
                new_TD = self.learn_TD(state_new, reward_new)
                self.TD_all[self.current_task] = np.copy(new_TD)
        # update currents
        self.state = state_new
        self.counter = self.counter + 1

    def joint_task(self, feature_iw, outcome_iw, real_task_last_ep):
        if self.joint_inf:
            if self.task_tot <= self.limit:
                if self.fake_outcome:
                    outcome_iw = np.zeros(10)
                    if real_task_last_ep == 1:
                        outcome_iw[0] = 0.95
                        outcome_iw[1] = 0.05
                    elif real_task_last_ep == 2:
                        outcome_iw[0] = 0.05
                        outcome_iw[1] = 0.95
                # if self.do_joint:
                self.current_task = np.argmax((outcome_iw + feature_iw) / 2)
                self.confidence = np.max((outcome_iw + feature_iw) / 2)
        pie = 'hi'
        # if np.max(feature_iw) > 0.9:
        #     self.current_task = np.argmax(feature_iw)
        #     self.confidence = np.max(feature_iw)

    def post_update(self, CR_obs, CR_state):
        self.conv_reward_map_update(CR_obs, CR_state)

    def update_estimates(self, world, reward_new, state_new, feature_iw):
        # filter: takes 7 values: 3 preceeding current state, 1 in current state, 3 after current stat
        if self.counter == 0:  # start of a new trial
            self.state_memory = []
            self.normalizer = [0] * (self.filter + 1)
            self.reward_memory = [0] * (self.filter + 1)  # 4 preceeding current state+delay of 3, 7 values
        # infer task
        state_flat = np.where(self.state == 1)[0][0]  # state not feature
        state_flat_new = np.where(state_new == 1)[0][0]  # state not feature
        self.state_memory.append(state_flat)
        self.reward_memory.append(0)
        self.normalizer.append(1)

        if self.multi:
            self.joint_task(feature_iw, self.stored_weights, world.current_act_task)

        if len(
                self.state_memory) > self.filter:  # once 3 states have been experienced state_memory=[s1,s2,s3, s4], r=[0,0,0,0,r1,r2,r3]
            # CR_obs = (np.asarray(self.kernel) @ np.asarray(self.reward_memory[-(self.filter * 2 + 1):])) / sum(self.kernel) #grab the last 7 reward values to calculate value of s1
            CR_obs = (np.asarray(self.kernel) @ np.asarray(self.reward_memory[-(self.filter * 2 + 1):])) / (
                    np.asarray(self.kernel) @ self.normalizer[-(self.filter * 2 + 1):])
            CR_state = self.state_memory[-(self.filter + 1)]  # selecting the state that the CR is centered on
            reward_observed = self.reward_memory[-(self.filter + 1)]
            self.conv_reward_map_learn(CR_obs, CR_state, reward_observed, world)  # this line is vital
            if self.multi:
                self.joint_task(feature_iw, self.importance_weights, world.current_act_task)
            self.post_update(CR_obs, CR_state)

        if state_flat_new in world.goal_state:  # if at the end of a trial
            self.state_memory.append(state_flat_new)
            self.reward_memory.append(reward_new)
            CR_obs = (np.asarray(self.kernel) @ np.asarray(self.reward_memory[-(self.filter * 2 + 1):])) / (
                    np.asarray(self.kernel) @ self.normalizer[-(self.filter * 2 + 1):])
            CR_state = self.state_memory[-(self.filter + 1)]  # selecting the state that the CR is centered on
            reward_observed = self.reward_memory[-(self.filter + 1)]
            self.conv_reward_map_learn(CR_obs, CR_state, reward_observed, world)
            if self.multi:
                self.joint_task(feature_iw, self.importance_weights, world.current_act_task)
            self.post_update(CR_obs, CR_state)

            for i in np.arange(
                    self.filter):  # i is 0,1,2  r=[0,0,0,  0  ,0,0,5] [state-3,   state-2, state-1 ,last state]
                # r = [0, 0, 0, 0, 0, 5, 0][state - 2, state - 1, last state, 0] first run
                # r = [0, 0, 0, 0, 5, 0, 0][state - 1, last state, 0, 0] second run
                # r = [0, 0, 0, 5, 0, 0, 0][last state, 0, 0, 0] third run
                # run at end with padded zeros
                self.state_memory.append(0)
                self.reward_memory.append(0)
                self.normalizer.append(0)
                CR_obs = (np.asarray(self.kernel) @ np.asarray(self.reward_memory[-(self.filter * 2 + 1):])) / (
                        np.asarray(self.kernel) @ self.normalizer[-(self.filter * 2 + 1):])
                CR_state = self.state_memory[-(self.filter + 1)]
                reward_observed = self.reward_memory[-(self.filter + 1)]
                self.conv_reward_map_learn(CR_obs, CR_state, reward_observed, world)
                if self.multi:
                    self.joint_task(feature_iw, self.importance_weights, world.current_act_task)
                self.post_update(CR_obs, CR_state)

        if state_flat_new in world.goal_state:
            end_task = np.copy(np.argmax(self.importance_weights))
            # self.trial_end_task.append(np.copy(np.argmax(self.importance_weights)))
            # self.trial_end_prob.append(self.importance_weights[end_task])
        if state_flat_new == 1:
            start_task = np.copy(np.argmax(self.importance_weights))
            # self.trial_begin_task.append(np.copy(np.argmax(self.importance_weights)))
            # self.trial_begin_prob.append(self.importance_weights[start_task])
        if world.inf_task in ['struct_disc', 'DNMS']:
            if (state_flat_new == 5 and world.task_num in [1, 4, 6]) or (
                    state_flat_new == 8 and world.task_num in [2, 3, 5]):
                self.correct_visits.append(1)
            elif (state_flat_new == 5 and world.task_num in [2, 3, 5]) or (
                    state_flat_new == 8 and world.task_num in [1, 4, 6]):
                self.correct_visits.append(0)

        else:
            if (state_flat_new == 5 and world.task_num == 1) or (state_flat_new == 8 and world.task_num == 2) or (
                    state_flat_new == 8 and world.task_num == 3) or (state_flat_new == 5 and world.task_num == 4):
                self.correct_visits.append(1)
            elif (state_flat_new == 5 and world.task_num == 2) or (state_flat_new == 8 and world.task_num == 1) or (
                    state_flat_new == 8 and world.task_num == 4) or (state_flat_new == 5 and world.task_num == 3):
                self.correct_visits.append(0)

        self.stored_weights = np.copy(self.importance_weights)

        if not self.multi:
            self.update_SR(state_new, reward_new, world)

    def update_lists(self, action_new):
        # self.selection_probabilities.append(np.copy(self.importance_weights))
        # self.selected_map.append(self.current_task)  # the one that was probabilistically selected
        # self.states.append(self.state)
        self.counter_list.append(self.counter)
        # self.task_mem.append(np.copy(np.argmax(self.importance_weights)))  # the one who's CR map is updated
        self.action = action_new
        # for i in np.arange(self.n_maps):
        # self.reward_list[i].append(self.reward_weights_all[i].copy())
        # self.SR_list[i].append(self.SR_all[i].copy())
        # self.CR_map_list[i].append(self.CR_maps_all[i].copy())


class feature_BSRAgent_SS(BSRAgent_SS):
    # learns with SR of state-action pairs, multiple maps
    def __init__(self, world, starting_state, inference, inf_concentration, SR_learning_rate, stickiness,
                 particle_full, kalman_SR, kalman, covariance_tracking, kalman_replay, TD, kalman_TD):
        multi = True
        fake_outcome = False
        super().__init__(world, starting_state, 1, SR_learning_rate, stickiness, particle_full, kalman_SR, kalman,
                         multi, inf_concentration, covariance_tracking, TD, kalman_TD, fake_outcome)
        self.n_maps = 10
        self.covariance_tracking = covariance_tracking
        self.kalman_replay = kalman_replay

        self.SR_list = [[] for i in range(self.n_maps)]
        self.reward_list = [[] for i in range(self.n_maps)]  # list of all reward weight vectors
        self.selected_map = []  # list of selected maps
        self.selection_probabilities = []  # probability of being in each map
        self.n_particles = 100
        self.n_context_history = 10
        self.particles = np.zeros((self.n_particles, self.n_context_history))  # 100 particles,  10 last contexts
        self.prob_tasks = np.array([1, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0])
        self.prob_tasks = self.prob_tasks / np.sum(self.prob_tasks)
        self.importance_weights = self.prob_tasks
        self.SR = np.zeros(self.n_states)
        self.SR[:] = 1
        self.SR = np.diag(self.SR)
        self.multi = True
        for j in np.arange(100):
            self.particles[j, :] = np.random.choice(self.n_maps, self.n_context_history, replace=True,
                                                    p=self.prob_tasks)
        self.reward_weights_all = [np.copy(self.reward_weights), np.copy(self.reward_weights),
                                   np.copy(self.reward_weights), np.copy(self.reward_weights),
                                   np.copy(self.reward_weights), np.copy(self.reward_weights),
                                   np.copy(self.reward_weights), np.copy(self.reward_weights),
                                   np.copy(self.reward_weights), np.copy(self.reward_weights),
                                   np.copy(self.reward_weights), np.copy(self.reward_weights)]
        self.SR_all = [np.copy(self.SR), np.copy(self.SR), np.copy(self.SR), np.copy(self.SR), np.copy(self.SR),
                       np.copy(self.SR), np.copy(self.SR), np.copy(self.SR), np.copy(self.SR), np.copy(self.SR),
                       np.copy(self.SR),
                       np.copy(self.SR)]
        self.task_mem = []
        self.current_task = int(np.random.choice(self.n_maps, 1, replace=True, p=self.importance_weights))
        self.dirichlet_conc = inf_concentration  # higher ->more likely to switch distributions #2 in paper #<=1.6 or =>2.1 give split via 3 first odours, 2 gives semi split into rewarded/unrewarded
        # never splits into more than 3 groups
        self.counter = 0
        self.inference = inference
        self.feature = world.feature
        self.alpha = SR_learning_rate  # 0.4
        self.states_for_ret = []
        self.feat_conf_list = []
        self.raw_prob = 0
        if self.kalman_SR:
            self.covariance_all = []
            self.kalman_gain_all = []
            self.residual_covariance_all = []
            for j in np.arange(self.n_maps):
                self.covariance_all.append(np.copy(self.covariance_default * 0.6))  # 0.6
                self.kalman_gain_all.append(np.copy(self.kalman_gain_default))
                if world.inf_task in ['struct_disc', 'DNMS']:
                    self.residual_covariance_all.append(np.copy(0.9))  # 1.6
                else:
                    self.residual_covariance_all.append(np.copy(1.6))  # 1.6
        self.gain_hist = []
        self.ep_end_SR = self.SR_all
        self.ep_end_rw = self.reward_weights_all
        self.ep_end_particles = self.particles
        self.ep_end_confidence = self.confidence
        self.task_assoc_weights = np.zeros((self.n_maps, self.n_maps))
        if self.kalman_SR:
            self.ep_end_cov = self.covariance_all
        self.task_rec = []
        self.ave_resid = []
        self.running_resid = []

    def infer_task(self, CR_pred, reward_obs, state_new, world):
        if world.inf_task in ['struct_disc', 'DNMS']:
            CR_sd = 0.9  # 1.6 in paper
        else:
            CR_sd = 1.6  # 1.6 in paper
        CR_sd_mult = np.diag([CR_sd] * self.n_states)
        if self.particle_full:
            new_contexts = self.full_particle_filter()
            list_of_new_contexts = new_contexts.tolist()
            possible_contexts_outcome = set(new_contexts)
            length_contexts = int(max(possible_contexts_outcome) + 1)
            list_possible_conts = np.arange(length_contexts)
            particles_per_context = [list_of_new_contexts.count(k) for k in
                                     list_possible_conts]  # how many particles in each context
            # 1) calculate the probability for each context based off of observation vs expectation
            context_probs = np.zeros(length_contexts)
            CR_pred = np.zeros((self.n_states, length_contexts))
            for i in list_possible_conts:
                if i >= len(self.SR_all):
                    new_SR = np.diag(np.ones(self.n_states))  # fix for default SR
                    self.SR_all.append(np.copy(new_SR))
                current_SR = self.SR_all[i]
                CR_pred[:, i] = (current_SR.T @ self.state - self.discount_factor * (current_SR.T @ state_new))[:,
                                0]  # predicted observation, taken from SR TD error
            for j in list_possible_conts:
                mean = CR_pred[:, j]
                if self.inference == 'all_features':
                    context_probs[j] = multivariate_normal.pdf(reward_obs.T, mean,
                                                               CR_sd_mult)
                    if self.feature:
                        context_probs[j] = multivariate_normal.pdf(reward_obs.T[0], mean,
                                                                   CR_sd_mult)  # this is the option that compares all features
                elif self.inference == 'r_u_features':
                    CR_sd_mult = np.diag([CR_sd] * 2)
                    context_probs[j] = multivariate_normal.pdf(reward_obs[-2:], mean[-2:],
                                                               CR_sd_mult)  # this is the option that compares all features
                elif self.inference == 'reward_feature':
                    context_probs[j] = norm.pdf(reward_obs[-3], mean[-3],
                                                CR_sd)  # this is the option that compares only the predictiveness of reward, which no longer makes any sense
        else:
            new_contexts = self.particle_filter()
            context_probs = np.zeros(self.n_maps)
            list_of_new_contexts = new_contexts.tolist()
            particles_per_context = [list_of_new_contexts.count(k) for k in
                                     np.arange(self.n_maps)]  # how many particles in each context
            # 1) calculate the probability for each context based off of observation vs expectation
            for j in np.arange(self.n_maps):
                if self.kalman_SR:
                    if self.covariance_tracking:
                        if world.inf_task in ['struct_disc', 'DNMS']:
                            self.mini_SR_filter(state_new, j)
                        CR_sd_mult = np.diag([self.residual_covariance_all[j].tolist()] * self.n_states)
                mean = CR_pred[:, j]
                if self.inference == 'all_features':
                    context_probs[j] = multivariate_normal.pdf(reward_obs.T, mean,
                                                               CR_sd_mult)
                    if self.feature:
                        context_probs[j] = multivariate_normal.pdf(reward_obs.T[0], mean,
                                                                   CR_sd_mult)  # this is the option that compares all features
                elif self.inference == 'r_u_features':
                    CR_sd_mult = np.diag([CR_sd] * 2)
                    context_probs[j] = multivariate_normal.pdf(reward_obs[-2:], mean[-2:],
                                                               CR_sd_mult)  # this is the option that compares all features
                elif self.inference == 'reward_feature':
                    context_probs[j] = norm.pdf(reward_obs[-2], mean[-2],
                                                CR_sd)  # this is the option that compares only the predictiveness of reward, which no longer makes any sense
        # 2) normalize each by the sum of all (weights_p)
        norm_context_probs = context_probs / sum(context_probs)
        # 3) multiply probs by number of particles in context (total_particle_weights)
        total_particle_weights = norm_context_probs * particles_per_context
        # 4) normalize by sum gives: (weights_t) THIS IS WHAT IS USED DOWNSTREAM AS IMPORTANCE WEIGHTS
        self.raw_prob = context_probs
        self.importance_weights = total_particle_weights / sum(total_particle_weights)
        if len(self.importance_weights[np.isnan(self.importance_weights)]) > 0:
            self.importance_weights[:] = 1 / 10
        inferred_task = int(np.argmax(self.importance_weights))
        # 5) resample:
        # 5.1) particle weights: normalized probability for each new particles context (weights_p), normalized by sum of all particle weights (total particle weights)
        new = [int(x) for x in list(new_contexts)]
        particle_weights = norm_context_probs[new[:]] / sum(total_particle_weights)
        if len(particle_weights[np.isnan(particle_weights)]) > 0:
            particle_weights[:] = 1 / 100
        # 5.2) resample particles and histories based on their particle weights, ie how relevant they are given current observations
        inds = np.random.choice(100, 100, replace=True, p=particle_weights)
        self.particles = np.copy(self.particles[inds[:]])  #
        max_here = max(self.importance_weights)
        # self.feat_conf_list.append(max_here)
        return inferred_task

    def update_SR_ret(self, task, state_new, reward_new, world):
        # pick relevant components
        self.SR = np.copy(self.SR_all[task])
        self.reward_weights = np.copy(self.reward_weights_all[task])

        # update model
        new_SR = self.learn_successor_S(state_new, reward_new, world)  # weighted or unweighted
        new_reward = self.learn_reward(reward_new, state_new)

        self.SR = np.copy(new_SR)
        self.reward_weights = np.copy(new_reward)

        self.SR_all[task] = np.copy(self.SR)
        self.reward_weights_all[task] = np.copy(self.reward_weights)

    def retrograde_update(self, state_mem, feat_conf_list, out_conf_list, world):
        rewards = np.zeros(len(state_mem))
        rewards[-1] = 10

        if self.kalman_replay:
            self.kalman_SR = True  # leave the update of the priors and the confidence being used is kalman gain
        else:
            self.kalman_SR = False
        SR_lr = self.alpha
        self.alpha = 0.1
        for j in np.arange(1):
            for i in np.arange(len(state_mem) - 1):
                self.state = state_mem[i]
                state_new = state_mem[i + 1]
                reward_new = rewards[i + 1]
                self.update_SR_ret(self.current_task, state_new, reward_new, world)

        self.alpha = SR_lr
        self.states_for_ret = []
        self.kalman_SR = True

    def update_estimates(self, world, reward_new, state_new):
        SR_pred = np.zeros((self.n_states, self.n_maps))
        self.states_for_ret.append(self.state)
        for i in np.arange(self.n_maps):
            current_SR = self.SR_all[i]
            SR_pred[:, i] = (current_SR @ self.state - self.discount_factor * (current_SR @ state_new))[:,
                            0]  # predicted observation, taken from SR TD error #the [:,0] is to convert column to row vector
            # taking out the current_SR.T makes this into prediction error, currently its an occupancy of current state thing
        SR_obs = self.state
        # inference will compare: observed features - predicted observed features
        inferred_task = self.infer_task(SR_pred,
                                        SR_obs, state_new, world)
        # self.current_task = int(np.random.choice(self.n_maps, 1, replace=True, p=self.importance_weights))
        self.current_task = int(np.argmax(self.importance_weights))
        state_flat_new = np.where(state_new == 1)[0][0]

        if world.inf_task in ['struct_disc', 'DNMS']:
            if (state_flat_new == 5 and world.task_num in [1, 4, 6]) or (
                    state_flat_new == 8 and world.task_num in [2, 3, 5]):
                self.correct_visits.append(1)
            elif (state_flat_new == 5 and world.task_num in [2, 3, 5]) or (
                    state_flat_new == 8 and world.task_num in [1, 4, 6]):
                self.correct_visits.append(0)
        else:
            if (state_flat_new == 5 and world.task_num == 1) or (state_flat_new == 8 and world.task_num == 2) or (
                    state_flat_new == 8 and world.task_num == 3) or (state_flat_new == 5 and world.task_num == 4):
                self.correct_visits.append(1)
                self.states_for_ret.append(state_new)
            elif (state_flat_new == 5 and world.task_num == 2) or (state_flat_new == 8 and world.task_num == 1) or (
                    state_flat_new == 8 and world.task_num == 4) or (state_flat_new == 5 and world.task_num == 3):
                self.correct_visits.append(0)
        if state_flat_new in world.goal_state:
            end_task = np.copy(np.argmax(self.importance_weights))
            # self.trial_end_task.append(np.copy(np.argmax(self.importance_weights)))
            # self.trial_end_prob.append(self.importance_weights[end_task])
        if state_flat_new == 1:
            start_task = np.copy(np.argmax(self.importance_weights))
            # self.trial_begin_task.append(np.copy(np.argmax(self.importance_weights)))
            # self.trial_begin_prob.append(self.importance_weights[start_task])

    def update_SR(self, state_new, reward_new, world):
        self.SR = np.copy(self.SR_all[self.current_task])
        new_SR = self.learn_successor_S(state_new, reward_new, world)
        self.SR_all[self.current_task] = np.copy(new_SR)
        if not self.TD_arg:
            self.reward_weights = np.copy(self.reward_weights_all[self.current_task])
            new_reward = self.learn_reward(reward_new, state_new)
            self.reward_weights_all[self.current_task] = np.copy(new_reward)
        if self.TD_arg:
            self.TD = np.copy(self.TD_all[self.current_task])
            new_TD = self.learn_TD(state_new, reward_new)
            self.TD_all[self.current_task] = np.copy(new_TD)
        # update currents
        self.state = state_new
        self.counter = self.counter + 1
        # self.task_rec.append(self.importance_weights)

    def update_lists(self, action_new):
        # self.selection_probabilities.append(np.copy(self.importance_weights))
        # self.selected_map.append(self.current_task)  # the one that was probabilistically selected
        # self.states.append(self.state)
        self.counter_list.append(self.counter)
        # self.task_mem.append(np.copy(np.argmax(self.importance_weights)))  # the one who's CR map is updated
        self.action = action_new
        # for i in np.arange(self.n_maps):
        #     self.reward_list[i].append(self.reward_weights_all[i].copy())
        #     self.SR_list[i].append(self.SR_all[i].copy())


class Environment:
    """sets up rewards and generates next states dependent on actions/tasks
    :param self.passive passive task where no actions are chosen, different states are presented in pre-defined orders
    """
    n_actions = 4

    def __init__(self, trials):
        self.trials = trials
        self.passive = False
        # self.do_joint = True


class EnvironmentT(Environment):
    """sets up a t-maze where the animal makes choices
    :param feature: takes True/False, adds 2 more state-spaces (ie features) to allow for a local or global contextual-feature to be introduced
    :param self.inf_task: inference tasks, takes 'reward_feature','local' or 'global'
        'reward_feature' adds one state-space, representing reward-feature
        'global_cue' adds 2 state-spaces, active in either 1 or the other task
        'local' adds 2 state-spaces, active in the first 3 states of either task
    :param self.state_space: takes True/False, adds 2 more state-spaces (ie features) to allow for rewarded/unrewarded locations to be different states
    """
    # t-maze
    rows = 7
    columns = 3

    def __init__(self, tasks, trials, state_space, inf_task, feature, cue_val, one_shot, reward_feature, limit,
                 joint_prior, joint_inf, replay, long, arm_length, noise, fc):
        super().__init__(trials)
        self.limit = limit
        self.joint_prior = joint_prior
        self.joint_inf = joint_inf
        self.replay = replay
        self.long = long
        self.noise = noise

        self.n_states = 9
        self.tasks = tasks
        self.state_space = state_space
        if state_space:
            self.n_states = self.n_states + 2
        self.inf_task = inf_task  # either local (states 0-1) or global (all states) or reward feature, ##fix starting state
        if feature:
            self.n_states = self.n_states + 2
        self.reward_feature = reward_feature
        if reward_feature:
            self.n_states = self.n_states + 1

        self.long = long
        self.arm_length = arm_length
        if self.long:
            self.n_states = int(np.copy(self.n_states + self.arm_length - 2))
        if self.noise:
            self.n_states = int(np.copy(self.n_states + self.arm_length - 2))
        if self.inf_task in ['struct_disc']:
            self.n_states = self.n_states + 1

        self.feature = feature
        self.goal_state = []
        self.environ = self.create_environment()
        self.cue_val = cue_val
        self.one_shot = one_shot
        self.task_tot = 0
        self.current_act_task = 1
        self.forced_choice = fc

    def starting_state(self):
        flat_state = 0
        state_new = np.zeros((self.n_states, 1))  # features
        state_new[flat_state, :] = 1
        if self.inf_task == 'global_cue':
            if self.task_num == 1:
                state_new[-2, :] = self.cue_val
                state_new[-1, :] = 0
            elif self.task_num == 2:
                state_new[-1, :] = self.cue_val
                state_new[-2, :] = 0
        if self.inf_task == 'local':
            if (self.task_num == 1) & (flat_state in [1]):
                state_new[-2, :] = self.cue_val
                state_new[-1, :] = 0
            elif (self.task_num == 2) & (flat_state in [1]):
                state_new[-1, :] = self.cue_val
                state_new[-2, :] = 0
        if self.inf_task == 'close_cue':
            if (self.task_num == 1) & (flat_state in [4]):
                state_new[-2, :] = self.cue_val
                state_new[-1, :] = 0
            elif (self.task_num == 2) & (flat_state in [7]):
                state_new[-1, :] = self.cue_val
                state_new[-2, :] = 0

        if self.inf_task == 'struct_disc':
            # A->B->reward
            # A->C->no reward
            # B->A->no reward
            # B->C->reward
            # C->B->no reward
            # C->A-> reward
            if self.task_num == 1:
                if flat_state in [1]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0
                if flat_state in [2]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
            # A->C->no reward
            elif self.task_num == 2:
                if flat_state in [1]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0
                if flat_state in [2]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # B->A->no reward
            elif self.task_num == 3:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [2]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0
            # B->C->reward
            elif self.task_num == 4:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [2]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # C->B->no reward
            elif self.task_num == 5:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [2]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
            # C->A-> reward
            elif self.task_num == 6:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [2]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0

        if self.inf_task == 'DNMS':
            # A->B->reward
            # A->A->no reward
            # B->B->no reward
            # B->A->reward

            # A->B->reward
            if self.task_num == 1:
                if flat_state in [1]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [2]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # A->A->no reward
            elif self.task_num == 2:
                if flat_state in [1]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [2]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
            # B->B->no reward
            elif self.task_num == 3:
                if flat_state in [1]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [2]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # B->A->reward
            elif self.task_num == 4:
                if flat_state in [1]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [2]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0

        return state_new

    def one_step_lookahead(self, state, action):
        if self.inf_task in ['struct_disc', 'DNMS']:
            if state not in [2, 3, 4, 5, 6, 7, 8]:
                action = 0
        flat_state = int(self.environ[state, action])
        state_new = np.zeros((self.n_states, 1))  # features
        state_new[flat_state, :] = 1
        if self.inf_task == 'global_cue':
            if self.task_num == 1:
                state_new[-2, :] = self.cue_val
                state_new[-1, :] = 0
            elif self.task_num == 2:
                state_new[-1, :] = self.cue_val
                state_new[-2, :] = 0
        if self.inf_task == 'local':
            if (self.task_num == 1) & (flat_state in [1]):
                state_new[-2, :] = self.cue_val
            elif (self.task_num == 2) & (flat_state in [1]):
                state_new[-1, :] = self.cue_val
            #  if (flat_state in [1]):
            #      state_new[-2, :] = self.cue_val
            #      state_new[-1, :] = self.cue_val
        if self.noise:
            if flat_state in [1]:
                trans_states = np.arange(self.n_states - 2)  # minus the cue features
                added_features = trans_states[trans_states > 8]  # these are the extra features
                state_new[added_features, :] = 1

        if self.inf_task == 'close_cue':
            if (self.task_num == 1) & (flat_state in [4]):
                state_new[-2, :] = self.cue_val
                state_new[-1, :] = 0
            elif (self.task_num == 2) & (flat_state in [7]):
                state_new[-1, :] = self.cue_val
                state_new[-2, :] = 0

        if self.inf_task == 'dual_cue':
            if self.task_num == 1:
                # task 1 has -2 cue first followed by -1 cue
                if flat_state in [1]:
                    state_new[-2, :] = self.cue_val
                elif flat_state in [2]:
                    state_new[-1, :] = self.cue_val
            elif self.task_num == 2:
                # task 2 has -1 cue first followed by -2 cue
                if flat_state in [1]:
                    state_new[-1, :] = self.cue_val
                elif flat_state in [2]:
                    state_new[-2, :] = self.cue_val

        if self.inf_task == 'struct_disc':
            # A->B->reward 1,4,6
            state_before_end = self.n_states - 4
            if self.task_num == 1:
                if flat_state in [1]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
            # A->C->no reward
            elif self.task_num == 2:
                if flat_state in [1]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # B->A->no reward
            elif self.task_num == 3:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0
            # B->C->reward
            elif self.task_num == 4:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # C->B->no reward
            elif self.task_num == 5:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [state_before_end]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
            # C->A-> reward
            elif self.task_num == 6:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [state_before_end]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0

                # A->B->r (1)
                # B->A->no reward (3)
                # C->A-> reward (6)
                # A->C->no reward (2)
                # B->C->reward(4)
                # C->B->no reward (5)

                # 1,3,2,6,4,5

        if self.inf_task == 'DNMS':
            state_before_end = self.n_states - 3  # argh
            # A->B->reward
            # A->A->no reward
            # B->B->no reward
            # B->A->reward

            # A->B->reward
            if self.task_num == 1:
                if flat_state in [1]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # A->A->no reward
            elif self.task_num == 2:
                if flat_state in [1]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
            # B->B->no reward
            elif self.task_num == 3:
                if flat_state in [1]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [state_before_end]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # B->A->reward
            elif self.task_num == 4:
                if flat_state in [1]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [state_before_end]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0

        return state_new

    def create_environment(self):
        # 0 is forward
        # 1 back
        # 2 left
        # 3 right
        #    -5 if rewarded 10 if not rewarded
        #    -4
        #    -3
        # 0-1-2
        #    -6
        #    -7
        #    -8 if rewarded 9 if not rewarded
        environ = np.zeros((self.n_states, self.n_actions))
        environ[0, 0] = 1
        environ[0, 1:4] = 0
        environ[1, 0] = 2
        environ[1, 2:4] = 1
        environ[1, 1] = 0
        environ[2, 0] = 2
        environ[2, 2] = 3
        environ[2, 3] = 6
        environ[2, 1] = 1
        environ[3, 0:2] = 3
        environ[3, 2] = 4
        environ[3, 3] = 2
        environ[4, 0:2] = 4
        environ[4, 2] = 5
        environ[4, 3] = 3
        environ[5, 0] = 5
        environ[5, 1] = 0
        environ[5, 2] = 5
        environ[5, 3] = 4
        environ[6, 0:2] = 6
        environ[6, 2] = 2
        environ[6, 3] = 7
        environ[7, 0:2] = 7
        environ[7, 2] = 6
        environ[7, 3] = 8
        # end state 1
        environ[8, 0] = 8
        environ[8, 1] = 8  # 0
        environ[8, 2] = 7
        environ[8, 3] = 8
        if self.state_space:
            environ[9, 0] = 9
            environ[9, 1] = 9  # 0
            environ[9, 2] = 7
            environ[9, 3] = 9
        # end state 2
        environ[5, 0] = 5
        environ[5, 1] = 5  # 0
        environ[5, 2] = 5
        environ[5, 3] = 4

        if self.state_space:
            environ[10, 0] = 10
            environ[10, 1] = 10  # 0
            environ[10, 2] = 10
            environ[10, 3] = 4

        if self.long:
            # arm length is 2 init

            environ = np.zeros((self.n_states, self.n_actions))

            trans_states = np.arange(self.n_states - 2)
            if self.inf_task == 'struct_disc':
                trans_states = np.arange(self.n_states - 3)
            if self.inf_task == 'DNMS':
                trans_states = np.arange(self.n_states - 2)
            int_states = trans_states[trans_states > 8]

            # slot new states between 1 and 2
            for state_now in int_states:
                environ[state_now, 0] = state_now + 1
                environ[state_now, 2:4] = state_now
                environ[state_now, 1] = state_now - 1

                if state_now == 9:
                    environ[state_now, 1] = 1  # move back to 1

                if state_now == max(int_states):
                    environ[state_now, 0] = 2  # move forwards to state 2

            environ[0, 0] = 1
            environ[0, 1:4] = 0

            environ[1, 0] = min(int_states)  # forwards
            environ[1, 2:4] = 1
            environ[1, 1] = 0

            environ[2, 0] = 2  # forwards
            environ[2, 2] = 3  # left
            environ[2, 3] = 6  # right
            environ[2, 1] = max(int_states)  # backwards

            environ[3, 0:2] = 3
            environ[3, 2] = 4
            environ[3, 3] = 2
            environ[4, 0:2] = 4
            environ[4, 2] = 5
            environ[4, 3] = 3
            environ[5, 0] = 5
            environ[5, 1] = 0
            environ[5, 2] = 5
            environ[5, 3] = 4
            environ[6, 0:2] = 6
            environ[6, 2] = 2
            environ[6, 3] = 7
            environ[7, 0:2] = 7
            environ[7, 2] = 6
            environ[7, 3] = 8
            # end state 1
            environ[8, 0] = 8
            environ[8, 1] = 8  # 0
            environ[8, 2] = 7
            environ[8, 3] = 8
            # end state 2
            environ[5, 0] = 5
            environ[5, 1] = 5  # 0
            environ[5, 2] = 5
            environ[5, 3] = 4

        return environ

    def create_task(self, task_num):
        self.task_num = task_num
        rewards = np.zeros((self.n_states, self.n_actions))
        goal_states = []
        if self.inf_task == 'struct_disc':
            if task_num in [1, 4, 6]:  # 1,4,6 are rewarded
                goal_states = [5, 8]
                rewards[5, :] = 10
                # when in state 5 I receive this reward independent of the action I take
            elif task_num in [2, 3, 5]:  # 2,3,5 are unrewarded
                rewards[8, :] = 10
                goal_states = [5, 8]
        elif self.inf_task == 'DNMS':
            if task_num in [1, 4]:  # 1,4 are rewarded
                goal_states = [5, 8]
                rewards[5, :] = 10
                # when in state 5 I receive this reward independent of the action I take
            elif task_num in [2, 3]:  # 2,3,5 are unrewarded
                rewards[8, :] = 10
                goal_states = [5, 8]
        else:
            if task_num == 1:
                goal_states = [5]
                rewards[5, :] = 10
                # when in state 5 I receive this reward independent of the action I take
            elif task_num == 2:
                rewards[8, :] = 10
                goal_states = [8]
            elif task_num == 3:
                rewards[8, :] = np.random.choice([10, 0], 1, p=[0.7, 0.3])
                rewards[5, :] = np.random.choice([10, 0], 1, p=[0.1, 0.9])
            elif task_num == 4:
                rewards[5, :] = np.random.choice([10, 0], 1, p=[0.7, 0.3])
                rewards[8, :] = np.random.choice([10, 0], 1, p=[0.1, 0.9])

        if self.one_shot:
            goal_states = [5, 8]
        self.goal_state = goal_states
        starting_state = self.starting_state()
        return rewards, starting_state

    def next_state(self, learner, rewards):
        state = np.where(learner.state == 1)[0][0]  # lowest value of 1 is state
        action = learner.action
        if self.inf_task in ['struct_disc', 'DNMS']:
            if state not in [2, 3, 4, 5, 6, 7, 8]:
                action = 0
        flat_state = int(self.environ[state, action])
        if self.forced_choice:
            if state == 2:
                if self.task_num == 1:
                    if action == 3:
                        flat_state = int(2)
                elif self.task_num == 2:
                    if action == 2:
                        flat_state = int(2)
        state_new = np.zeros((self.n_states, 1))  # vector format
        reward_new = int(rewards[flat_state, action])
        if self.state_space:
            if reward_new == 0:
                if flat_state == 5:
                    flat_state = 10
                elif flat_state == 8:
                    flat_state = 9
        state_new[flat_state, :] = 1
        if self.inf_task == 'global_cue':
            # global feature
            if self.task_num == 1:
                state_new[-2, :] = self.cue_val
                state_new[-1, :] = 0
            elif self.task_num == 2:
                state_new[-1, :] = self.cue_val
                state_new[-2, :] = 0
        if self.inf_task == 'local':
            if (self.task_num == 1) & (flat_state in [1]):
                state_new[-2, :] = self.cue_val
            elif (self.task_num == 2) & (flat_state in [1]):
                state_new[-1, :] = self.cue_val

        if self.noise:
            if flat_state in [1]:
                trans_states = np.arange(self.n_states - 2)  # minus the cue features
                added_features = trans_states[trans_states > 8]  # these are the extra features
                state_new[added_features, :] = 1

        if self.reward_feature:
            if (self.task_num == 1) & (flat_state == 5):
                state_new[9, :] = 1
            elif (self.task_num == 2) & (flat_state == 8):
                state_new[9, :] = 1
        if self.inf_task == 'close_cue':
            if (self.task_num == 1) & (flat_state in [4]):
                state_new[-2, :] = self.cue_val
                state_new[-1, :] = 0
            elif (self.task_num == 2) & (flat_state in [7]):
                state_new[-1, :] = self.cue_val
                state_new[-2, :] = 0
        if self.inf_task == 'dual_cue':
            if self.task_num == 1:
                # task 1 has -2 cue first followed by -1 cue
                if flat_state in [1]:
                    state_new[-2, :] = self.cue_val
                elif flat_state in [2]:
                    state_new[-1, :] = self.cue_val
            elif self.task_num == 2:
                # task 2 has -1 cue first followed by -2 cue
                if flat_state in [1]:
                    state_new[-1, :] = self.cue_val
                elif flat_state in [2]:
                    state_new[-2, :] = self.cue_val

        if self.inf_task == 'struct_disc':
            # A->B->reward
            state_before_end = self.n_states - 4
            if self.task_num == 1:
                if flat_state in [1]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
            # A->C->no reward
            elif self.task_num == 2:
                if flat_state in [1]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # B->A->no reward
            elif self.task_num == 3:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0
            # B->C->reward
            elif self.task_num == 4:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # C->B->no reward
            elif self.task_num == 5:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [state_before_end]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
            # C->A-> reward
            elif self.task_num == 6:
                if flat_state in [1]:
                    state_new[-3, :] = 0
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [state_before_end]:
                    state_new[-3, :] = self.cue_val
                    state_new[-2, :] = 0
                    state_new[-1, :] = 0

        if self.inf_task == 'DNMS':
            state_before_end = self.n_states - 3
            # A->B->reward
            # A->A->no reward
            # B->B->no reward
            # B->A->reward

            # A->B->reward
            if self.task_num == 1:
                if flat_state in [1]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # A->A->no reward
            elif self.task_num == 2:
                if flat_state in [1]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
                if flat_state in [state_before_end]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0
            # B->B->no reward
            elif self.task_num == 3:
                if flat_state in [1]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [state_before_end]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
            # B->A->reward
            elif self.task_num == 4:
                if flat_state in [1]:
                    state_new[-2, :] = 0
                    state_new[-1, :] = self.cue_val
                if flat_state in [state_before_end]:
                    state_new[-2, :] = self.cue_val
                    state_new[-1, :] = 0

        return state_new, reward_new

    def expand_states(self, state_flat):
        i = state_flat
        if i == 0:
            state_matrix = [3, 0]
        elif i == 1:
            state_matrix = [3, 1]
        elif i == 2:
            state_matrix = [3, 2]
        elif i == 3:
            state_matrix = [2, 2]
        elif i == 4:
            state_matrix = [1, 2]
        elif i == 5:
            state_matrix = [0, 2]
        elif i == 6:
            state_matrix = [4, 2]
        elif i == 7:
            state_matrix = [5, 2]
        elif i == 8:
            state_matrix = [6, 2]
        return state_matrix

    def expand_state_action_values(self, state_action_values):
        # for t-maze
        grid_reward_pred = np.zeros((self.rows, self.columns, self.n_actions, len(state_action_values)))
        for i in np.arange(len(state_action_values)):
            states_here = np.reshape(np.asarray(state_action_values[i]),
                                     (self.n_states, self.n_actions))  # flattened states, by actions
            for j in np.arange(self.n_actions):
                for k in np.arange(self.n_states):
                    state_matrix = self.expand_states(k)
                    grid_reward_pred[state_matrix[0], state_matrix[1], j, i] = states_here[k, j]
        return grid_reward_pred

    def expand_state_value(self, state_values):
        grid_reward_pred = np.zeros((self.rows, self.columns, len(state_values)))
        for i in np.arange(len(state_values)):
            states_here = state_values[i]
            for k in np.arange(self.n_states):
                state_matrix = self.expand_states(k)
                grid_reward_pred[state_matrix[0], state_matrix[1], i] = states_here[k]
        return grid_reward_pred

    def expand_SR(self, SR_list):
        grid_reward_pred = np.zeros((self.rows, self.columns, self.rows, self.columns, self.n_actions, len(SR_list)))
        for i in np.arange(len(SR_list)):
            states_here = np.reshape(np.asarray(SR_list[i]),
                                     (self.n_states, self.n_states,
                                      self.n_actions))  # flattened states, by actions
            for j in np.arange(self.n_actions):
                for k in np.arange(self.n_states):
                    state_matrix = self.expand_states(k)
                    for m in np.arange(self.n_states):
                        state_matrix2 = self.expand_states(m)
                        grid_reward_pred[state_matrix2[0], state_matrix2[1], state_matrix[0], state_matrix[1], j, i] = \
                            states_here[m, k, j]
        return grid_reward_pred

    def expand_SR_S(self, SR_list):
        grid_reward_pred = np.zeros((self.rows, self.columns, self.rows, self.columns, len(SR_list)))
        for i in np.arange(len(SR_list)):
            states_here = SR_list[i]
            for k in np.arange(self.n_states):
                state_matrix = self.expand_states(k)
                for m in np.arange(self.n_states):
                    state_matrix2 = self.expand_states(m)
                    grid_reward_pred[state_matrix2[0], state_matrix2[1], state_matrix[0], state_matrix[1], i] = \
                        states_here[m, k]
        return grid_reward_pred
    # states are rows x state-actions are columns, to [row, row, column, actions, trial]
    # value for each state, given x,y,state,trial


def run_multiple(joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue, limit, outcome,
                 arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR, single_TD,
                 single_SR_unlearning, record_SRs, inf_task, CR_record):
    noise_input = noise_here
    # tasks = [1,2] * 40
    # trials = 5
    steps_record_500 = []
    steps_record_400 = []
    steps_record_300 = []
    steps_record_200 = []
    steps_record_100 = []

    trials_on_reversal = []
    trials_on_reversal_task1 = []
    trials_on_reversal_task2 = []

    percent_correct_100 = []
    percent_correct_100_task1 = []
    percent_correct_100_task2 = []
    percent_correct_200 = []
    percent_correct_300 = []
    percent_correct_400 = []
    percent_correct_500 = []
    percent_correct_500_task1 = []
    percent_correct_500_task2 = []

    n_switch_100 = []
    n_switch_200 = []
    n_switch_300 = []
    n_switch_400 = []
    n_switch_500 = []

    inc_switch_100 = []
    inc_switch_100_task1 = []
    inc_switch_100_task2 = []

    inc_switch_500 = []
    inc_switch_500_task1 = []
    inc_switch_500_task2 = []

    num_attempts_trials_100 = []
    SRs_100 = []
    prob_100 = []
    states_100 = []

    num_attempts_trials_start = []
    prob_start = []
    states_start = []
    SRs_start = []

    num_attempts_trials_end = []
    prob_end = []
    states_end = []

    # revision lists
    trial_outcomes = []
    task_ids = []
    locations = []
    probabilities_SR = []
    probabilities_CR = []
    CR_maps = []
    CR_recorded = []
    state_record_all = []  # records locations
    prob_record_SR = [] # records probabilities
    prob_record_CR = [] #prob_record_CR

    for run in np.arange(5):  # number of agents per file
        random.seed(run + rand_init_add)
        np.random.seed(run + rand_init_add)
        trials = 1

        if inf_task == 'struct_disc':
            conc = 0.8
            stickiness = 8
            tasks_rand = np.random.choice([1, 3, 2, 6, 4, 5], 500).tolist()
            tasks = ([1] * 50 \
                     + [4] * 50 \
                     + [6] * 50) * 3 \
                    + ([2] * 50 \
                       + [1] * 50 \
                       + [3] * 50 \
                       + [4] * 50 \
                       + [5] * 50 \
                       + [6] * 50) * 3 + tasks_rand
        elif inf_task == 'DNMS':
            stickiness = 8
            conc = 0.8
            tasks_rand = np.random.choice([1, 2, 3, 4],
                                          500).tolist()  # involves making a different choice, learning this will be hard and would require outcome niference?
            tasks = ([1] * 50 + [4] * 50) * 6 + ([2] * 50 + [1] * 50 + [3] * 50 + [4] * 50) * 5 + tasks_rand

        # test for block->interleaved strucutre: inf_task = ['random', 'block_10', 'block_20', 'block_30', 'block_40']
        elif inf_task in ['cue_random', 'noise_random']:
            conc = 0.1
            stickiness = 5
            tasks = np.random.choice([1, 2], 1500).tolist()
            tasks_rand = tasks[-500:]
            inf_task = 'local'
        elif inf_task in ['cue_block_1', 'noise_block_1']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = ([1] + [2]) * 500 + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_block_5', 'noise_block_5']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = ([1] * 5 + [2] * 5) * 100 + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_block_10', 'noise_block_10']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = ([1] * 10 + [2] * 10) * 50 + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_block_20', 'noise_block_20']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = ([1] * 20 + [2] * 20) * 25 + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_block_30', 'noise_block_30']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = ([1] * 30 + [2] * 30) * 17 + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_block_40', 'noise_block_40']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = ([1] * 40 + [2] * 40) * 13 + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_block_70', 'noise_block_70']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = ([1] * 70 + [2] * 70) * 8 + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_block_100', 'noise_block_100']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = ([1] * 100 + [2] * 100) * 5 + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_prob4', 'noise_prob4']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = []
            for block in np.arange(10):
                task_1 = np.array([1] * 48 + [2] * 2)
                np.random.shuffle(task_1)
                task_2 = np.array([2] * 48 + [1] * 2)
                np.random.shuffle(task_2)
                tasks = tasks + task_1.tolist() + task_2.tolist()
            tasks = tasks + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_prob10', 'noise_prob10']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = []
            for block in np.arange(10):
                task_1 = np.array([1] * 45 + [2] * 5)
                np.random.shuffle(task_1)
                task_2 = np.array([2] * 45 + [1] * 5)
                np.random.shuffle(task_2)
                tasks = tasks+task_1.tolist()+task_2.tolist()
            tasks = tasks + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_prob20', 'noise_prob20']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = []
            for block in np.arange(10):
                task_1 = np.array([1] * 40 + [2] * 10)
                np.random.shuffle(task_1)
                task_2 = np.array([2] * 40 + [1] * 10)
                np.random.shuffle(task_2)
                tasks = tasks+task_1.tolist()+task_2.tolist()
            tasks = tasks + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_prob30', 'noise_prob30']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = []
            for block in np.arange(10):
                task_1 = np.array([1] * 35 + [2] * 15)
                np.random.shuffle(task_1)
                task_2 = np.array([2] * 35 + [1] * 15)
                np.random.shuffle(task_2)
                tasks = tasks+task_1.tolist()+task_2.tolist()
            tasks = tasks + tasks_rand
            inf_task = 'local'
        elif inf_task in ['cue_prob40', 'noise_prob40']:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = []
            for block in np.arange(10):
                task_1 = np.array([1] * 30 + [2] * 20)
                np.random.shuffle(task_1)
                task_2 = np.array([2] * 30 + [1] * 20)
                np.random.shuffle(task_2)
                tasks = tasks+task_1.tolist()+task_2.tolist()
            tasks = tasks + tasks_rand
            inf_task = 'local'
        else:
            conc = 0.1
            stickiness = 5
            tasks_rand = np.random.choice([1, 2], 500).tolist()
            tasks = ([1] * 50 + [2] * 50) * 10 + tasks_rand
            inf_task = 'local'

        if explore:
            tasks = [100000000] + ([1] * 50 + [2] * 50) * 10 + tasks_rand

        if noise_input:
            long_here = False
            noise_here = True
        else:
            noise_here = False
            if arm_length == 2:
                long_here = False
            else:
                long_here = True

        world = EnvironmentT(tasks=tasks, trials=trials, state_space=False, inf_task=inf_task, feature=True,
                             cue_val=cue, one_shot=True,
                             reward_feature=reward_feat, limit=limit, joint_prior=joint_prior, joint_inf=joint_inf,
                             replay=replay, long=long_here,
                             arm_length=arm_length, noise=noise_here,
                             fc=fc)  # tasks = [1, 2] * 500) #cue is usually in 1
        if outcome:
            ret_n_trials = []
            ret_perf = []

            learner = BSRAgent_SS(world, starting_state=0, filter=2, SR_learning_rate=0.2, stickiness=0.7,
                                  particle_full=False,
                                  kalman_SR=False, kalman=True, multi=False, conc=conc, covariance_tracking=out_cov,
                                  TD=td, kalman_TD=kalm_td, fake_outcome=fake_out)
            ret_perf, ret_n_trials, prob_record_start, state_record_start, prob_record_end, state_record_end, state_record, prob_record, CR_recorded, prob_record_CR, state_record_all = episode_run(
                world, learner, ret_n_trials, ret_perf, CR_record)
            SR_record = []
            SR_record_start = []

        elif single_SR:
            ret_perf = []
            ret_n_trials = []
            learner = SR_SF_Agent(world, starting_state=0, kalman_SR=True, kalman=False, TD=False, kalman_TD=False)
            ret_perf, ret_n_trials, prob_record_start, state_record_start, prob_record_end, state_record_end, state_record, prob_record, CR_recorded, prob_record_CR, state_record_all = episode_run(
                world, learner, ret_n_trials, ret_perf, CR_record)
            SR_record = []
            SR_record_start = []

        elif single_SR_unlearning:
            # kalman parameter is lr rw=1
            ret_perf = []
            ret_n_trials = []
            learner = SR_SF_Agent(world, starting_state=0, kalman_SR=True, kalman=True, TD=False, kalman_TD=False)
            ret_perf, ret_n_trials, prob_record_start, state_record_start, prob_record_end, state_record_end, state_record, prob_record, CR_recorded, prob_record_CR, state_record_all = episode_run(
                world, learner, ret_n_trials, ret_perf, CR_record)
            SR_record = []
            SR_record_start = []

        elif single_TD:
            ret_perf = []
            ret_n_trials = []
            learner = SR_SF_Agent(world, starting_state=0, kalman_SR=False, kalman=False, TD=True, kalman_TD=True)
            ret_perf, ret_n_trials, prob_record_start, state_record_start, prob_record_end, state_record_end, state_record, prob_record, CR_recorded, prob_record_CR, state_record_all = episode_run(
                world, learner, ret_n_trials, ret_perf, CR_record)
            SR_record = []
            SR_record_start = []

        elif not outcome:
            learner = BSRAgent_SS(world, starting_state=0, filter=2, SR_learning_rate=0.2, stickiness=0.7,
                                  particle_full=False,
                                  kalman_SR=False, kalman=True, multi=True, conc=conc, covariance_tracking=out_cov,
                                  TD=td, kalman_TD=kalm_td, fake_outcome=fake_out)
            learner_feature = feature_BSRAgent_SS(world, starting_state=0, inference='all_features',
                                                  inf_concentration=0.5,
                                                  SR_learning_rate=0.2, stickiness=stickiness, particle_full=False,
                                                  kalman_SR=True,
                                                  kalman=False, covariance_tracking=feat_cov,
                                                  kalman_replay=kalman_replay, TD=td, kalman_TD=kalm_td)
            # updated the conc of outcome to 0.3 from 0.2
            # nf_concentration=2, SR_learning_rate=0.2, stickiness=1 works for joint 7

            # for sr 5: #conc = 2, stick = 1; or #stick=2, conc=2

            ret_n_trials = []
            ret_perf = []

            ret_perf, ret_n_trials, out_conf_list, feat_conf_list, err, SR_record, state_record, prob_record, prob_record_start, state_record_start, prob_record_end, state_record_end, SR_record_start, state_record_all, prob_record_SR, prob_record_CR = dual_episode_run(
                world, learner,
                learner_feature,
                ret_n_trials, ret_perf, record_SRs)

        try_me = np.array(ret_n_trials[-100:])
        per_corr_100 = len(try_me[try_me == 1])
        print(per_corr_100)

        try_me = np.array(ret_n_trials[-200:-100])
        per_corr_200 = len(try_me[try_me == 1])

        try_me = np.array(ret_n_trials[-300:-200])
        per_corr_300 = len(try_me[try_me == 1])

        try_me = np.array(ret_n_trials[-400:-300])
        per_corr_400 = len(try_me[try_me == 1])

        try_me = np.array(ret_n_trials[-500:-400])
        per_corr_500 = len(try_me[try_me == 1])

        trials = np.array(ret_n_trials[-100:])
        tasks_here = np.array(tasks_rand[-100:])
        task_1 = trials[tasks_here == 1]
        task_2 = trials[tasks_here == 2]
        per_corr_100_task1 = len(task_1[task_1 == 1]) / len(task_1)
        per_corr_100_task2 = len(task_2[task_2 == 1]) / len(task_2)

        trials = np.array(ret_n_trials[-500:-400])
        tasks_here = np.array(tasks_rand[-500:-400])
        task_1 = trials[tasks_here == 1]
        task_2 = trials[tasks_here == 2]
        per_corr_500_task1 = len(task_1[task_1 == 1]) / len(task_1)
        per_corr_500_task2 = len(task_2[task_2 == 1]) / len(task_2)

        # last_hund = np.sum(ret_n_trials[-400:])
        # all = ret_n_trials[-400:]
        # all = np.array(all)
        # percent_correct[run] = len(all[all == 1])
        if inf_task == 'struct_disc':
            revs = np.mean(ret_n_trials[17 * 50:27 * 50:50])
            revs_1 = np.mean(ret_n_trials[10 * 50:20 * 50:100])  # from task 2 to task 1
            revs_2 = np.mean(ret_n_trials[11 * 50:20 * 50:100])  # from task 1 to task 2
        elif inf_task == 'DNMS':
            revs = np.mean(ret_n_trials[22 * 50:32 * 50:50])
            revs_1 = np.mean(ret_n_trials[22 * 50:32 * 50:100])  # from task 2 to task 1
            revs_2 = np.mean(ret_n_trials[21 * 50:31 * 50:100])  # from task 1 to task 2
        else:
            revs = np.mean(ret_n_trials[10 * 50:20 * 50:50])
            revs_1 = np.mean(ret_n_trials[10 * 50:20 * 50:100])  # from task 2 to task 1
            revs_2 = np.mean(ret_n_trials[11 * 50:20 * 50:100])  # from task 1 to task 2

        steps_record_100.append(np.mean(learner.time_per_episode[-100:]))
        steps_record_200.append(np.mean(learner.time_per_episode[-200:-100]))
        steps_record_300.append(np.mean(learner.time_per_episode[-300:-200]))
        steps_record_400.append(np.mean(learner.time_per_episode[-400:-300]))
        steps_record_500.append(np.mean(learner.time_per_episode[-500:-400]))

        percent_correct_100.append(per_corr_100)
        percent_correct_200.append(per_corr_200)
        percent_correct_300.append(per_corr_300)
        percent_correct_400.append(per_corr_400)
        percent_correct_500.append(per_corr_500)
        percent_correct_100_task1.append(per_corr_100_task1)
        percent_correct_100_task2.append(per_corr_100_task2)
        percent_correct_500_task1.append(per_corr_500_task1)
        percent_correct_500_task2.append(per_corr_500_task2)

        trials_on_reversal.append(revs)
        trials_on_reversal_task1.append(revs_1)  # from task 2 to task 1
        trials_on_reversal_task2.append(revs_2)  # from task 1 to task 2

        try_me = np.array(ret_n_trials[-100:])
        num_attempts_trials_100.append(try_me)

        SRs_100.append(SR_record)
        prob_100.append(prob_record)
        states_100.append(state_record)

        try_me = np.array(ret_n_trials[0:200])
        num_attempts_trials_start.append(try_me)
        prob_start.append(prob_record_start)
        states_start.append(state_record_start)
        SRs_start.append(SR_record_start)

        try_me = np.array(ret_n_trials[800:1000])
        num_attempts_trials_end.append(try_me)
        prob_end.append(prob_record_end)
        states_end.append(state_record_end)

        # number to switch
        inds = np.where(np.asarray(tasks_rand[:-1]) != np.asarray(tasks_rand[1:]))[0]
        int_inds = inds[inds < 100]
        num_to_switch = np.asarray(ret_n_trials)[int_inds + 1001]
        n_switch_100.append(np.mean(num_to_switch))
        num_to_switch_incorr_trials = num_to_switch[num_to_switch != 1]
        inc_switch_100.append(np.mean(num_to_switch_incorr_trials))

        int_inds = inds[(inds < 200) & (inds >= 100)]
        num_to_switch = np.asarray(ret_n_trials)[int_inds + 1001]
        n_switch_200.append(np.mean(num_to_switch))

        int_inds = inds[(inds < 300) & (inds >= 200)]
        num_to_switch = np.asarray(ret_n_trials)[int_inds + 1001]
        n_switch_300.append(np.mean(num_to_switch))

        int_inds = inds[(inds < 400) & (inds >= 300)]
        num_to_switch = np.asarray(ret_n_trials)[int_inds + 1001]
        n_switch_400.append(np.mean(num_to_switch))

        int_inds = inds[(inds < 500) & (inds >= 400)]
        num_to_switch = np.asarray(ret_n_trials)[int_inds + 1001]
        n_switch_500.append(np.mean(num_to_switch))
        num_to_switch_incorr_trials = num_to_switch[num_to_switch != 1]
        inc_switch_500.append(np.mean(num_to_switch_incorr_trials))

        # find trials before a switch
        inds = np.where(np.asarray(tasks_rand[:-1]) != np.asarray(tasks_rand[1:]))[0]
        inds_1 = inds[np.asarray(tasks_rand)[inds] == 2]
        inds_2 = inds[np.asarray(tasks_rand)[inds] == 1]

        int_inds = inds_1[inds_1 < 100]
        num_to_switch = np.asarray(ret_n_trials)[int_inds + 1001]
        num_to_switch_incorr_trials = num_to_switch[num_to_switch != 1]
        inc_switch_100_task1.append(np.mean(num_to_switch_incorr_trials))

        int_inds = inds_2[inds_2 < 100]
        num_to_switch = np.asarray(ret_n_trials)[int_inds + 1001]
        num_to_switch_incorr_trials = num_to_switch[num_to_switch != 1]
        inc_switch_100_task2.append(np.mean(num_to_switch_incorr_trials))

        int_inds = inds_1[(inds_1 < 500) & (inds_1 >= 400)]
        num_to_switch = np.asarray(ret_n_trials)[int_inds + 1001]
        num_to_switch_incorr_trials = num_to_switch[num_to_switch != 1]
        inc_switch_500_task1.append(np.mean(num_to_switch_incorr_trials))

        int_inds = inds_2[(inds_2 < 500) & (inds_2 >= 400)]
        num_to_switch = np.asarray(ret_n_trials)[int_inds + 1001]
        num_to_switch_incorr_trials = num_to_switch[num_to_switch != 1]
        inc_switch_500_task2.append(np.mean(num_to_switch_incorr_trials))

        # needed for learning curve
        trial_outcomes.append(ret_n_trials)  # number of tries to reverse
        task_ids.append(tasks)  # task identities
        # analysis: plot proportion correct around reversals in 1 plot?

        # needed for probability comparison within trials
        locations.append(state_record_all)  # records locations
        probabilities_SR.append(prob_record_SR)  # records probabilities
        probabilities_CR.append(prob_record_CR)  # records probabilities
        CR_maps.append(CR_recorded)  # last 100 CR maps
        # analysis: once well learnt probability colormaps in each state (last 500 block and last 100 random trials)

    return steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations,probabilities_SR, probabilities_CR, CR_maps


def chosen_ones(i):
    # define data type here!
    data_type = 'noise_block_1' # in cue_SRstart, noise_SRstart, noise, cue, fig_1, other_noise, other_cue, struct_disc, DNMS
    # for different setups data_type in ['random', 'block_10', 'block_20', 'block_30', 'block_40']

    if 'noise' in data_type:
        noise_here = True  # set up whether noise or cue task
    else:
        noise_here = False

    # init everything to False
    joint_prior = False
    reward_feat = 0
    joint_inf = False
    replay = False
    kalman_replay = False
    out_cov = False
    feat_cov = False
    outcome = False
    explore = False
    limit = 10000000
    fc = 0
    single_SR = False
    single_TD = False
    single_SR_unlearning = False
    record_SRs = False
    CR_record = False

    td = 1  # turns on TD for decision making
    kalm_td = 1  # use kalman learning rate for TD maps
    fake_out = 0

    rand_init_adds = [10, 15, 20, 25, 30, 35, 40, 45]
    if data_type == 'DNMS':
        cues = [2]
    else:
        cues = [4]
    if data_type in ['struct_disc', 'DNMS']:
        arm_lengths = [3]
    else:
        arm_lengths = [2, 5, 7, 10, 12, 15, 17, 20]

    if data_type in ['cue_SRstart', 'noise_SRstart']:
        groups = ['switching_SR', 'joint_inf_priors']
        record_SRs = True  # careful these files are large (up to 20GB)

    if data_type in ['noise', 'cue']:
        groups = ['switching_SR', 'joint_inf_priors']

    if data_type in ['fig_1']:
        groups = ['single_SR', 'single_TD', 'single_SR_unlearning']
        arm_lengths = [2]

    if data_type in ['other_noise', 'other_cue']:
        groups = ['switching_SR', 'joint_inf_priors', 'rew_feat', 'fc', 'explore', 'fake_out', 'replay']

    if data_type in ['struct_disc', 'DNMS']:
        groups = ['switching_SR', 'joint_inf_priors']

    # testing for different training structures
    if data_type in ['cue_random', 'cue_block_5', 'cue_block_10', 'cue_block_20', 'cue_block_30', 'cue_block_40', 'noise_random', 'noise_block_5', 'noise_block_10', 'noise_block_20', 'noise_block_30', 'noise_block_40', 'noise_block_70', 'noise_block_100', 'cue_block_70', 'cue_block_100', 'cue_block_1', 'noise_block_1']:
        groups = ['switching_SR', 'joint_inf_priors', 'outcome']

    if data_type in ['cue_prob4', 'cue_prob10', 'cue_prob20', 'cue_prob30', 'cue_prob40', 'noise_prob4', 'noise_prob10', 'noise_prob20', 'noise_prob30', 'noise_prob30']:
        groups = ['switching_SR', 'joint_inf_priors', 'outcome']

    if data_type in ['cue_CR_outcome', 'noise_CR_outcome']:
        groups = ['outcome']
        CR_record = True # records CRs on last 500 block trials

    if data_type in ['cue_dist', 'noise_dist']:
        groups = [50,100,200,500,750,1000,1250,1500]

    input_list = [cues, arm_lengths, groups, rand_init_adds]
    inputs = list(itertools.product(*input_list))

    cue = inputs[i][0]
    arm_length = inputs[i][1]
    rand_init_add = inputs[i][3]
    group = inputs[i][2]

    if data_type in ['cue_dist', 'noise_dist']:
        limit_set = group
        group = 'joint_inf_priors'

    if group == 'switching_SR':
        cov_types = [[0, 1]]  # first is feat, second is out cov tracking
        for cov in cov_types:
            feat_cov = False
            out_cov = True

            filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(
                cov[0]) + '_cov_out_' + str(cov[1]) + str(rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
                kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat) + 'data_type' + str(data_type)
            print(filename)
            steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
                joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue,
                limit, outcome, arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR,
                single_TD, single_SR_unlearning, record_SRs, data_type, CR_record)
            with open(filename, "wb") as fp:  # Pickling
                pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                             steps_record_100, trials_on_reversal, percent_correct_100,
                             percent_correct_200, percent_correct_300, percent_correct_400,
                             percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                             n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                             percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                             percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                             inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100,
                             SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                             num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)

    elif group == 'outcome':
        outcome = True
        out_cov = 1
        filename = str(arm_length) + '_' + group + '_cov_out_' + str(out_cov) + str(rand_init_add) + 'td' + str(
            td) + 'kalmtd' + str(kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat) + 'data_type' + str(data_type)
        print(filename)
        steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
            joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue, limit, outcome,
            arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR, single_TD,
            single_SR_unlearning, record_SRs, data_type, CR_record)
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump(
                [steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100,
                 trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300,
                 percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                 n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                 percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                 percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500,
                 inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100,
                 num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps],
                fp)

    elif group == 'replay':
        replay = True
        kalman_replay = True
        limit = 200
        cov = [0, 1]
        feat_cov = False
        out_cov = True

        filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(
            cov[0]) + '_cov_out_' + str(cov[1]) + str(rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
            kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat)
        print(filename)
        steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
            joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue, limit, outcome,
            arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR, single_TD,
            single_SR_unlearning, record_SRs, data_type, CR_record)
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                         steps_record_100, trials_on_reversal, percent_correct_100,
                         percent_correct_200, percent_correct_300, percent_correct_400,
                         percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                         n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                         percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                         percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                         inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100,
                         prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                         num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)

    elif group == 'joint_inf_priors':
        joint_prior = True
        joint_inf = True
        cov = [0, 1]
        feat_cov = False
        out_cov = True

        if data_type == 'struct_disc':
            limit = 1350
        elif data_type == 'DNMS':
            limit = 1600
        elif data_type in ['cue_dist', 'noise_dist']:
            limit = limit_set
        else:
            limit = 200
        filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(
            cov[0]) + '_cov_out_' + str(cov[1]) + str(rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
            kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat) + 'lim' + str(
            limit) + 'data_type' + str(data_type)
        print(filename)
        steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
            joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue, limit, outcome,
            arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR, single_TD,
            single_SR_unlearning, record_SRs, data_type, CR_record)
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                         steps_record_100, trials_on_reversal, percent_correct_100,
                         percent_correct_200, percent_correct_300, percent_correct_400,
                         percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                         n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                         percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                         percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                         inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100,
                         prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                         num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)

    elif group == 'explore':
        # has pre-exploration phase and has map at the beginning of stuff, switching SR, pre-exploration and covariance transferred to task 1/task 2 maps
        cov_types = [[0, 1]]  # first is feat, second is out cov tracking
        for cov in cov_types:
            feat_cov = False
            out_cov = True
            explore = True

            filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(
                cov[0]) + '_cov_out_' + str(cov[1]) + str(rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
                kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat)
            print(filename)
            steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
                joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue,
                limit, outcome, arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR,
                single_TD, single_SR_unlearning, record_SRs, data_type, CR_record)
            with open(filename, "wb") as fp:  # Pickling
                pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                             steps_record_100, trials_on_reversal, percent_correct_100,
                             percent_correct_200, percent_correct_300, percent_correct_400,
                             percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                             n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                             percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                             percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                             inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100,
                             SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                             num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)

    elif group == 'fc':
        # has forced choice at the beginning, switching SR
        limit = 200
        cov_types = [[0, 1]]  # first is feat, second is out cov tracking
        for cov in cov_types:
            feat_cov = False
            out_cov = True
            fc = 1

            filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(
                cov[0]) + '_cov_out_' + str(cov[1]) + str(rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
                kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat) + '_fc'
            print(filename)
            steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
                joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue, limit, outcome,
                arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR, single_TD,
                single_SR_unlearning, record_SRs, data_type, CR_record)
            with open(filename, "wb") as fp:  # Pickling
                pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                             steps_record_100, trials_on_reversal, percent_correct_100,
                             percent_correct_200, percent_correct_300, percent_correct_400,
                             percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                             n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                             percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                             percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                             inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100,
                             SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                             num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)

    elif group == 'rew_feat':
        # has forced choice at the beginning, switching SR
        cov_types = [[0, 1]]  # first is feat, second is out cov tracking
        for cov in cov_types:
            feat_cov = False
            out_cov = True
            reward_feat = 1

            filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(
                cov[0]) + '_cov_out_' + str(cov[1]) + str(rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
                kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat)
            print(filename)
            steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
                joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue, limit, outcome,
                arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR, single_TD,
                single_SR_unlearning, record_SRs, data_type, CR_record)
            with open(filename, "wb") as fp:  # Pickling
                pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                             steps_record_100, trials_on_reversal, percent_correct_100,
                             percent_correct_200, percent_correct_300, percent_correct_400,
                             percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                             n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                             percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                             percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                             inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100,
                             SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                             num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)

    elif group == 'fake_out':
        # fake outcome during learning
        limit = 200
        joint_inf = True
        joint_prior = True
        cov_types = [[0, 1]]  # first is feat, second is out cov tracking
        for cov in cov_types:
            feat_cov = False
            out_cov = True
            fake_out = 1

            filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(
                cov[0]) + '_cov_out_' + str(cov[1]) + str(rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
                kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat)
            print(filename)
            steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
                joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue, limit, outcome,
                arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR, single_TD,
                single_SR_unlearning, record_SRs, data_type, CR_record)
            with open(filename, "wb") as fp:  # Pickling
                pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                             steps_record_100, trials_on_reversal, percent_correct_100,
                             percent_correct_200, percent_correct_300, percent_correct_400,
                             percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                             n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                             percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                             percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                             inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100,
                             SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                             num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)

    elif group == 'single_SR':
        single_SR = True
        filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(0) + '_cov_out_' + str(0) + str(
            rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
            kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat)
        print(filename)
        steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
            joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue,
            limit, outcome, arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR,
            single_TD, single_SR_unlearning, record_SRs, data_type, CR_record)
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                         steps_record_100, trials_on_reversal, percent_correct_100,
                         percent_correct_200, percent_correct_300, percent_correct_400,
                         percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                         n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                         percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                         percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                         inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100,
                         prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                         num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)

    elif group == 'single_SR_unlearning':
        single_SR_unlearning = True
        filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(0) + '_cov_out_' + str(0) + str(
            rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
            kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat)
        print(filename)
        steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
            joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue,
            limit, outcome, arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR,
            single_TD, single_SR_unlearning, record_SRs, data_type, CR_record)
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                         steps_record_100, trials_on_reversal, percent_correct_100,
                         percent_correct_200, percent_correct_300, percent_correct_400,
                         percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                         n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                         percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                         percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                         inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100,
                         prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                         num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)

    elif group == 'single_TD':
        single_TD = True
        filename = str(arm_length) + '_' + str(cue) + group + '_cov_feat_' + str(0) + '_cov_out_' + str(0) + str(
            rand_init_add) + 'td' + str(td) + 'kalmtd' + str(
            kalm_td) + 'fake_out' + str(fake_out) + 'rew_feat' + str(reward_feat)
        print(filename)
        steps_record_500, steps_record_400, steps_record_300, steps_record_200, steps_record_100, trials_on_reversal, percent_correct_100, percent_correct_200, percent_correct_300, percent_correct_400, percent_correct_500, n_switch_100, n_switch_200, n_switch_300, n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2, percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1, percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2, inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100, SRs_100, prob_100, states_100, num_attempts_trials_start, prob_start, states_start, num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps = run_multiple(
            joint_prior, joint_inf, replay, kalman_replay, out_cov, feat_cov, reward_feat, cue,
            limit, outcome, arm_length, rand_init_add, td, kalm_td, fake_out, noise_here, explore, fc, single_SR,
            single_TD, single_SR_unlearning, record_SRs, data_type, CR_record)
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump([steps_record_500, steps_record_400, steps_record_300, steps_record_200,
                         steps_record_100, trials_on_reversal, percent_correct_100,
                         percent_correct_200, percent_correct_300, percent_correct_400,
                         percent_correct_500, n_switch_100, n_switch_200, n_switch_300,
                         n_switch_400, n_switch_500, trials_on_reversal_task1, trials_on_reversal_task2,
                         percent_correct_100_task1, percent_correct_100_task2, percent_correct_500_task1,
                         percent_correct_500_task2, inc_switch_100, inc_switch_100_task1, inc_switch_100_task2,
                         inc_switch_500, inc_switch_500_task1, inc_switch_500_task2, num_attempts_trials_100,
                         SRs_100,
                         prob_100, states_100, num_attempts_trials_start, prob_start, states_start,
                         num_attempts_trials_end, prob_end, states_end, SRs_start, trial_outcomes, task_ids, locations, probabilities_SR, probabilities_CR, CR_maps], fp)


if __name__ == '__main__':
    run_opts = True  # will run 40 agents of the defined type
    run_single = False  # runs a single agent, generates figure 1b example plots
    run_on = 'multi_node'  # 'single_node' multiprocessing on a single node; if not run on multiple nodes
    # request number of cores corresponding to len(index) in run_opts function
    data_type = 'noise'  # in cue_SRstart, noise_SRstart, noise, cue, fig_1, other_noise, other_cue, struct_disc, DNMS
    # for revisions:
    # CR maps ['cue_CR_outcome', 'noise_CR_outcome']
    # different lengths of joint inference ['cue_dist', 'noise_dist']
    # different block lengths ['cue_block_5', 'cue_block_10', 'cue_block_20', 'cue_block_30', 'cue_block_40', 'noise_block_5', 'noise_block_10', 'noise_block_20', 'noise_block_30', 'noise_block_40', 'noise_block_70', 'noise_block_100', 'cue_block_70', 'cue_block_100', 'cue_block_1', 'noise_block_1']
    # different block probabilities ['cue_random', 'cue_prob4', 'cue_prob10', 'cue_prob20', 'cue_prob30', 'cue_prob40', 'noise_random', 'noise_prob4', 'noise_prob10', 'noise_prob20', 'noise_prob30', 'noise_prob30']
    # data_type must be set here and in chosen_ones function above!

    rand_init_adds_here = [10, 15, 20, 25, 30, 35, 40, 45]
    arm_lengths_here = [2, 5, 7, 10, 12, 15, 17, 20]

    if data_type in ['cue_SRstart', 'noise_SRstart']:
        groups_here = ['switching_SR', 'joint_inf_priors']

    if data_type in ['noise', 'cue']:
        groups_here = ['switching_SR', 'joint_inf_priors']

    if data_type in ['fig_1']:
        groups_here = ['switching_SR', 'outcome', 'single_SR', 'single_TD', 'single_SR_unlearning']
        arm_lengths_here = [2]

    if data_type in ['other_noise', 'other_cue']:
        groups_here = ['switching_SR', 'joint_inf_priors', 'rew_feat', 'fc', 'explore', 'fake_out', 'replay']

    if data_type in ['struct_disc', 'DNMS']:
        arm_lengths_here = [3]
        groups_here = ['switching_SR', 'joint_inf_priors']

    # testing for different training structures
    if data_type in ['cue_CR_outcome', 'noise_CR_outcome']:
        groups_here = ['outcome']

    if data_type in ['cue_random', 'cue_block_5', 'cue_block_10', 'cue_block_20', 'cue_block_30', 'cue_block_40', 'noise_random', 'noise_block_5', 'noise_block_10', 'noise_block_20', 'noise_block_30', 'noise_block_40', 'noise_block_70', 'noise_block_100', 'cue_block_70', 'cue_block_100', 'cue_block_1', 'noise_block_1']:
        groups_here = ['switching_SR', 'joint_inf_priors', 'outcome']

    if data_type in ['cue_prob4', 'cue_prob10', 'cue_prob20', 'cue_prob30', 'cue_prob40', 'noise_prob4', 'noise_prob10', 'noise_prob20', 'noise_prob30', 'noise_prob30']:
        groups_here = ['switching_SR', 'joint_inf_priors', 'outcome']

    if data_type in ['cue_dist', 'noise_dist']:
        groups_here = [50,100,200,500,750,1000,1250,1500]

    if run_opts:
        if run_on == 'single_node':
            from multiprocessing import Pool
            import multiprocessing

            if data_type == 'DNMS':
                cues = [2]
            else:
                cues = [4]
            groups = groups_here
            arm_lengths = arm_lengths_here
            rand_init_adds = rand_init_adds_here

            input_list = [cues, arm_lengths, groups, rand_init_adds]
            inputs = list(itertools.product(*input_list))
            index = np.arange(len(inputs))  # indices of different parameter choices that are being investigated

            max_cpu = multiprocessing.cpu_count()
            pool = Pool(
                processes=max_cpu)  # defines the amount of things you want to run in parallel, would recommend using AT LEAST one cpu less than your computer has
            a = pool.map(chosen_ones, index)

        else:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if comm.rank == 0:
                if data_type == 'DNMS':
                    cues = [2]
                else:
                    cues = [4]
                groups = groups_here
                arm_lengths = arm_lengths_here
                rand_init_adds = rand_init_adds_here

                input_list = [cues, arm_lengths, groups, rand_init_adds]
                inputs = list(itertools.product(*input_list))
                index = np.arange(len(inputs))  # indices of different parameter choices that are being investigated
                work_list = index
            else:
                work_list = None

            my_work = comm.scatter(work_list)
            chosen_ones(my_work)

    elif run_single:

        random.seed(30)  # random seed 3
        np.random.seed(30)
        trials = 1

        tasks_rand = np.random.choice([1, 2], 500).tolist()
        tasks = ([1] * 50 + [2] * 50) * 10 + tasks_rand

        joint_prior = False
        reward_feat = False
        joint_inf = False
        replay = False
        kalman_replay = False
        out_cov = False
        feat_cov = False
        outcome = False
        limit = 10000000
        fc = 0
        record_SRs = False

        td = True
        kalm_td = True
        fake_out = False

        group = 'switching_SR'
        arm_length = 2
        cue = 4

        if group == 'switching_SR':
            feat_cov = False
            out_cov = False

        elif group == 'outcome':
            outcome = True
            out_cov = True

        elif group == 'joint_inference':
            joint_inf = True
            feat_cov = False
            out_cov = True
            limit = 200

        elif group == 'joint_inf_priors':
            joint_prior = True
            joint_inf = True
            feat_cov = False
            out_cov = True
            limit = 200

        elif group == 'replay':
            replay = True
            # constant replay
            kalman_replay = False
            limit = 500
            kal_replay = False
            cov = [0, 1]
            feat_cov = False
            out_cov = True

        if arm_length == 2:
            long_here = False
            noise_here = False
        else:
            long_here = True
            noise_here = False

        world = EnvironmentT(tasks=tasks, trials=trials, state_space=False, inf_task='local', feature=True,
                             cue_val=cue, one_shot=True,
                             reward_feature=reward_feat, limit=limit, joint_prior=joint_prior, joint_inf=joint_inf,
                             replay=replay, long=long_here,
                             arm_length=arm_length, noise=noise_here,
                             fc=fc)  # tasks = [1, 2] * 500) #cue is usually in 1
        if not outcome:
            learner = BSRAgent_SS(world, starting_state=0, filter=2, SR_learning_rate=0.2, stickiness=0.7,
                                  particle_full=False,
                                  kalman_SR=False, kalman=True, multi=True, conc=0.1, covariance_tracking=out_cov,
                                  TD=td, kalman_TD=kalm_td, fake_outcome=fake_out)
            learner_feature = feature_BSRAgent_SS(world, starting_state=0, inference='all_features',
                                                  inf_concentration=0.5,
                                                  SR_learning_rate=0.2, stickiness=5, particle_full=False,
                                                  kalman_SR=True,
                                                  kalman=False, covariance_tracking=feat_cov,
                                                  kalman_replay=kalman_replay, TD=td, kalman_TD=kalm_td)
            # updated the conc of outcome to 0.3 from 0.2
            # nf_concentration=2, SR_learning_rate=0.2, stickiness=1 works for joint 7

            # for sr 5: #conc = 2, stick = 1; or #stick=2, conc=2
            ret_n_trials = []
            ret_perf = []

            ret_perf, ret_n_trials, out_conf_list, feat_conf_list, err, SR_record, state_record, prob_record, prob_record_start, state_record_start, prob_record_end, state_record_end, SR_record_start = dual_episode_run(
                world, learner,
                learner_feature,
                ret_n_trials, ret_perf, record_SRs)
        if outcome:
            ret_n_trials = []
            ret_perf = []

            learner = BSRAgent_SS(world, starting_state=0, filter=2, SR_learning_rate=0.2, stickiness=0.7,
                                  particle_full=False,
                                  kalman_SR=False, kalman=True, multi=False, conc=0.1, covariance_tracking=out_cov,
                                  TD=td, kalman_TD=kalm_td, fake_outcome=fake_out)
            ret_perf, ret_n_trials, prob_record_start, state_record_start, prob_record_end, state_record_end = episode_run(
                world, learner, ret_n_trials, ret_perf)

        try_me = np.array(ret_n_trials[-100:])
        print(len(try_me[try_me == 1]))

        print(np.mean(ret_n_trials[10 * 50:20 * 50:50]))

        # plots for figure 1b
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams.update({'font.size': 60})
        plt.rcParams["figure.figsize"] = (40, 4)
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['pdf.use14corefonts'] = True
        matplotlib.rcParams['text.usetex'] = False
        matplotlib.rcParams['font.family'] = 'Myriad Pro'
        matplotlib.rcParams['font.sans-serif'] = 'Myriad Pro'

        # task setup
        fig, ax = plt.subplots()
        plt.margins(x=0)
        ax.plot(tasks, c='#666666', linewidth=2)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.yticks([1, 2], ['1', '2'])
        plt.xticks([500, 1000, 1500], [500, 1000, 1500])
        plt.ylabel('Task')
        plt.xlabel('Trial number')
        plt.savefig('figures/fig_2b.pdf', dpi=500, bbox_inches='tight', format='pdf')
        plt.show()

        # block trials
        plt.rcParams["figure.figsize"] = (20, 2)
        # zoom in on
        fig, ax = plt.subplots()
        plt.margins(x=0)
        ax.plot(np.arange(500, 1000), tasks[500:1000], c='#666666', linewidth=2)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.yticks([1, 2], ['', ''])
        plt.xticks([500, 750, 1000], ['', '', ''])
        # plt.ylabel('Task')
        plt.xticks([], [])
        plt.savefig('figures/fig_2c_lefttop.pdf', dpi=500, bbox_inches='tight', format='pdf')
        plt.show()

        matplotlib.rcParams.update({'font.size': 20})
        plt.rcParams["figure.figsize"] = (20, 2)
        fig, ax = plt.subplots()
        plt.margins(x=0)
        plt.plot(np.arange(500, 1000), ret_n_trials[500:1000], c='#666666', linewidth=2)
        plt.yticks([1, 2], ['', ''])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.xticks([500, 750, 1000], ['', '', ''])
        # plt.ylabel('Performance')
        # plt.xlabel('Trial number')
        plt.savefig('figures/fig_2c_leftbottom.pdf', dpi=500, bbox_inches='tight', format='pdf')
        plt.show()

        # random trials
        plt.rcParams["figure.figsize"] = (20, 2)
        # zoom in on
        fig, ax = plt.subplots()
        plt.margins(x=0)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.yticks([1, 2], ['', ''])
        plt.xticks([500, 750, 1000], ['', '', ''])
        # plt.ylabel('Task')
        plt.xticks([], [])
        ax.plot(np.arange(1400, 1500), tasks[1400:1500], c='#666666', linewidth=2)
        plt.xticks([], [])
        # plt.ylabel('Task')
        plt.savefig('figures/fig_2c_rightbottom.pdf', dpi=500, bbox_inches='tight', format='pdf')
        plt.show()

        plt.rcParams["figure.figsize"] = (20, 2)
        fig, ax = plt.subplots()
        plt.margins(x=0)
        ax.plot(np.arange(1400, 1500), ret_n_trials[1400:1500], c='#666666', linewidth=2)
        # plt.ylabel('Performance')
        # plt.xlabel('Trial number')
        plt.yticks([1, 2], ['', ''])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.xticks([1400, 1500], ['', ''])
        plt.savefig('figures/fig_2c_righttop.pdf', dpi=500, bbox_inches='tight', format='pdf')
        plt.show()
