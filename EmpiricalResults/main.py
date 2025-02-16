import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd

import gym
from gym.spaces import Discrete, Tuple, MultiDiscrete
import math
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm

from scipy import stats

def compute_mean_and_ci(data, confidence=0.95):
    """
    Compute the mean and the confidence interval for a numpy array of floats.

    Parameters:
    - data (numpy array): The input data.
    - confidence (float): The desired confidence level (e.g., 0.95 for a 95% CI).

    Returns:
    - mean (float): The mean of the data.
    - (ci_lower, ci_upper) (tuple): The lower and upper bounds of the confidence interval.
    """

    # Compute mean
    mean = np.mean(data)

    # Compute sample standard deviation
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation

    # Sample size
    n = len(data)

    # Compute the standard error
    se = std_dev / np.sqrt(n)

    # Get t-value for the desired confidence and n-1 degrees of freedom
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha/2, n-1)

    # Compute the margin of error
    margin_error = t_value * se

    # Compute confidence intervals
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error

    return mean, (ci_lower, ci_upper)



def get_args():
    parser = argparse.ArgumentParser(description='Glosten-Milgrom market making simulation')

    parser.add_argument('--p_ext', type=float, default=100, help='Initial true price')
    parser.add_argument('--spread', type=float, default=2, help='Initial spread')
    parser.add_argument('--mu', type=float, default=0.1, help='Mu parameter')
    
    parser.add_argument('--spread_exp', type=float, default=2, help='Spread penalty exponent')
    parser.add_argument('--max_history_len', type=int, default=20, help='History length for calculating imbalance')
    parser.add_argument('--max_episode_len', type=int, default=200000, help='Number of time slots')
    parser.add_argument('--max_episodes', type=int, default=1, help='Number of training episodes')
    parser.add_argument('--ema_base', type=int, default=-1, help='exponential moving average')
    
    parser.add_argument('--informed', type=float, default=0.9, help='Percentage of informed traders')
    parser.add_argument('--vary_informed', type=bool, default=False, help='vary the informed trader proportion')
    
    parser.add_argument('--jump_prob', type=float, default=1.0, help='Probability of price jump')
    parser.add_argument('--jump_variance', type=float, default=1.0, help='price volatilty')
    parser.add_argument('--vary_jump_prob', type=bool, default=False, help='vary the volatility')
    parser.add_argument('--jump_mode', type=str, default="linear", help='type of price jump') # linear,log,adversarial
    
    parser.add_argument('--jump_size', type=int, default=1, help='Size of price jump')
    parser.add_argument('--jump_at', type=int, default=-1, help='= -1 if no jumps, if positive, then jumps at that time by 1000*jump_size and stays constant')
    
    parser.add_argument('--use_short_term', type=bool, default=False, help='Use short-term imbalance')
    parser.add_argument('--use_endogynous', type=bool, default=False, help='Use endogenous variables')
    parser.add_argument('--n_price_adjustments', type=int, default=3, help='Number of actions to adjust mid price')
    parser.add_argument('--adjust_mid_spread', type=bool, default=False, help='Adjust mid + spread')
    parser.add_argument('--fixed_spread', type=bool, default=False, help='Fix the spread')
    
    parser.add_argument('--use_stored_path', type=bool, default=False, help='Use a generated sample path again')
    
    parser.add_argument('--compare', type=bool, default=False, help='Compare with !use_endogynous')
    parser.add_argument('--compare_with_bayes', type=bool, default=False, help='Compare with bayesian agent')
    
    parser.add_argument('--state_is_vec', type=bool, default=False, help='is state a vector')
    
    parser.add_argument('--special_string', type=str, default=None, help='Special string for output folder')
    
    parser.add_argument('--model_transfer', type=bool, default=False, help='Reuse the same agent')
    
    parser.add_argument('--agent_type', type=str, default="QT", help='RL agent type (QT, DQN, SARSA)')
    parser.add_argument('--AMM_type', type=str, default="mean", help='AMM type (mean,sum)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount rate of future rewards')
    parser.add_argument('--epsilon', type=float, default=0.9999, help='Starting exploration probability')
    
    parser.add_argument('--mode', type=str, default="valid", help='Mode for moving average calculation')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='checkpoint model after training iterations')
    
    parser.add_argument('--noise_type', type=str, default="Gaussian", help='Type of noise in trader price belief')
    # CHOICES FOR NOISE TYPE : "Bernoulli", "Gaussian", "Laplacian", "GeomGaussian"
    parser.add_argument('--noise_mean', type=float, default=0.0, help='mean of the noise')
    parser.add_argument('--noise_variance', type=float, default=1, help='variance of the noise')

    parser.add_argument('--window_cap', type=int, default=None, help='adaptive kalan window')
    parser.add_argument('--vary_jump_size', type=float, default=0.01, help='jump size of parameter which varies')

    args = parser.parse_args()
    return args


args = get_args()

# Define the initial true price, initial spread
p_ext = 10000
spread = 2

# Define other env params
mu = 18
spread_exp = 1  # spread penalty = - mu * (spread)^(spread_exp)

max_history_len = 21 # history over which imbalance calculated
max_episode_len = args.max_episode_len # number of time slots
max_episodes = args.max_episodes # number of episodes
average_over_episodes = True # Plot average over episodes?
start_average_from = 0
ema_base = -1 #0.97 # exponential moving average - set to -1 if want to use moving window instead

informed = args.informed # ALPHA : percentage of informed traders
vary_informed = args.vary_informed


jump_prob = args.jump_prob # SIGMA : probability of price jump for fixed trade size model
jump_variance = args.jump_variance # SIGMA : variance of jump for variable trade size model
vary_jump_prob = args.vary_jump_prob
jump_size = 1
jump_at = -100000 # = -1 if no jumps, if positive, then jumps at that time by 1000*jump_size and stays constant
jump_mode = args.jump_mode

use_short_term = True # use short term imbalance ?
use_endogynous = True # use endogynous variables or exogenous

n_price_adjustments = 3 # number of actions of agent to increase/decrease mid price

adjust_mid_spread = True # adjust mid + spread or adjust bid, ask separately
fixed_spread = False # fix the spread?
use_stored_path = True # use a stored sample path again?

# Compare with !use_endogynous ?
compare = False

special_string = args.special_string
model_transfer = False # set to True if you are reusing the same agent - modify the special string above to indicate that

# Define type of the RL agent 
agent_type = args.agent_type # BA = Bayesian, QT = q learning using table, AKF = AdaptiveKalmanFilter, UBA = Unknown bayesian - variational inference, QUCB = q learning with UCB and table, DQN = deep q network, SARSA = sarsa dq
AMM_type = args.AMM_type
state_is_vec = False
# Define agent params
alpha = 0.06 # learning rate
gamma = 0.99 # discount rate of future rewards
epsilon = 0.99 # probability of exploration vs exploitation - decays over time, this is only the starting epsilon
c = 0.001 # UCB factor
if model_transfer:
    epsilon = 0.1
    
# Define plot params
moving_avg = int(max_episode_len/200)
if moving_avg < 10:
    moving_avg = 1
mode="valid"
moving_avg = 1

# General trader models
noise_type = args.noise_type # "Bernoulli", "Gaussian", "Laplacian", "GeomGaussian"
noise_mean = 0
noise_variance = args.noise_variance

# Variable trade size parameters
do_variable_trade_size = True
normal_AMM = False # If true then bayesian amm is static and not dynamic. If false then behaves like a bayesian AMM
if do_variable_trade_size:
    compare_with_normal = True
    compare_with_bayes = True
max_trade_size = 1
n_slippage_adjustments = 3
slippage = 0.1
if do_variable_trade_size:
    epsilon = 1e-2
    if jump_mode == "log":
        jump_variance *= 0.001
        noise_variance *= 0.001
    if jump_mode == "adversarial":
        informed /= 100

if args.window_cap < 0:
    args.window_cap=None

vary_jump_size = 0.001*args.vary_jump_size

print("ALPHA = {0}, SIGMA = {1}".format(informed,jump_prob))
print("ETA = {0}, SIGMA = {1}".format(noise_variance,jump_variance))

from env import GlostenMilgromEnv, bayesian_demand_curve, discretized_gaussian
from agent import DQN_Agent, QLearningAgent, QLearningAgentUpperConf, BayesianAgent, PPOKalmanFilter, UnknownBayesianAgent, AdaptiveKalmanFilter, RobustKalmanFilter, save_checkpoint, load_checkpoint

# Create the environment
env = GlostenMilgromEnv(
    p_ext, 
    spread, 
    mu, 
    jump_prob=jump_prob,
    jump_variance=jump_variance,
    jump_mode=jump_mode,
    informed=informed, 
    max_episode_len=max_episode_len, 
    max_history_len=max_history_len,
    use_short_term=use_short_term,
    use_endogynous=use_endogynous,
    n_price_adjustments=n_price_adjustments,
    adjust_mid_spread=adjust_mid_spread,
    fixed_spread=fixed_spread,
    use_stored_path=use_stored_path,
    spread_exp=spread_exp,
    jump_size=jump_size,
    vary_informed=vary_informed,
    vary_jump_prob=vary_jump_prob,
    ema_base=ema_base,
    compare_with_bayes = compare_with_bayes,
    jump_at=jump_at,
    noise_type=noise_type,
    noise_variance=noise_variance,
    n_slippage_adjustments=n_slippage_adjustments,
    max_trade_size=max_trade_size,
    slippage=slippage,
    normal_AMM=normal_AMM,
    AMM_type=AMM_type,
    vary_jump_size=vary_jump_size,
)


env_compare = GlostenMilgromEnv(
    p_ext, 
    spread, 
    mu, 
    jump_prob=jump_prob,
    jump_variance=jump_variance,
    jump_mode=jump_mode,
    informed=informed, 
    max_episode_len=max_episode_len, 
    max_history_len=max_history_len,
    use_short_term=use_short_term,
    use_endogynous=not use_endogynous,
    n_price_adjustments=n_price_adjustments,
    adjust_mid_spread=adjust_mid_spread,
    fixed_spread=fixed_spread,
    use_stored_path=use_stored_path,
    spread_exp=spread_exp,
    jump_size=jump_size,
    vary_informed=vary_informed,
    vary_jump_prob=vary_jump_prob,
    ema_base=ema_base,
    compare_with_bayes = compare_with_bayes,
    jump_at=jump_at,
    noise_type=noise_type,
    noise_variance=noise_variance,
    n_slippage_adjustments=n_slippage_adjustments,
    max_trade_size=max_trade_size,
    slippage=slippage,
    normal_AMM=True,
    AMM_type=AMM_type,
    vary_jump_size=vary_jump_size,
)

env_bayes = GlostenMilgromEnv(
    p_ext, 
    spread, 
    mu, 
    jump_prob=jump_prob,
    jump_variance=jump_variance,
    jump_mode=jump_mode,
    informed=informed, 
    max_episode_len=max_episode_len, 
    max_history_len=max_history_len,
    use_short_term=use_short_term,
    use_endogynous=use_endogynous,
    n_price_adjustments=n_price_adjustments,
    adjust_mid_spread=adjust_mid_spread,
    fixed_spread=fixed_spread,
    use_stored_path=use_stored_path,
    spread_exp=spread_exp,
    jump_size=jump_size,
    vary_informed=vary_informed,
    vary_jump_prob=vary_jump_prob,
    ema_base=ema_base,
    compare_with_bayes = compare_with_bayes,
    jump_at=jump_at,
    noise_type=noise_type,
    noise_variance=noise_variance,
    n_slippage_adjustments=n_slippage_adjustments,
    max_trade_size=max_trade_size,
    slippage=slippage,
    normal_AMM=False,
    AMM_type=AMM_type,
    vary_jump_size=vary_jump_size,
)
# Create the agent

if model_transfer:
    if agent_type == "DQN" or agent_type == "PPO":
        state_is_vec = True
else:
    if agent_type == "QT": # tabular q learning with epsilon exploration
        n_states = 2*max_history_len + 1  # Define the number of discrete states for the given history window
        if do_variable_trade_size:
            agent = QLearningAgent(
                n_actions=[env.action_space[0].n,env.action_space[1].n,env.n_slippage_adjustments], 
                n_states=n_states, 
                alpha=alpha, 
                gamma=gamma, 
                epsilon=epsilon,
                variable_trade_size=True
            )
            comparison_agent = QLearningAgent(
                n_actions=[env.action_space[0].n,env.action_space[1].n,env.n_slippage_adjustments], 
                n_states=n_states, 
                alpha=alpha, 
                gamma=gamma, 
                epsilon=epsilon,
                variable_trade_size=True
            )
        else:
            agent = QLearningAgent(
                n_actions=[env.action_space[0].n,env.action_space[1].n], 
                n_states=n_states, 
                alpha=alpha, 
                gamma=gamma, 
                epsilon=epsilon
            )
            comparison_agent = QLearningAgent(
                n_actions=[env.action_space[0].n,env.action_space[1].n], 
                n_states=n_states, 
                alpha=alpha, 
                gamma=gamma, 
                epsilon=epsilon
            )
    elif agent_type == "QUCB": # tabular q learning + ucb exploration
        n_states = 2*max_history_len + 1  # Define the number of discrete states for the given history window
        agent = QLearningAgentUpperConf(
            n_actions=[env.action_space[0].n,env.action_space[1].n], 
            n_states=n_states, 
            alpha=alpha, 
            gamma=gamma, 
            c=c
        )
        comparison_agent = QLearningAgentUpperConf(
            n_actions=[env.action_space[0].n,env.action_space[1].n], 
            n_states=n_states, 
            alpha=alpha, 
            gamma=gamma, 
            c=c
        )
    elif agent_type == "DQN":
        state_dim = 1
        state_is_vec = True
        agent = DQN_Agent(
            max_history_len,
            n_price_adjustments,
            num_adjustments=n_price_adjustments,
            window=max_history_len,
            hidden_size=64,
            lr=1e-3,
            gamma=gamma,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )
    elif agent_type == "SARSA":
        state_dim = 1
        agent = SARSA_agent(
            n_actions=[env.action_space[0].n,env.action_space[1].n],
            state_dim=state_dim,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
    elif agent_type == "UCRL":
        pass
    elif agent_type == "TD":
        pass
    elif agent_type == "AI":
        pass
    elif agent_type == "BA":
        n_states = 2*max_history_len + 1
        agent = BayesianAgent(
            n_actions=[env.action_space[0].n,env.action_space[1].n], 
            n_states=n_states, 
            alpha=alpha, 
            gamma=gamma, 
            epsilon=epsilon
        )
    elif agent_type == "UBA":
        n_states = 2*max_history_len + 1
        agent = UnknownBayesianAgent(
            n_actions=[env.action_space[0].n,env.action_space[1].n], 
            n_states=n_states, 
            alpha=alpha, 
            gamma=gamma, 
            epsilon=epsilon
        )
    elif agent_type == "AKF":
        agent = AdaptiveKalmanFilter(
                initial_price=env.p_ext,
                epsilon=epsilon,
                jump_mode=jump_mode,
                window_cap=args.window_cap
            )
    elif agent_type == "RKF":
        agent = RobustKalmanFilter(
                initial_price=env.p_ext,
                epsilon=epsilon,
                jump_mode=jump_mode,
                eta=noise_variance,
                sigma=jump_variance
            )
    elif agent_type == "PPOKF":
        agent = PPOKalmanFilter(
                initial_price=env.p_ext,
                epsilon=0.2
            )
    else:
        print("ERROR_UNKNOWN_AGENT_TYPE")

if compare_with_bayes:
    n_states_bayes = 2*max_history_len + 1
    bayesian_agent = BayesianAgent(
        n_actions=[env.action_space[0].n,env.action_space[1].n], 
        n_states=n_states_bayes, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon
    )

if compare_with_normal:
    n_states_bayes = 2*max_history_len + 1
    comparison_agent = BayesianAgent(
        n_actions=[env.action_space[0].n,env.action_space[1].n], 
        n_states=n_states_bayes, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon
    )

import os

# Train the agent for some number of episodes - ideally should only need one episode for training the network
# output detailed plots for the last episode

total_rewards = []
monetary_losses = []

total_rewards_compare = []
monetary_losses_compare = []



rewards_vs_time = [0.0]
monetary_losses_vs_time = [0.0]
spread_vs_time = []
ask_vs_time = []
bid_vs_time = []
mid_price_vs_time = []

rewards_vs_time_compare = [0.0]
monetary_losses_vs_time_compare = [0.0]
spread_vs_time_compare = []
ask_vs_time_compare = []
bid_vs_time_compare = []
mid_price_vs_time_compare = []

total_rewards_bayes = []
monetary_losses_bayes = []
rewards_vs_time_bayes = [0.0]
monetary_losses_vs_time_bayes = [0.0]
spread_vs_time_bayes = []
ask_vs_time_bayes = []
bid_vs_time_bayes = []
mid_price_vs_time_bayes = []


p_ext_vs_time = []
price_of_ask_over_time = []

def normalize_trade_imbalance(imbalance, n_states):
    return min(max(int(imbalance + n_states // 2), 0), n_states - 1)


if not state_is_vec:
    n_states = agent.n_states

    
for episode in tqdm(range(max_episodes)):
    
    env.reset()
    env.resetAllVars(reset_stored_path=True)
    
    env_compare.reset()
    env_compare.resetAllVars(reset_stored_path=False)
    
    env_bayes.reset()
    env_bayes.resetAllVars(reset_stored_path=False)
    
    env_compare.price_path = env.price_path
    env_bayes.price_path = env.price_path
    
    env_compare.trader_price_path = env.trader_price_path
    env_bayes.trader_price_path = env.trader_price_path
    
    env_compare.price_path_variable = env.price_path_variable
    env_bayes.price_path_variable = env.price_path_variable
    
    env_compare.trader_price_path_variable = env.trader_price_path_variable
    env_bayes.trader_price_path_variable = env.trader_price_path_variable
    
    if not state_is_vec:
        state = normalize_trade_imbalance(env.imbalance, n_states)
        state_compare = normalize_trade_imbalance(env_compare.imbalance, n_states)
    else:
        state = torch.zeros(1,max_history_len)
        state_compare = torch.zeros(1,max_history_len)

    if do_variable_trade_size:
        agent.reset()
    
    state_bayes = normalize_trade_imbalance(env_bayes.imbalance, n_states)
    
    done = False
    
    total_reward = 0
    total_reward_compare = 0
    total_reward_bayes = 0
    
    time = 0

    for i in tqdm(range(max_episode_len)):
        action = agent.choose_action(state,epsilon=agent.epsilon**time)
        
        if do_variable_trade_size:
            (next_trade_history, next_imbalance, next_trader_price), reward, done, extra_dict = env.step_variable(action)
        else:
            (next_trade_history, next_imbalance), reward, done, extra_dict = env.step(action)

        if i==max_episode_len-1:
            done=True
        
        if state_is_vec:
            next_state = torch.tensor(next_trade_history).permute((1,0)).float()
            agent.update(state, action, reward, next_state, done=done)
        else:
            if do_variable_trade_size:
                next_state = next_trader_price
            else:
                next_state = normalize_trade_imbalance(next_imbalance, n_states)
            agent.update(state, action, reward, next_state, done=done)

        state = next_state
        total_reward += reward
        time += 1
        
        if not average_over_episodes:
            if episode == max_episodes-1:
                p_ext_vs_time.append(extra_dict["p_ext"])
                rewards_vs_time.append(reward)
                monetary_losses_vs_time.append(extra_dict["monetary_loss"])
                spread_vs_time.append(extra_dict["spread"])
                ask_vs_time.append(extra_dict["ask"])
                bid_vs_time.append(extra_dict["bid"])
                mid_price_vs_time.append(extra_dict["mid"])
                #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
        else:
            if episode == start_average_from:
                p_ext_vs_time.append(extra_dict["p_ext"])
                rewards_vs_time.append(reward)
                monetary_losses_vs_time.append(extra_dict["monetary_loss"])
                spread_vs_time.append(extra_dict["spread"])
                ask_vs_time.append(extra_dict["ask"])
                bid_vs_time.append(extra_dict["bid"])
                mid_price_vs_time.append(extra_dict["mid"])
                #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
            elif episode > start_average_from:
                p_ext_vs_time[time-1] += (extra_dict["p_ext"])
                rewards_vs_time[time-1] +=(reward)
                monetary_losses_vs_time[time-1] +=(extra_dict["monetary_loss"])
                spread_vs_time[time-1] +=(extra_dict["spread"])
                ask_vs_time[time-1] +=(extra_dict["ask"])
                bid_vs_time[time-1] +=(extra_dict["bid"])
                mid_price_vs_time[time-1] +=(extra_dict["mid"])
            if episode == max_episodes-1:
                p_ext_vs_time[time-1] /= max_episodes
                rewards_vs_time[time-1] /= max_episodes
                monetary_losses_vs_time[time-1] /= max_episodes
                spread_vs_time[time-1] /= max_episodes
                ask_vs_time[time-1] /= max_episodes
                bid_vs_time[time-1] /= max_episodes
                mid_price_vs_time[time-1] /= max_episodes
    
        if compare or compare_with_normal:
            action = comparison_agent.choose_action(state_compare,epsilon=epsilon**time)

            if do_variable_trade_size:
                (next_trade_history, next_imbalance, next_trader_price), reward, done, extra_dict = env_compare.step_variable(action)
            else:
                (next_trade_history, next_imbalance), reward, done, extra_dict = env_compare.step(action)

            next_state_compare = normalize_trade_imbalance(next_imbalance, n_states)

            comparison_agent.update(state_compare, action, reward, next_state_compare)

            state_compare = next_state_compare
            total_reward_compare += reward

            if not average_over_episodes:
                if episode == max_episodes-1:
                    #p_ext_vs_time_compare.append(extra_dict["p_ext"])
                    rewards_vs_time_compare.append(reward)
                    monetary_losses_vs_time_compare.append(extra_dict["monetary_loss"])
                    spread_vs_time_compare.append(extra_dict["spread"])
                    ask_vs_time_compare.append(extra_dict["ask"])
                    bid_vs_time_compare.append(extra_dict["bid"])
                    mid_price_vs_time_compare.append(extra_dict["mid"])
                    #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
            else:
                if episode == start_average_from:
                    #p_ext_vs_time_compare.append(extra_dict["p_ext"])
                    rewards_vs_time_compare.append(reward)
                    monetary_losses_vs_time_compare.append(extra_dict["monetary_loss"])
                    spread_vs_time_compare.append(extra_dict["spread"])
                    ask_vs_time_compare.append(extra_dict["ask"])
                    bid_vs_time_compare.append(extra_dict["bid"])
                    mid_price_vs_time_compare.append(extra_dict["mid"])
                    #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
                elif episode > start_average_from:
                    #p_ext_vs_time_compare[time-1] += (extra_dict["p_ext"])
                    rewards_vs_time_compare[time-1] +=(reward)
                    monetary_losses_vs_time_compare[time-1] +=(extra_dict["monetary_loss"])
                    spread_vs_time_compare[time-1] +=(extra_dict["spread"])
                    ask_vs_time_compare[time-1] +=(extra_dict["ask"])
                    bid_vs_time_compare[time-1] +=(extra_dict["bid"])
                    mid_price_vs_time_compare[time-1] +=(extra_dict["mid"])
                if episode == max_episodes-1:
                    #p_ext_vs_time_compare[time-1] /= max_episodes
                    rewards_vs_time_compare[time-1] /= max_episodes
                    monetary_losses_vs_time_compare[time-1] /= max_episodes
                    spread_vs_time_compare[time-1] /= max_episodes
                    ask_vs_time_compare[time-1] /= max_episodes
                    bid_vs_time_compare[time-1] /= max_episodes
                    mid_price_vs_time_compare[time-1] /= max_episodes
        if compare_with_bayes :

            action = bayesian_agent.choose_action(state_bayes,epsilon=epsilon**time)
            
            if do_variable_trade_size:
                (next_trade_history, next_imbalance, next_trader_price), reward, done, extra_dict = env_bayes.step_variable(action)
            else:
                (next_trade_history, next_imbalance), reward, done, extra_dict = env_bayes.step(action)

            next_state_bayes = normalize_trade_imbalance(next_imbalance, n_states)

            bayesian_agent.update(state_bayes, action, reward, next_state_bayes)

            state_bayes = next_state_bayes
            total_reward_bayes += reward

            if not average_over_episodes:
                if episode == max_episodes-1:
                    #p_ext_vs_time_bayes.append(extra_dict["p_ext"])
                    rewards_vs_time_bayes.append(reward)
                    monetary_losses_vs_time_bayes.append(extra_dict["monetary_loss"])
                    spread_vs_time_bayes.append(extra_dict["spread"])
                    ask_vs_time_bayes.append(extra_dict["ask"])
                    bid_vs_time_bayes.append(extra_dict["bid"])
                    mid_price_vs_time_bayes.append(extra_dict["mid"])
                    #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
            else:
                if episode == start_average_from:
                    #p_ext_vs_time_bayes.append(extra_dict["p_ext"])
                    rewards_vs_time_bayes.append(reward)
                    monetary_losses_vs_time_bayes.append(extra_dict["monetary_loss"])
                    spread_vs_time_bayes.append(extra_dict["spread"])
                    ask_vs_time_bayes.append(extra_dict["ask"])
                    bid_vs_time_bayes.append(extra_dict["bid"])
                    mid_price_vs_time_bayes.append(extra_dict["mid"])
                    #print(time," : {0},{1}".format(extra_dict["ask"],extra_dict["bid"]))
                elif episode > start_average_from:
                    #p_ext_vs_time_bayes[time-1] += (extra_dict["p_ext"])
                    rewards_vs_time_bayes[time-1] +=(reward)
                    monetary_losses_vs_time_bayes[time-1] +=(extra_dict["monetary_loss"])
                    spread_vs_time_bayes[time-1] +=(extra_dict["spread"])
                    ask_vs_time_bayes[time-1] +=(extra_dict["ask"])
                    bid_vs_time_bayes[time-1] +=(extra_dict["bid"])
                    mid_price_vs_time_bayes[time-1] +=(extra_dict["mid"])
                if episode == max_episodes-1:
                    #p_ext_vs_time_bayes[time-1] /= max_episodes
                    rewards_vs_time_bayes[time-1] /= max_episodes
                    monetary_losses_vs_time_bayes[time-1] /= max_episodes
                    spread_vs_time_bayes[time-1] /= max_episodes
                    ask_vs_time_bayes[time-1] /= max_episodes
                    bid_vs_time_bayes[time-1] /= max_episodes
                    mid_price_vs_time_bayes[time-1] /= max_episodes
                        
    total_rewards.append(total_reward)
    monetary_losses.append(env.cumulative_monetary_loss)
    if compare or compare_with_normal:
        total_rewards_compare.append(total_reward_compare)
        monetary_losses_compare.append(env_compare.cumulative_monetary_loss)
    if compare_with_bayes:
        total_rewards_bayes.append(total_reward_bayes)
        monetary_losses_bayes.append(env_bayes.cumulative_monetary_loss)





#______________________________________________________________________________________________________________________________#

if noise_type == "Bernoulli":
    figure_path = "modelFreeGM/informed_{0}_jump_{1}_mu_{2}/fixedSpread_{10}_useShortTerm_{3}_useEndo_{4}_maxHistoryLen_{5}/agentType_{6}_alpha_{7}_gamma_{8}_epsilon_{9}".format(
        informed,
        jump_prob,
        mu,
        use_short_term,
        use_endogynous,
        max_history_len,
        agent_type,
        alpha,
        gamma,
        epsilon,
        fixed_spread
    )
else:
    if do_variable_trade_size:
        figure_path = "variableTradeFigures/{0}_{10}_jump_{1}_mu_{2}/fixedSpread_{10}_useShortTerm_{3}_useEndo_{4}_maxHistoryLen_{5}/agentType_{6}_alpha_{7}_gamma_{8}_epsilon_{9}".format(
            noise_type,
            jump_prob,
            mu,
            use_short_term,
            use_endogynous,
            max_history_len,
            agent_type,
            alpha,
            gamma,
            epsilon,
            fixed_spread,
            noise_variance
        )
    else:
        figure_path = "modelFreeGM/{0}_{10}_jump_{1}_mu_{2}/fixedSpread_{10}_useShortTerm_{3}_useEndo_{4}_maxHistoryLen_{5}/agentType_{6}_alpha_{7}_gamma_{8}_epsilon_{9}".format(
            noise_type,
            jump_prob,
            mu,
            use_short_term,
            use_endogynous,
            max_history_len,
            agent_type,
            alpha,
            gamma,
            epsilon,
            fixed_spread,
            noise_variance
        )

    
if special_string is not None:
    figure_path = figure_path + "/{0}".format(special_string)

if ema_base != -1:
    figure_path = figure_path + "/ema_base_{0}".format(ema_base)
    
if not adjust_mid_spread:
    figure_path = figure_path + "/direct_ask_bid_control"

os.makedirs(figure_path , exist_ok=True)
print("FIGURES STORED IN {0}".format(figure_path))

# Calculate the average total reward and monetary loss
average_total_reward = np.mean(total_rewards)
average_monetary_loss = np.mean(monetary_losses)
print("Average total reward main:", average_total_reward/max_episode_len)
print("Average monetary loss main :", average_monetary_loss/max_episode_len)
print("Mean spread main",np.mean(np.array(spread_vs_time)))
print("Mean mid dev main",np.sqrt(np.mean((np.array(mid_price_vs_time)-np.array(p_ext_vs_time))**2)))
print(" ")
if compare or compare_with_normal:
    average_total_reward_compare = np.mean(total_rewards_compare)
    average_monetary_loss_compare = np.mean(monetary_losses_compare)
    print("Average total reward compare:", average_total_reward_compare/max_episode_len)
    print("Average monetary loss compare:", average_monetary_loss_compare/max_episode_len)
    print("Mean spread compare",np.mean(np.array(spread_vs_time_compare)))
    print("Mean mid dev compare",np.sqrt(np.mean((np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time))**2)))
    print(" ")
if compare_with_bayes:
    average_total_reward_bayes = np.mean(total_rewards_bayes)
    average_monetary_loss_bayes = np.mean(monetary_losses_bayes)
    print("Average total reward reference:", average_total_reward_bayes/max_episode_len)
    print("Average monetary loss reference:", average_monetary_loss_bayes/max_episode_len)
    print("Mean spread reference",np.mean(np.array(spread_vs_time_bayes)))
    print("Mean mid dev reference",np.sqrt(np.mean((np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time))**2)))
    print(" ")

import csv

if compare_with_bayes or compare_with_normal:
    if do_variable_trade_size:
        # Always included metrics
        base_row = [
            env.noise_variance,
            env.jump_variance,
            " ",
            np.mean(np.array(monetary_losses)[start_average_from:]/max_episode_len),
            np.median(np.array(monetary_losses)[start_average_from:]/max_episode_len),
            np.mean(np.array(spread_vs_time)),
            np.median(np.array(spread_vs_time)),
            np.mean(abs(np.array(mid_price_vs_time)-np.array(p_ext_vs_time))),
            np.median(abs(np.array(mid_price_vs_time)-np.array(p_ext_vs_time))),
            np.sqrt(np.mean((np.array(mid_price_vs_time)-np.array(p_ext_vs_time))**2)),
            " "
        ]

        # Metrics calculated from variables with _compare
        compare_metrics = [
            np.mean(np.array(monetary_losses_compare)[start_average_from:]/max_episode_len),
            np.median(np.array(monetary_losses_compare)[start_average_from:]/max_episode_len),
            np.mean(np.array(spread_vs_time_compare)),
            np.median(np.array(spread_vs_time_compare)),
            np.mean(abs(np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time))),
            np.median(abs(np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time))),
            np.sqrt(np.mean((np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time))**2)),
            " "
        ] if compare_with_normal else []

        # Metrics calculated from variables with _bayes
        bayes_metrics = [
            np.mean(np.array(monetary_losses_bayes)[start_average_from:]/max_episode_len),
            np.median(np.array(monetary_losses_bayes)[start_average_from:]/max_episode_len),
            np.mean(np.array(spread_vs_time_bayes)),
            np.median(np.array(spread_vs_time_bayes)),
            np.mean(abs(np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time))),
            np.median(abs(np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time))),
            np.sqrt(np.mean((np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time))**2)),
            " "
        ] if compare_with_bayes else []

        # Combine the rows and write to CSV
        file_name = 'losses_and_spreads_variable_trade_size'
        if special_string is not None:
            file_name = file_name + "{0}".format(special_string)

        file_name = file_name + ".csv"
        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(base_row + compare_metrics + bayes_metrics)
    else:
        with open('losses_and_spreads_modified.csv', 'a', newline='') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow([
            env.informed,
            env.jump_prob,
            " ",
            np.mean(np.array(monetary_losses)[start_average_from:]/max_episode_len),
            np.median(np.array(monetary_losses)[start_average_from:]/max_episode_len),
            np.mean(np.array(spread_vs_time)),
            np.median(np.array(spread_vs_time)),
            np.mean(abs(np.array(mid_price_vs_time)-np.array(p_ext_vs_time))),
            np.median(abs(np.array(mid_price_vs_time)-np.array(p_ext_vs_time))),
            " ",
            np.mean(np.array(monetary_losses_compare)[start_average_from:]/max_episode_len),
            np.median(np.array(monetary_losses_compare)[start_average_from:]/max_episode_len),
            np.mean(np.array(spread_vs_time_compare)),
            np.median(np.array(spread_vs_time_compare)),
            np.mean(abs(np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time))),
            np.median(abs(np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time))),
            " ",
            np.mean(np.array(monetary_losses_bayes)[start_average_from:]/max_episode_len),
            np.median(np.array(monetary_losses_bayes)[start_average_from:]/max_episode_len),
            np.mean(np.array(spread_vs_time_bayes)),
            np.median(np.array(spread_vs_time_bayes)),
            np.mean(abs(np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time))),
            np.median(abs(np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time))),
            ])

# Plot the average monetary loss over all episodes
plt.plot(monetary_losses)
if compare or compare_with_normal:
    plt.plot(monetary_losses_compare)
if compare_with_bayes:
    plt.plot(monetary_losses_bayes)
plt.xlabel("Episode Number")
plt.ylabel("Average Monetary Loss vs time")
plt.title("Average Monetary Loss over Episodes")
plt.legend()
filename = "Monetary_Loss.pdf"
file_path = os.path.join(figure_path, filename)
plt.savefig(file_path)  
plt.close()
 

# Plot the average total reward over all episodes
plt.plot(total_rewards)
if compare or compare_with_normal:
    plt.plot(total_rewards_compare)
if compare_with_bayes:
    plt.plot(total_rewards_bayes)
plt.xlabel("Episode Number")
plt.ylabel("Average total reward")
plt.title("Average total reward over Episodes")
plt.legend()
filename = "Total_Reward.pdf"
file_path = os.path.join(figure_path, filename)
plt.savefig(file_path)
plt.close()



# Plot the monetary loss over time for the last episode
plt.plot(np.convolve(np.array(monetary_losses_vs_time),np.ones(moving_avg)/moving_avg,mode=mode),label="Main")
if compare or compare_with_normal:
    plt.plot(np.convolve(np.array(monetary_losses_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")
if compare_with_bayes:
    plt.plot(np.convolve(np.array(monetary_losses_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode),label="Reference")
plt.xlabel("Time")
plt.ylabel("Monetary Loss")
plt.title("Monetary Loss over time")
plt.legend()
filename = "Loss_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
plt.savefig(file_path)
plt.close()


# Plot the reward over time for the last episode
plt.plot(np.convolve(np.array(rewards_vs_time),np.ones(moving_avg)/moving_avg,mode=mode),label="orig")
if compare:
    plt.plot(np.convolve(np.array(rewards_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")
plt.xlabel("Time")
plt.ylabel("Reward")#
plt.title("Reward over time")
plt.legend()
filename = "Reward_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
plt.savefig(file_path)
plt.close()


# Plot the spread over time for the last episode
fig, ax = plt.subplots()

ax.plot(np.convolve(np.array(spread_vs_time),np.ones(moving_avg)/moving_avg,mode=mode),label="Q-learning")# spread should decay with time
# if compare:
#     plt.plot(np.convolve(np.array(spread_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")# spread should decay with time
if compare_with_bayes:
    ax.plot(np.convolve(np.array(spread_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode),label="Bayesian")# spread should decay with time
if compare or compare_with_normal:
    ax.plot(np.convolve(np.array(spread_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="Bayesian")# spread should decay with time

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Spread', fontsize=14)
ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
#ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing

print("Spread over time : one sample path")
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

filename = "Spread_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
fig.savefig(file_path, format='pdf', bbox_inches='tight')
plt.close()

# Plot the spread distribution for the last episode
pd.Series(spread_vs_time).hist()
# if compare:
#     plt.plot(np.convolve(np.array(spread_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode),label="compare")# spread should decay with time
# if compare_with_bayes:
#     plt.plot(np.convolve(np.array(spread_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode),label="Bayes")# spread should decay with time
# plt.xlabel("Time")
# plt.ylabel("Spread")
# plt.title("Spread over time : one sample path")
# plt.legend()
# filename = "Spread_Vs_time.pdf"
# file_path = os.path.join(figure_path, filename)
# plt.savefig(file_path)
plt.close()

# Plot the ask,bid and external price over time for the last episode
fig, ax = plt.subplots()

ax.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{ext}$")
ax.plot(np.convolve(np.array(ask_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{ask}$")
ax.plot(np.convolve(np.array(bid_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{bid}$")

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Price', fontsize=14)

ax.legend()

ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
#ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing

ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

print("Q-learning Ask and Bid over time : one sample path")  # Adjust the title and fontsize as required

filename = "AskBid_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
fig.savefig(file_path, format='pdf', bbox_inches='tight')

plt.close()

if compare or compare_with_normal:
    plt.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="P_ext")
    plt.plot(np.convolve(np.array(ask_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode), label="Ask")
    plt.plot(np.convolve(np.array(bid_vs_time_compare),np.ones(moving_avg)/moving_avg,mode=mode), label="Bid")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Compare Ask and Bid over time : one sample path (reward has p_ext)")
    plt.legend()
    filename = "AskBid_Vs_time_compare.pdf"
    file_path = os.path.join(figure_path, filename)
    plt.savefig(file_path)
    plt.close()
if compare_with_bayes:
    fig, ax = plt.subplots()

    ax.plot(np.convolve(np.array(p_ext_vs_time),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{ext}$")
    ax.plot(np.convolve(np.array(ask_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{ask}$")
    ax.plot(np.convolve(np.array(bid_vs_time_bayes),np.ones(moving_avg)/moving_avg,mode=mode), label="$p_{bid}$")
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    
    #ax.title("Reference Ask and Bid over time : one sample path")
    ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    filename = "AskBid_Vs_time_bayes.pdf"
    file_path = os.path.join(figure_path, filename)
    plt.savefig(file_path)
    plt.close()

# Plot the mid and ext price over time for the last episode
fig, ax = plt.subplots()

ax.plot(abs(np.convolve((np.array(mid_price_vs_time)-np.array(p_ext_vs_time)),np.ones(moving_avg)/moving_avg,mode=mode)), label="Q-learning")
if compare_with_bayes:
    ax.plot(abs(np.convolve((np.array(mid_price_vs_time_bayes)-np.array(p_ext_vs_time)),np.ones(moving_avg)/moving_avg,mode=mode)), label="Bayesian")# need to make sure E[(mid-p_ext)^2]-> 0 as t becomes large for the last episode
if compare:
    ax.plot(abs(np.convolve((np.array(mid_price_vs_time_compare)-np.array(p_ext_vs_time)),np.ones(moving_avg)/moving_avg,mode=mode)), label="Bayesian")# need to make sure E[(mid-p_ext)^2]-> 0 as t becomes large for the last episode

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Mid Price Deviation', fontsize=14)

ax.legend()

ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
#ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing

ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

print("mid price deviation")  # Adjust the title and fontsize as required

filename = "Mid_Vs_time.pdf"
file_path = os.path.join(figure_path, filename)
fig.savefig(file_path, format='pdf', bbox_inches='tight')

plt.close()

fig, ax = plt.subplots()
if do_variable_trade_size:
    if env.vary_jump_prob:
        ax.plot(np.convolve(np.array(env.jump_prob_path_variable),np.ones(moving_avg)/moving_avg,mode=mode), label="Price Volatility")
    if env.vary_informed:
        ax.plot(np.convolve(np.array(env.informed_path_variable),np.ones(moving_avg)/moving_avg,mode=mode), label="Trader Noise")
else:
    if env.vary_jump_prob:
        ax.plot(np.convolve(np.array(env.jump_prob_path),np.ones(moving_avg)/moving_avg,mode=mode), label="Price Volatility")
    if env.vary_informed:
        ax.plot(np.convolve(np.array(env.informed_path),np.ones(moving_avg)/moving_avg,mode=mode), label="Trader Noise")


if env.vary_informed or env.vary_jump_prob:
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Parameter Value', fontsize=14)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(50000))  # Change 1 to desired x-spacing
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Change 1 to desired y-spacing

    #plt.yscale("log")
    print("Variable volatility and trader noise")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    filename = "Params_Vs_time.pdf"
    file_path = os.path.join(figure_path, filename)
    fig.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.close()
# _______________
# spread_mean=[]
# mid_dev_mean=[]
# alphas=[]
# sigmas=[]

# spread_mean.append(np.mean(np.array(spread_vs_time)))
# mid_dev_mean.append(np.mean(abs(np.array(mid_price_vs_time))))
# alphas.append(informed)
# sigmas.append(jump_prob)
# spread_mean


