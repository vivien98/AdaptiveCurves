import gym
import numpy as np
from gym.spaces import Discrete, Tuple, MultiDiscrete
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pickle

import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys,csv,math
import numpy as np
import os

import random

def save_checkpoint(agent, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(agent, f)

def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        agent = pickle.load(f)
    return agent

class CFMM:
    """docstring for CFMM"""
    def __init__(self, reserves, amm_type, params, fees):
        super(CFMM, self).__init__()
        self.reserves = reserves
        self.type = amm_type# mean, sum
        self.fees = fees
        self.params = params

    def update(self,trader_price,trade_size=-1):
        if self.type == "mean":
            fees = self.fees
            theta = self.params[0]
            x_0 = self.reserves[0]
            y_0 = self.reserves[1]
            k = (x_0**theta) * (y_0**(1-theta))
            current_price = (theta*x_0)/((1-theta)*y_0)

            if trader_price > current_price/(1-fees):
                x_1 = k*(theta/(trader_price*(1-theta)/(1-fees)))**(1-theta)
                y_1 = k*((trader_price*(1-theta)/(1-fees))/theta)**(theta)
                x_1 = x_0 + (x_1-x_0)/(1-fees)

                self.reserves[0] = x_1
                self.reserves[1] = y_1
            elif trader_price < current_price*(1-fees):
                x_1 = k*(theta/(trader_price*(1-theta)*(1-fees)))**(1-theta)
                y_1 = k*(trader_price*(1-theta)*(1-fees)/theta)**(theta)
                y_1 = y_0 + (y_1-y_0)/(1-fees)

                self.reserves[0] = x_1
                self.reserves[1] = y_1
            else:
                x_1 = x_0
                y_1 = y_0

            return (x_1-x_0,y_1-y_0)

        elif self.type == "sum":
            fees = self.fees
            theta = self.params[0]
            x_0 = self.reserves[0]
            y_0 = self.reserves[1]
            k = x_0 * theta + y_0
            current_price = theta
            if trader_price > current_price/(1-fees):
                x_1 = 0
                y_1 = k
                x_1 = x_0 + (x_1-x_0)/(1-fees)

                self.reserves[0] = x_1
                self.reserves[1] = y_1
            elif trader_price < current_price*(1-fees):
                x_1 = k/theta
                y_1 = 0
                y_1 = y_0 + (y_1-y_0)/(1-fees)

                self.reserves[0] = x_1
                self.reserves[1] = y_1
            else:
                x_1 = x_0
                y_1 = y_0

            return (x_1-x_0,y_1-y_0)

    def modify(self,params):
        self.params = params




# class AdaptiveKalmanFilter:
#     def __init__(
#         self,
#         initial_price=1000,
#         epsilon=0.01,
#         jump_mode="linear",
#         window_cap=None
#     ):
#         self.n_actions = 0
#         self.n_states = 0
#         self.jump_mode = jump_mode

#         self.initial_price = initial_price

#         self.epsilon = epsilon  # em_algo_stop_tolerance

#         self.sigma_estimate = 1
#         self.eta_estimate = 0.9
#         self.num_observations = 1

#         self.price_estimates = np.zeros(self.num_observations)
#         if self.jump_mode == "linear":
#             self.price_estimates[0] = self.initial_price  # initial price is known exactly
#             self.trader_prices = np.array([self.initial_price])
#         elif self.jump_mode in ["log", "adversarial"]:
#             self.sigma_estimate *= math.sqrt(0.001)
#             self.eta_estimate *= math.sqrt(0.001)
#             self.price_estimates[0] = math.log(self.initial_price)  # initial price is known exactly
#             self.trader_prices = np.array([math.log(self.initial_price)])
#         self.variances = np.array([0.0])  # P_t^n variable
#         self.covariances = np.array([0.0])  # P_{t,t-1}^n variable
#         self.kalman_gains = np.array([0.0])  # K_t variable
#         self.log_likelihood = -float("inf")
#         self.error = float("inf")

#         self.fwd_variances = np.array([[0.0, 0.0]])
#         self.J = np.array([0.0])

#         self.window_cap = window_cap

#     def reset(self):
#         self.sigma_estimate = 1
#         self.eta_estimate = 0.9
#         self.num_observations = 1

#         self.price_estimates = np.zeros(self.num_observations)
#         if self.jump_mode == "linear":
#             self.price_estimates[0] = self.initial_price  # initial price is known exactly
#             self.trader_prices = np.array([self.initial_price])
#         elif self.jump_mode in ["log", "adversarial"]:
#             self.sigma_estimate *= math.sqrt(0.001)
#             self.eta_estimate *= math.sqrt(0.001)
#             self.price_estimates[0] = math.log(self.initial_price)  # initial price is known exactly
#             self.trader_prices = np.array([math.log(self.initial_price)])
#         self.variances = np.array([0.0])  # P_t^n variable
#         self.covariances = np.array([0.0])  # P_{t,t-1}^n variable
#         self.kalman_gains = np.array([0.0])  # K_t variable
#         self.log_likelihood = -float("inf")
#         self.error = float("inf")

#         self.fwd_variances = np.array([[0.0, 0.0]])
#         self.J = np.array([0.0])

    # def get_window_indices(self):
    #     if self.window_cap is None:
    #         return range(self.num_observations)
    #     return range(max(0, self.num_observations - self.window_cap), self.num_observations)

#     def forward_pass(self):
#         window_indices = list(self.get_window_indices())
#         if window_indices[0] == 0:
#             self.fwd_variances = np.array([[0.0, 0.0]])
#         else:
#             initial_variance = self.variances[window_indices[0] - 1]
#             self.fwd_variances = np.array([[initial_variance, initial_variance]])

#         self.kalman_gains = np.zeros(len(window_indices))
        # for i, t in enumerate(window_indices):
        #     if i == 0:
        #         continue
        #     else:
        #         p_ext_t_1 = self.price_estimates[window_indices[i - 1]]
        #         P_t_1_t = self.fwd_variances[i - 1, 1] + self.sigma_estimate ** 2
        #         K_t = P_t_1_t / (P_t_1_t + self.eta_estimate ** 2)
        #         p_ext_t = p_ext_t_1 + K_t * (self.trader_prices[t] - p_ext_t_1)
        #         P_t_t = (1 - K_t) * P_t_1_t
        #         self.fwd_variances = np.append(self.fwd_variances, np.array([[P_t_1_t, P_t_t]]), axis=0)
        #         self.kalman_gains[i] = K_t
        #         self.price_estimates[t] = p_ext_t

    # def backward_pass(self):
    #     window_indices = list(self.get_window_indices())
    #     reversed_window_indices = list(reversed(window_indices))
    #     for i, t in enumerate(reversed_window_indices):
    #         if i == 0:
    #             P_n_t_t_1 = (1 - self.kalman_gains[-1]) * self.fwd_variances[-1, 1]
    #             self.variances[t] = self.fwd_variances[-1, 1]
    #             self.covariances[t] = P_n_t_t_1
    #             continue
    #         else:
    #             t_next = reversed_window_indices[i - 1]
    #             J_t_1 = self.fwd_variances[-(i + 1), 1] / self.fwd_variances[-i, 0]
    #             p_ext_n_t_1 = self.price_estimates[t] + J_t_1 * (self.price_estimates[t_next] - self.price_estimates[t])
    #             P_n_t_1 = self.fwd_variances[-(i + 1), 1] + J_t_1 ** 2 * (self.variances[t_next] - self.fwd_variances[-i, 0])
    #             self.J[t] = J_t_1
    #             self.variances[t] = P_n_t_1
    #             self.price_estimates[t] = p_ext_n_t_1

    #     for t in range(self.num_observations - 2, -1, -1):
    #         if t not in window_indices:
    #             continue
    #         P_n_t_1_t_2 = self.fwd_variances[t, 1] * self.J[t - 1] + self.J[t - 1] * self.J[t] * (self.covariances[t + 1] - self.fwd_variances[t, 1])
    #         self.covariances[t] = P_n_t_1_t_2

#     def calculate_max(self):
#         window_indices = list(self.get_window_indices())
#         self.A = (
#             np.sum(self.variances[window_indices[1:]]) + np.sum(self.variances[window_indices[:-1]]) -
#             2 * np.sum(self.covariances[window_indices[1:]]) +
#             np.sum((self.price_estimates[window_indices[1:]] - self.price_estimates[window_indices[:-1]]) ** 2)
#         )
#         self.B = (
#             np.sum(self.trader_prices[window_indices[1:]] ** 2) + np.sum(self.variances[window_indices[1:]]) +
#             np.sum(self.price_estimates[window_indices[1:]] ** 2) -
#             2 * np.sum(np.multiply(self.trader_prices[window_indices[1:]], self.price_estimates[window_indices[1:]]))
#         )
#         self.eta_estimate = np.sqrt(2 * self.B / (len(window_indices) - 1))
#         self.sigma_estimate = np.sqrt(2 * self.A / (len(window_indices) - 1))

#     def calculate_error(self):
#         prev_log_likelihood = self.log_likelihood

#         self.log_likelihood = -(len(self.get_window_indices()) - 1) / 2 * np.log(self.sigma_estimate) - (len(self.get_window_indices()) - 1) / 2 * np.log(self.eta_estimate)
#         self.log_likelihood -= 1 / (2 * self.sigma_estimate ** 2) * self.A
#         self.log_likelihood -= 1 / (2 * self.eta_estimate ** 2) * self.B

#         self.error = abs(self.log_likelihood - prev_log_likelihood)

#     def choose_action(self, state, epsilon=-1):
#         p_0 = self.price_estimates[-1]
#         print(p_0)
#         if self.jump_mode == "linear":
#             return p_0
#         elif self.jump_mode in ["log", "adversarial"]:
#             return math.exp(p_0)

#     def update(self, state, action, reward, next_state, done=False):
#         if self.jump_mode == "linear":
#             trader_price = next_state
#         elif self.jump_mode in ["log", "adversarial"]:
#             trader_price = math.log(next_state)
#         self.num_observations += 1
#         self.trader_prices = np.append(self.trader_prices, trader_price)
#         self.price_estimates = np.append(self.price_estimates, trader_price)

#         self.variances = np.append(self.variances, 0.0)
#         self.covariances = np.append(self.covariances, 0.0)
#         self.J = np.append(self.J, 0.0)
#         self.kalman_gains = np.append(self.kalman_gains, 0.0)
#         self.fwd_variances = np.append(self.fwd_variances, np.array([[0.0, 0.0]]), axis=0)

        
#         while self.error > self.epsilon:
#             self.forward_pass()
#             self.backward_pass()
#             self.calculate_max()
#             self.calculate_error()

#         self.log_likelihood = -float("inf")
#         self.error = float("inf")

class AdaptiveKalmanFilter:
    def __init__(
        self,
        initial_price = 1000,
        epsilon = 0.001,
        jump_mode = "linear",
        window_cap = None
    ):
        self.n_actions = 0
        self.n_states = 0
        self.jump_mode = jump_mode

        self.initial_price = initial_price

        self.epsilon = epsilon # em_algo_stop_tolerance
        
        self.sigma_estimate = 1
        self.eta_estimate = 1
        self.num_observations = 1

        self.price_estimates = np.zeros(self.num_observations)
        if self.jump_mode == "linear":
            self.price_estimates[0] = self.initial_price # initial price is known exactly
            self.trader_prices = np.array([self.initial_price])
        elif self.jump_mode == "log" or self.jump_mode == "adversarial":
            self.sigma_estimate *= math.sqrt(0.001)
            self.eta_estimate *= math.sqrt(0.001)
            self.price_estimates[0] = math.log(self.initial_price) # initial price is known exactly
            self.trader_prices = np.array([math.log(self.initial_price)])
        self.variances = np.array([0]) # P_t^n variable
        self.covariances = np.array([0]) # P_{t,t-1}^n variable
        self.kalman_gains = np.array([0]) # K_t variable
        self.log_likelihood = -float("inf")
        self.error = float("inf")

        self.window_cap = window_cap
        
        

    def reset(self):
        self.sigma_estimate = 1
        self.eta_estimate = 1
        self.num_observations = 1

        self.price_estimates = np.zeros(self.num_observations)
        if self.jump_mode == "linear":
            self.price_estimates[0] = self.initial_price # initial price is known exactly
            self.trader_prices = np.array([self.initial_price])
        elif self.jump_mode == "log" or self.jump_mode == "adversarial":
            self.sigma_estimate *= math.sqrt(0.001)
            self.eta_estimate *= math.sqrt(0.001)
            self.price_estimates[0] = math.log(self.initial_price) # initial price is known exactly
            self.trader_prices = np.array([math.log(self.initial_price)])
        self.variances = np.array([0]) # P_t^n variable
        self.covariances = np.array([0]) # P_{t,t-1}^n variable
        self.kalman_gains = np.array([0]) # K_t variable
        self.log_likelihood = -float("inf")
        self.error = float("inf")

    def get_window_indices(self):
        if self.window_cap is None:
            return range(self.num_observations)
        return range(max(0, self.num_observations - self.window_cap), self.num_observations)


    def forward_pass(self):
        self.fwd_variances = np.array([[0,0]])
        self.kalman_gains = np.array([0]) # K_t variable
        window_indices = list(self.get_window_indices())

        for i, t in enumerate(window_indices):
            if i == 0:
                continue
            else:
                p_ext_t_1 = self.price_estimates[window_indices[i - 1]]
                P_t_1_t = self.fwd_variances[i - 1, 1] + self.sigma_estimate ** 2
                K_t = P_t_1_t / (P_t_1_t + self.eta_estimate ** 2)
                p_ext_t = p_ext_t_1 + K_t * (self.trader_prices[t] - p_ext_t_1)
                P_t_t = (1 - K_t) * P_t_1_t
                self.fwd_variances = np.append(self.fwd_variances, np.array([[P_t_1_t, P_t_t]]), axis=0)
                self.kalman_gains = np.append(self.kalman_gains,K_t)
                self.price_estimates[t] = p_ext_t

    # def backward_pass(self):
    #     window_indices = list(self.get_window_indices())
    #     reversed_window_indices = list(reversed(window_indices))
    #     for i, t in enumerate(reversed_window_indices):
    #         if i == 0:
    #             P_n_t_t_1 = (1 - self.kalman_gains[-1]) * self.fwd_variances[-1, 1]
    #             self.variances[t] = self.fwd_variances[-1, 1]
    #             self.covariances[t] = P_n_t_t_1
    #             continue
    #         else:
    #             t_next = reversed_window_indices[i - 1]
    #             J_t_1 = self.fwd_variances[-(i + 1), 1] / self.fwd_variances[-i, 0]
    #             p_ext_n_t_1 = self.price_estimates[t] + J_t_1 * (self.price_estimates[t_next] - self.price_estimates[t])
    #             P_n_t_1 = self.fwd_variances[-(i + 1), 1] + J_t_1 ** 2 * (self.variances[t_next] - self.fwd_variances[-i, 0])
    #             self.J[t] = J_t_1
    #             self.variances[t] = P_n_t_1
    #             self.price_estimates[t] = p_ext_n_t_1

    #     for t in range(self.num_observations - 2, -1, -1):
    #         if t not in window_indices:
    #             continue
    #         P_n_t_1_t_2 = self.fwd_variances[t, 1] * self.J[t - 1] + self.J[t - 1] * self.J[t] * (self.covariances[t + 1] - self.fwd_variances[t, 1])
    #         self.covariances[t] = P_n_t_1_t_2

    def backward_pass(self):
        self.variances = np.zeros(self.num_observations)
        self.covariances = np.zeros(self.num_observations)
        self.J = np.zeros(self.num_observations)

        window_indices = list(self.get_window_indices())
        reversed_window_indices = list(reversed(window_indices))

        for i, t in enumerate(reversed_window_indices):
            if t == self.num_observations-1:
                P_n_t_t_1 = (1-self.kalman_gains[-1])*self.fwd_variances[-2,1]
                self.variances[t] = self.fwd_variances[-1,1]
                self.covariances[t] = P_n_t_t_1
                continue
            else:
                J_t_1 = self.fwd_variances[t-self.num_observations,1]/self.fwd_variances[t+1-self.num_observations,0]
                p_ext_n_t_1 = self.price_estimates[t] + J_t_1*(self.price_estimates[t+1] - self.price_estimates[t])
                P_n_t_1 = self.fwd_variances[t-self.num_observations,1] + J_t_1**2 * (self.variances[t+1] - self.fwd_variances[t+1-self.num_observations,0])
                self.J[t] = J_t_1
                self.variances[t] = P_n_t_1
                self.price_estimates[t] = p_ext_n_t_1
        
        for t in range(self.num_observations-2,-1,-1):
            if t not in window_indices:
                break 
            P_n_t_1_t_2 = self.fwd_variances[t-self.num_observations,1]*self.J[t-1] + self.J[t-1]*self.J[t]*(self.covariances[t+1] - self.fwd_variances[t-self.num_observations,1])
            self.covariances[t] = P_n_t_1_t_2

    def calculate_max(self):    
        self.A = np.sum(self.variances[1:]) + np.sum(self.variances[:-1]) - 2*np.sum(self.covariances[1:]) + np.sum((self.price_estimates[1:]-self.price_estimates[:-1])**2)
        self.B = np.sum(self.trader_prices[1:]**2) + np.sum(self.variances[1:]) + np.sum(self.price_estimates[1:]**2) - 2*np.sum(np.multiply(self.trader_prices[1:],self.price_estimates[1:]))
        # print(self.price_estimates)
        self.eta_estimate = np.sqrt(2*self.B/(self.num_observations-1))
        self.sigma_estimate = np.sqrt(2*self.A/(self.num_observations-1))

    def calculate_error(self):
        prev_log_likelihood = self.log_likelihood
        
        self.log_likelihood = - (self.num_observations-1)/2 * np.log(self.sigma_estimate) - (self.num_observations-1)/2 * np.log(self.eta_estimate) 
        self.log_likelihood -= 1/(2*self.sigma_estimate**2) * self.A
        self.log_likelihood -= 1/(2*self.eta_estimate**2) * self.B

        self.error = abs(self.log_likelihood - prev_log_likelihood)
       

    def choose_action(self, state, epsilon=-1):
        p_0 = self.price_estimates[-1]
        #print(p_0,self.variances[-1],self.kalman_gains[-1])
        if self.jump_mode == "linear":
            return p_0
        elif self.jump_mode == "log" or self.jump_mode == "adversarial":
            return math.exp(p_0)# + self.variances[-1]/(2*(1-self.kalman_gains[-1])))

    def update(self, state, action, reward, next_state, done=False):
        if self.jump_mode == "linear":
            trader_price = next_state
        elif self.jump_mode == "log" or self.jump_mode == "adversarial":
            trader_price = math.log(next_state)
        self.num_observations += 1
        self.trader_prices = np.append(self.trader_prices,trader_price)
        self.price_estimates = np.append(self.price_estimates,trader_price)

        while self.error > self.epsilon:
            #print("Error : ",self.error)
            self.forward_pass()
            self.backward_pass()
            self.calculate_max()
            self.calculate_error()
            #print(self.sigma_estimate)
        self.log_likelihood = -float("inf")
        self.error = float("inf")



class RobustKalmanFilter:
    def __init__(
        self,
        initial_price = 1000,
        epsilon = 0.001,
        jump_mode = "linear",
        eta = 1,
        sigma = 1,
    ):
        self.n_actions = 0
        self.n_states = 0
        self.jump_mode = jump_mode

        self.initial_price = initial_price

        self.epsilon = epsilon # em_algo_stop_tolerance
        
        self.sigma_estimate = sigma
        self.eta_estimate = eta
        self.num_observations = 1

        self.price_estimates = np.zeros(self.num_observations)
        if self.jump_mode == "linear" or self.jump_mode == "adversarial":
            self.price_estimates[0] = self.initial_price # initial price is known exactly
            self.trader_prices = np.array([self.initial_price])
        elif self.jump_mode == "log":
            self.price_estimates[0] = math.log(self.initial_price) # initial price is known exactly
            self.trader_prices = np.array([math.log(self.initial_price)])
        self.weight_estimates = np.array([1.0]) # estimates of weights (less weight -> more likely adversarial)
        self.variances = np.array([0]) # P_t^n variable
        self.covariances = np.array([0]) # P_{t,t-1}^n variable
        self.kalman_gains = np.array([0]) # K_t variable
        self.log_likelihood = -float("inf")
        self.error = float("inf")
        
        

    def reset(self):
        self.num_observations = 1

        self.price_estimates = np.zeros(self.num_observations)
        if self.jump_mode == "linear" or self.jump_mode == "adversarial":
            self.price_estimates[0] = self.initial_price # initial price is known exactly
            self.trader_prices = np.array([self.initial_price])
        elif self.jump_mode == "log":
            self.price_estimates[0] = math.log(self.initial_price) # initial price is known exactly
            self.trader_prices = np.array([math.log(self.initial_price)])
        self.weight_estimates = np.array([1.0]) # estimates of weights (less weight -> more likely adversarial)
        self.variances = np.array([0]) # P_t^n variable
        self.covariances = np.array([0]) # P_{t,t-1}^n variable
        self.kalman_gains = np.array([0]) # K_t variable
        self.log_likelihood = -float("inf")
        self.error = float("inf")


    def forward_pass(self):
        self.fwd_variances = np.array([[0,0]])
        self.kalman_gains = np.array([0]) # K_t variable
        for t in range(self.num_observations):
            if t == 0:
                continue
            else:
                p_ext_t_1 = self.price_estimates[t-1]
                P_t_1_t = self.fwd_variances[t-1,1] + self.sigma_estimate**2
                K_t = P_t_1_t/(P_t_1_t + self.eta_estimate**2/self.weight_estimates[t])
                p_ext_t = p_ext_t_1 + K_t * (self.trader_prices[t] - p_ext_t_1)
                P_t_t = (1 - K_t)*P_t_1_t
                self.fwd_variances = np.append(self.fwd_variances,np.matrix([[P_t_1_t,P_t_t]]),axis=0)
                self.kalman_gains = np.append(self.kalman_gains,K_t)
                self.price_estimates[t] = p_ext_t


    def backward_pass(self):
        self.variances = np.zeros(self.num_observations)
        self.covariances = np.zeros(self.num_observations)
        self.J = np.zeros(self.num_observations)

        for t in range(self.num_observations-1,-1,-1):
            if t == self.num_observations-1:
                #print(self.fwd_variances[0])
                P_n_t_t_1 = (1-self.kalman_gains[t])*self.fwd_variances[t-1,1]
                self.variances[t] = self.fwd_variances[t,1]
                self.covariances[t] = P_n_t_t_1
                continue
            else:
                J_t_1 = self.fwd_variances[t,1]/self.fwd_variances[t+1,0]
                p_ext_n_t_1 = self.price_estimates[t] + J_t_1*(self.price_estimates[t+1] - self.price_estimates[t])
                P_n_t_1 = self.fwd_variances[t,1] + J_t_1**2 * (self.variances[t+1] - self.fwd_variances[t+1,0])
                self.J[t] = J_t_1
                self.variances[t] = P_n_t_1
                self.price_estimates[t] = p_ext_n_t_1
        
        for t in range(self.num_observations-2,-1,-1):   
            P_n_t_1_t_2 = self.fwd_variances[t,1]*self.J[t-1] + self.J[t-1]*self.J[t]*(self.covariances[t+1] - self.fwd_variances[t,1])
            self.covariances[t] = P_n_t_1_t_2

    def calculate_max(self):    
        self.A = np.sum(self.variances[1:]) + np.sum(self.variances[:-1]) - 2*np.sum(self.covariances[1:]) + np.sum((self.price_estimates[1:]-self.price_estimates[:-1])**2)
        self.B = self.trader_prices**2 + self.variances + self.price_estimates**2 - 2*np.multiply(self.trader_prices,self.price_estimates)

        # print(self.price_estimates)
        # self.eta_estimate = np.sqrt(2*self.B/(self.num_observations-1))
        # self.sigma_estimate = np.sqrt(2*self.A/(self.num_observations-1))

        self.weight_estimates = 5*self.eta_estimate**2/(2*self.B)

    def calculate_error(self):
        prev_log_likelihood = self.log_likelihood
        
        self.log_likelihood = - (self.num_observations-1)/2 * np.log(self.sigma_estimate) - (self.num_observations-1)/2 * np.log(self.eta_estimate) 
        self.log_likelihood -= 1/(2*self.sigma_estimate**2) * self.A
        self.log_likelihood -= 1/(2*self.eta_estimate**2) * np.dot(self.B,self.weight_estimates)
        self.log_likelihood += 1/4*np.sum(np.log(self.weight_estimates))

        self.error = abs(self.log_likelihood - prev_log_likelihood)
       

    def choose_action(self, state, epsilon=-1):
        p_0 = self.price_estimates[-1]
        if self.jump_mode == "linear" or self.jump_mode == "adversarial":
            return p_0
        elif self.jump_mode == "log":
            return math.exp((p_0) + self.variances[-1]/(2*(1-self.kalman_gains[-1])))

    def update(self, state, action, reward, next_state, done=False):
        self.num_observations += 1
        if self.jump_mode == "linear" or self.jump_mode == "adversarial":
            trader_price = next_state
        elif self.jump_mode == "log":
            trader_price = math.log(next_state)
        
        self.trader_prices = np.append(self.trader_prices,trader_price)
        self.price_estimates = np.append(self.price_estimates,trader_price)
        self.weight_estimates = np.append(self.weight_estimates,1.0)

        while self.error > self.epsilon:
            #print("Error : ",self.error)
            self.forward_pass()
            self.backward_pass()
            self.calculate_max()
            self.calculate_error()
        #print("Error : ",self.error)
        self.log_likelihood = -float("inf")
        self.error = float("inf")

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, x):
        return self.network(x)

class DQN_Agent:
    def __init__(
        self,
        state_size,
        action_size,
        num_adjustments=1,
        window=20,
        hidden_size=64,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9999
    ):
        action_size = (num_adjustments)**2
        self.state_size = state_size
        self.action_size = action_size
        self.window = window
        self.num_adjustments = num_adjustments
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_network = DQN( window, action_size, hidden_size)
        self.target_network = DQN( window, action_size, hidden_size)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < self.epsilon:
            flat_action = np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.q_network(state)
            flat_action = torch.argmax(action_values).item()
        return int(flat_action/(self.num_adjustments)),flat_action%(self.num_adjustments),-1
        
    
    def update(self, state, action, reward, next_state, done=False):
        d = 0
        if done:
            d = 1
        action = self.num_adjustments*action[0]+action[1]
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.tensor([action])
        reward = torch.tensor([reward]).float()
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        done = torch.tensor([done], dtype=torch.bool)
        
        q_values = self.q_network(state).squeeze(0)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_state).squeeze(0)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + self.gamma * next_q_value * (1 - d)
        
        loss = self.loss_fn(q_value, target_q_value).float()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update the target network
        self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

        

class PPO_Agent(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        num_adjustments=1,
        window=20,
        hidden_size=64,
        lr=1e-3,
        eps_clip=0.2,
        gamma=0.99
    ):
        super(PPO_Agent, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.window = window
        self.num_adjustments = num_adjustments
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        self.actor = nn.Sequential(
            nn.Linear(state_size * window, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_size * window, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def evaluate(self, state, action):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(torch.tensor(action))
        entropy = dist.entropy()
        value = self.critic(state)
        return log_prob, entropy, value.squeeze(1)

    def update(self, memory):
        states, actions, rewards, log_probs_old, dones = memory.sample()
        
        # Calculate discounted rewards
        G = []
        g = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + self.gamma * g * (1 - d)
            G.insert(0, g)
        G = torch.tensor(G)
        
        for _ in range(10):  # Number of PPO epochs
            log_probs, entropies, values = self.evaluate(states, actions)
            advantages = G - values.detach()
            
            # Calculate the ratio
            ratio = torch.exp(log_probs - torch.tensor(log_probs_old))
            
            # Calculate the surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate the actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, G)
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropies.mean()
            
            # Update the network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class QLearningAgent:
    def __init__(
        self,
        n_actions,
        n_states,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        variable_trade_size=False,
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.variable_trade_size = variable_trade_size
        self.q_table = self.init_q_table(n_states, n_actions)

    def init_q_table(self, n_states, n_actions):
        q_table = {}
        for state in range(n_states):
            q_table[state] = {}
            for action_0 in range(n_actions[0]):
                q_table[state][action_0] = {}
                for action_1 in range(n_actions[1]):
                    if self.variable_trade_size:
                        q_table[state][action_0][action_1] = {}
                        for action_2 in range(n_actions[2]):
                            q_table[state][action_0][action_1][action_2] = 0
                    else:
                        q_table[state][action_0][action_1] = 0
        return q_table

    def choose_action(self, state, epsilon=-1):
        
        if epsilon == -1:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            if self.variable_trade_size:
                return np.random.choice(self.n_actions[0]), np.random.choice(self.n_actions[1]), np.random.choice(self.n_actions[2])  # Explore with prob epsilon
            else:
                return np.random.choice(self.n_actions[0]), np.random.choice(self.n_actions[1]), -1  # Explore with prob epsilon
        else:
            action_values = self.q_table[state]

            # Extract all Q-values into a list
            q_values = []
            for action_0 in action_values:
                for action_1 in action_values[action_0]:
                    if self.variable_trade_size:
                        for action_2 in action_values[action_0][action_1]:
                            q_values.append((action_values[action_0][action_1][action_2], (action_0, action_1, action_2)))
                    else:
                        q_values.append((action_values[action_0][action_1], (action_0, action_1, -1)))

            # Get the max Q-value
            max_q_value = max(q_values, key=lambda item: item[0])[0]

            # Get all actions that have the max Q-value
            max_actions = [action for q_value, action in q_values if q_value == max_q_value]

            # Choose a random action from those with the max Q-value
            max_action = random.choice(max_actions)

            return max_action  # Exploit
            # action_values = self.q_table[state]
            
            # action0 = max(action_values, key=lambda x: max(action_values[x][a][b].values())) 
            # mv = max(action_values[action0].values())
            # action0 = random.choice([k for (k, v) in action_values.items() if max(v.values()) == mv])

            # if self.variable_trade_size:
            #     action1 = random.choice([k for (k, v) in action_values[action0].items() if max(v.values()) == mv])
            # else:
            #     action1 = random.choice([k for (k, v) in action_values[action0].items() if v == mv])

            # action2 = -1
            # if self.variable_trade_size:
            #     action2 = random.choice([k for (k, v) in action_values[action0][action1].items() if v == mv])

            # return action0,action1,action2  # Exploit

    def update(self, state, action, reward, next_state):
        if self.variable_trade_size:
            predict = self.q_table[state][action[0]][action[1]][action[2]]
            target = reward + self.gamma * max(self.q_table[next_state][a][b][c] for a in range(self.n_actions[0]) for b in range(self.n_actions[1]) for c in range(self.n_actions[2]) )
            self.q_table[state][action[0]][action[1]][action[2]] += self.alpha * (target - predict)
        else:
            predict = self.q_table[state][action[0]][action[1]]
            target = reward + self.gamma * max(self.q_table[next_state][a][b] for a in range(self.n_actions[0]) for b in range(self.n_actions[1]))
            self.q_table[state][action[0]][action[1]] += self.alpha * (target - predict)
        


class QLearningAgentUpperConf:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99, c=0.1, epsilon=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.c = c
        self.Q = np.ones((n_states, n_actions[0],n_actions[1])) * 1/(1-gamma)
        self.Q_hat = np.ones((n_states, n_actions[0],n_actions[1])) * 1/(1-gamma)
        self.state_visit_count = np.zeros(n_states)
        self.state_action_count = np.zeros((n_states,n_actions[0],n_actions[1]))
        self.epsilon=epsilon

    def choose_action(self, state, epsilon = 0):
        if self.state_visit_count[state] == 0:
            self.state_visit_count[state] += 1
            return (np.random.randint(self.n_actions[0]),np.random.randint(self.n_actions[1]))
        
        UCB_values = self.Q_hat[state]
        action_candidates_flat = np.flatnonzero(UCB_values == UCB_values.max())
        action_idx = np.random.choice(action_candidates_flat)
        action = (int(action_idx/self.n_actions[1]),action_idx%self.n_actions[1])
        
        return action

    def update(self, state, action, reward, next_state):
        self.state_action_count[state][action] += 1
        
        exploration_term = self.c / (1-self.gamma) * np.sqrt(np.log(self.state_action_count[state][action]) / self.state_action_count[state][action])
        
        target = reward + exploration_term + self.gamma * np.max(self.Q_hat[next_state])
        error = target - self.Q[state][action]
        self.Q[state][action] += self.alpha * 1/(self.state_action_count[state][action]) * error
        self.Q_hat[state][action] = min(self.Q[state][action],self.Q_hat[state][action])
        
class BayesianAgent:
    def __init__(
        self,
        n_actions,
        n_states,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, epsilon=-1):
        return 0

    def update(self, state, action, reward, next_state,done=False):
        pass

    def reset(self):
        pass

class UnknownBayesianAgent:
    def __init__(
        self,
        n_actions,
        n_states,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, epsilon=-1):
        return -1000

    def update(self, state, action, reward, next_state, done=False):
        pass


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.distributions as dist
import gym

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor_mu = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Output mean of the action distribution
        )
        self.actor_sigma = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # Output log std of the action distribution
            nn.Softplus()  # Ensure that the standard deviation is positive
        )
        
        # Critic network remains the same
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, state):
        mu = self.actor_mu(state)
        sigma = self.actor_sigma(state) + 1e-6
        
        # Create a normal distribution
        dist_normal = dist.Normal(mu, sigma)
        
        # Sample an action
        action = dist_normal.sample()
        
        # Calculate the log probability of the action
        log_prob = dist_normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Get the value function
        value = self.critic(state)
        
        return action, log_prob, value

# Define the PPO algorithm
class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, epochs):
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).float()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
    
    def compute_advantages(self, rewards, values, dones):
        # Compute GAE (Generalized Advantage Estimation)
        advantages = []
        returns = []
        gae = 0
        R = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(advantages).float(), torch.tensor(returns).float()
    
    def update(self, states, actions, log_probs_old, advantages, returns):
        for _ in range(self.epochs):
            # Evaluate current policy
            _, log_probs, values = self.policy.act(states)
            ratio = (log_probs - log_probs_old).exp()
            
            # Compute surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Compute critic loss
            critic_loss = 0.5 * (returns - values).pow(2).mean()
            
            # Total loss
            loss = actor_loss + critic_loss
            
            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class PPOKalmanFilter:
    def __init__(
        self,
        initial_price = 1000,
        epsilon = 0.2,
    ):
        self.n_actions = 0
        self.n_states = 0

        self.initial_price = initial_price

        self.epsilon = epsilon # em_algo_stop_tolerance
        
        self.sigma_estimate = 1
        self.eta_estimate = 1
        self.num_observations = 1

        self.price_estimate = self.initial_price
        self.kalman_gain = 0
        self.P_t_t = 0

        state_dim = 1
        action_dim = 2
        self.memory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}
        self.ppo_agent = PPO(state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, epsilon=epsilon, epochs=4)
        

    def reset(self):
        self.sigma_estimate = 1
        self.eta_estimate = 1
        self.num_observations = 1

        self.price_estimate = self.initial_price

        state_dim = 1
        action_dim = 2
        self.memory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}
       

    def choose_action(self, state, epsilon=-1):
        p_0 = self.price_estimate
        return p_0

    def update(self, state, action, reward, next_state,done=False):
        trader_price = next_state
        reward = -self.kalman_gain * abs(trader_price - self.price_estimate)

        self.num_observations += 1

        state_tensor = torch.tensor(trader_price).float().unsqueeze(0).unsqueeze(0)
        print(state_tensor)
        action, log_prob, _ = self.ppo_agent.policy.act(state_tensor)


        self.memory["states"].append(trader_price)
        self.memory["actions"].append(action)
        self.memory["log_probs"].append(log_prob.item())
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)

        print(action)
        self.eta_estimate = action[0][0]
        self.sigma_estimate = action[0][1]

        self.kalman_gain = (self.P_t_t + self.sigma_estimate**2)/(self.P_t_t + self.sigma_estimate**2 + self.eta_estimate**2)
        self.price_estimate = self.price_estimate + self.kalman_gain*(trader_price-self.price_estimate)
        self.P_t_t = (1-self.kalman_gain)*(self.P_t_t + self.sigma_estimate**2)
        
        if done:
            # Compute advantages and returns
            with torch.no_grad():
                _, _, next_value = self.ppo_agent.policy.act(torch.tensor(next_state).float().unsqueeze(0))
                values = self.ppo_agent.policy.critic(torch.tensor(self.memory["states"]).float()).detach().squeeze().numpy().tolist()
                values.append(next_value.item())
                advantages, returns = ppo_agent.compute_advantages(self.memory["rewards"], values, self.memory["dones"])
                
                # Update policy
                ppo_agent.update(
                    torch.tensor(memory["states"]).float(),
                    torch.tensor(memory["actions"]),
                    torch.tensor(memory["log_probs"]).float(),
                    advantages,
                    returns
                )


# # Example usage with OpenAI Gym
# env = gym.make('CartPole-v1')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

# # Initialize PPO agent
# ppo_agent = PPO(state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=4)

# # Training loop
# num_episodes = 1000
# max_timesteps = 200

# for episode in range(num_episodes):
#     state = env.reset()
#     memory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}
    
#     for t in range(max_timesteps):
#         state_tensor = torch.tensor(state).float().unsqueeze(0)
#         action, log_prob, _ = ppo_agent.policy.act(state_tensor)
        
#         next_state, reward, done, _ = env.step(action.item())
        
#         # Store transition
#         memory["states"].append(state)
#         memory["actions"].append(action.item())
#         memory["log_probs"].append(log_prob.item())
#         memory["rewards"].append(reward)
#         memory["dones"].append(done)
        
#         if done or t == max_timesteps-1:
#             # Compute advantages and returns
#             with torch.no_grad():
#                 _, _, next_value = ppo_agent.policy.act(torch.tensor(next_state).float().unsqueeze(0))
#             values = ppo_agent.policy.critic(torch.tensor(memory["states"]).float()).detach().squeeze().numpy().tolist()
#             values.append(next_value.item())
#             advantages, returns = ppo_agent.compute_advantages(memory["rewards"], values, memory["dones"])
            
#             # Update policy
#             ppo_agent.update(
#                 torch.tensor(memory["states"]).float(),
#                 torch.tensor(memory["actions"]),
#                 torch.tensor(memory["log_probs"]).float(),
#                 advantages,
#                 returns
#             )
#             break
        
#         state = next_state
    
#     # Logging
#     if episode % 10 == 0:
#         print(f"Episode: {episode}, Total Reward: {sum(memory['rewards'])}")

# env.close()

#         