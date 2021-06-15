import torch 
import matplotlib.pyplot as plt 
from Envs.Env import Env
from Agents.Agents import *
from Networks.Networks import Network

from Memory.memory import *
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


torch.manual_seed(1234)  # python random number generator seed

algo = 'DQN'
lr = 0.1  # learning rate
gamma = 0.95  # gamma parameter
num_episodes = 10000  # number of steps (episodes) in epsilon log-space
num_PU = 2
num_SU = 2
Horizon =20 
num_possible_actions = 3
TARGET_UPDATE = 50
PRINT_EVERY = 50
BATCH_SIZE = 60
obs_space_len = Horizon+1
state_space_n = 1
action_space_len = num_SU*num_PU*num_possible_actions
running_delta = []  # running delta (e.g. the last running_len delta update)
running_acc = []  # running accuracy (e.g. the last running_len accuracy)