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

# memory = ReplayMemory(Horizon*num_episodes)


# create Environment/ Import environment
env = Env(Horizon= Horizon, num_PU=num_PU, num_SU = num_SU, num_possible_actions=num_possible_actions)

obs_space_len = Horizon+1
action_space_len = num_SU*num_PU*num_possible_actions
if algo == 'DQN':
    agent = DQNagent(lr=lr, gamma=gamma, numEpisodes=num_episodes, epsilon_type='regular', 
    horizon = Horizon, action_space = env.action_space,num_channels=num_PU, num_agents=num_SU, 
    num_possible_actions=num_possible_actions, BATCH_SIZE = BATCH_SIZE)  
else:
    agent = Agent(lr=lr, gamma=gamma, numEpisodes=num_episodes, epsilon_type='regular', 
    horizon = Horizon, action_space = env.action_space,num_channels=num_PU, num_agents=num_SU, 
    num_possible_actions=num_possible_actions)

running_delta = []  # running delta (e.g. the last running_len delta update)
running_acc = []  # running accuracy (e.g. the last running_len accuracy)



for episode_i in range(num_episodes): # episode loop
    done = False
    delta_update = []  # delta update of our Q-table
    n_successes: int = 0  # number of optimal actions (actions with maximum reward)
    cumul_r: float = 0.0  # cumulative reward
    o = env.reset()
    delta_update = 0 
    while not done:
        a =  agent.act(o = o,  inumEP = episode_i)
        o_prime, r, done, _ = env.step(action=a, o=o)
        # delta_update.append(agent.update(o=o,action=a,r=r,o_prime=o_prime))  # update agent with transition, get delta update
        delta_update += (agent.update(o=o,action=a,r=r,o_prime=o_prime))  # update agent with transition, get delta update
        o = o_prime
        cumul_r += r  # add reward to cumulative reward
        n_successes += sum(sum(r))  # success if optimal action-reward of 1.0
        
        agent.memory.push(o, a, o_prime, r)
        if episode_i % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    # print("Episode: {} Reward: {}".format(i_episode,reward_per_episode[i_episode]))
    running_acc.append(n_successes*100. / (env.Horizon*num_PU*num_SU))  # add latest accuracy to running data
    running_delta.append(delta_update)  # add latest update delta to running data
    
  # print(running_delta)
    if episode_i % PRINT_EVERY == 0:
        print(
        'episode {episode_i}: cumul_reward={cumul_r}, accuracy:{acc}, '
        'cumul_delta={cumul_delta:2.2}'.format(
            episode_i=episode_i, cumul_r=n_successes.item(), acc=running_acc[-1].item(),
            cumul_delta=running_delta[episode_i]
        )
        )

if algo == 'DQN':
    with open ('memory.pickle','wb') as f :
        pickle.dump(agent.memory,f)


env.close()  # close gym environment

plt.plot(running_acc)
plt.show()
