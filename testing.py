import torch
import pickle
from Memory.memory import Transition
from Networks.Networks import Network
import time


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

with open('memory.pickle', 'rb') as f: 
    memory = pickle.load(f)


from Parameters.Parameters import *



transitions = memory.sample(BATCH_SIZE)

batch = Transition(*zip(*transitions))
# print("State batch :",batch.state)
state_batch = torch.tensor(batch.state).to(device)
# print("State batch :",state_batch)

action_batch = torch.cat(batch.action).to(device)
# print("\nAcions batch: ",action_batch)

reward_batch = torch.cat(batch.reward).to(device)
# print("\nReward batch: ",reward_batch)

next_state_batch = torch.tensor(batch.next_state).to(device)
# print("\nNext State batch :",next_state_batch)

policy_net = Network(state_space_n, action_space_len)
target_net = Network(state_space_n, action_space_len)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

fake_a = action_batch.view(BATCH_SIZE,num_SU,num_PU).to(device)
state_action_values = policy_net(state_batch*1.0).view(BATCH_SIZE,num_SU,num_PU,num_possible_actions).gather(3,fake_a.unsqueeze(3))
best_actions = policy_net(state_batch*1.0).view(BATCH_SIZE,num_SU,num_PU,num_possible_actions).max(3)[1]
best_values = policy_net(state_batch*1.0).view(BATCH_SIZE,num_SU,num_PU,num_possible_actions).max(3)[0]

print("state action vlues: ",state_action_values.shape)
print("best actions: ",best_actions)
print("best values: ",best_values)


