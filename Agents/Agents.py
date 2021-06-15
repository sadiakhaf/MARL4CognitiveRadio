import torch 
import torch.nn as nn
import torch.optim as optim
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from SUs.SU import SU
from Policies.Policy import *
from Networks.Networks import Network
from Memory.memory import *
device = 'cpu'

class Agent(SU): 
    
    def __init__(self, lr=0.1, gamma=0.1, numEpisodes=200, epsilon_type='regular', horizon = 20, action_space = None, num_agents = 5, num_channels = 10, num_possible_actions =3):
        super(Agent, self).__init__()
        self.lr = lr #learning rate
        self.num_agents = num_agents
        self.num_channels = num_channels
        self.num_possible_actions = num_possible_actions
        self.gamma = gamma
        self.numEpisodes = numEpisodes
        # self.qtable = torch.zeros(horizon+1, 2)  # Q-table with time-step as keys for a 2D Q-table
        self.qtable = torch.zeros(horizon+1,num_agents, num_channels, num_possible_actions)  # Q-table with time-step as keys for a 2D Q-table
        self.epsilon_type = epsilon_type
        self.horizon = horizon
        self.action_space = action_space
        self.CreateEpsilonFunction()


    def CreateEpsilonFunction(self,sigma=5):
        if self.epsilon_type == 'wavy': #The default epsilon type is regular-exponential, the other option is 'wavy'
            Epsilonfn = wavyexponential(self.numEpisodes,sigma=sigma)
      # Epsilonfn.render()
        elif self.epsilon_type == 'regular':
            Epsilonfn = regexponential(self.numEpisodes,sigma=sigma)
        elif self.epsilon_type == 'constant':
            Epsilonfn = regexponential(self.numEpisodes,sigma=sigma)
            Epsilonfn.epsilon = torch.ones((self.numEpisodes,1))*0.1
        
        self.epsilon = Epsilonfn.epsilon
        # Epsilonfn.render()
    

    def act(self, inumEP, o ): 
        actions = torch.zeros(self.num_agents, self.num_channels,dtype=torch.int)
        
        for i_agent in range(self.num_agents): 
            if torch.rand(1) < self.epsilon [inumEP]: 
                temp = self.action_space.sample()
                actions [i_agent,:] = torch.tensor(temp[i_agent, :],dtype=int)
            else:
                actions = self.qtable[o].max(2)[1]

        return actions

    def update(self, o: int, action, r, o_prime: int):  # agent update function (e.g. Q-learning update)
        delta_update  = 0
        for i_agent in range(self.num_agents): 
            # old_o_a_value  = self.qtable[o][action]  #store the Q-value for observation o and action a 
            old_o_a_value  = self.qtable[o][i_agent][range(self.num_channels),action[i_agent,:].long()]  #store the Q-value for observation o and action a 
            q_prime  = self.qtable[o_prime][i_agent].max(1)[0] # estimate of optiomal future value (maximum Q-value in observation o_prime)

            # Update Q-table based on the equation above
            td_target = r[i_agent] + self.gamma * q_prime

            self.qtable[o][i_agent][range(self.num_channels),action[i_agent,:].long()] += self.lr* (td_target - old_o_a_value)
            
            delta_update += self.qtable[o][i_agent][range(self.num_channels),action[i_agent,:].long()] - old_o_a_value

        return delta_update # return delta update to training loop


class DQNagent(Agent):
    def __init__(self, lr=0.1, gamma=0.1, numEpisodes=200, epsilon_type='regular', horizon = 20, action_space = None,
     num_agents = 5, num_channels = 10, num_possible_actions =3, BATCH_SIZE = 60):
        super(DQNagent, self).__init__(lr=lr, gamma=gamma, numEpisodes=numEpisodes, epsilon_type=epsilon_type, 
        horizon = horizon, action_space = action_space, num_agents = num_agents, num_channels = num_channels, num_possible_actions =num_possible_actions)
        self.state_space_n = 1
        self.action_space_len = num_agents*num_channels*num_possible_actions
        self.memory = ReplayMemory(10000)
        self.BATCH_SIZE = BATCH_SIZE
        self.policy_net = Network(self.state_space_n, self.action_space_len)
        self.target_net = Network(self.state_space_n, self.action_space_len)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)

    def act(self, inumEP, o ): 
        actions = torch.zeros(self.num_agents, self.num_channels,dtype=torch.int)
        
        for i_agent in range(self.num_agents): 
            if torch.rand(1) < self.epsilon [inumEP]: 
                temp = self.action_space.sample()
                actions [i_agent,:] = torch.tensor(temp[i_agent, :],dtype=int)
            else:
                with torch.no_grad():
                    actions = self.policy_net(torch.tensor(o*1.0)).view(self.num_agents, self.num_channels,self.num_possible_actions).max(2)[1]
                
        return actions


    def update(self, o: int, action, r, o_prime: int):  # agent update function (e.g. Q-learning update)
        if len(self.memory) < self.BATCH_SIZE:
            return 0.99
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(batch.state).to(device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.tensor(batch.next_state).to(device)

        fake_a = action_batch.view(self.BATCH_SIZE,self.num_agents,self.num_channels).long() #actons reshaped for getting q values
        #Old Q values
        state_action_values = self.policy_net(state_batch*1.0).view(self.BATCH_SIZE,self.num_agents,self.num_channels,self.num_possible_actions).gather(3,fake_a.unsqueeze(3))
        next_state_values = self.target_net(next_state_batch*1.0).view(self.BATCH_SIZE,self.num_agents,self.num_channels,self.num_possible_actions).max(3)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.view(self.BATCH_SIZE,self.num_agents,self.num_channels)
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # print("Actual: {} Predicted: {}".format(state_action_values.shape,expected_state_action_values.shape))
        loss = criterion(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.data # return delta update to training loop 


if __name__ == "__main__": 
    s = Agent()
    s.CreateEpsilonFunction()
    print(s.epsilon)
   