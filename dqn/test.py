import numpy as np
import math 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.action_dim = 2
        self.state_dim = 4
        self.fc1 = nn.Linear(self.state_dim, 16)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(16, self.action_dim)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class Environment(object):
    def __init__(self, filename):
        ### data format(5g_est, wifi_est, 5g_prob, wifi_prob)
        ### action 0:5g, 1:wifi
        self.ho_factor = 0.95 #handover cost
        self.data = []
        self.training_data = []
        self.index = 0
        self.action = 0
        self.filename = filename
        fp = open(self.filename, "r")
        line = fp.readline()
        while line:
            self.data.append([float(x) for x in line.split()])
            line = fp.readline()
        fp.close()

        self.total_nb = len(self.data)
    def reset(self):
        self.index = 0
        self.training_data = self.data
        ret_state = self.training_data[self.index].copy()
        ret_state[(self.action^1)+2] = 0
        return ret_state
    
    def step(self, action):
        reward = 0
        current_state = self.training_data[self.index]
        if action!=self.action:
            reward = (self.ho_factor * current_state[action+2] - current_state[(action^1)+2])
        else:
            reward = (current_state[action+2] - current_state[(action^1)+2])
        if reward > 0:
            reward = 0.5
        else:
            reward /= (current_state[action+2] if current_state[action+2]>10 else 10)
        state_after = self.training_data[self.index].copy()
        state_after[(action^1)+2] = 0
        return (reward, state_after)
   
    def next_state(self):
        self.action^=1
        if self.action==0:
            self.index+=1
        if self.index==self.total_nb:
            return self.reset()
        ret_state = self.training_data[self.index].copy()
        ret_state[(self.action^1)+2] = 0
        return ret_state
        
def select_action(eval_net, state):
    state = torch.unsqueeze(torch.FloatTensor(state), 0)
    actions_value = eval_net(state)
    action = torch.max(actions_value, 1)[1].data.numpy()[0]
    return action

def main():
    # file_name = "1000_500_b512"
    # dqn = torch.load("good_pretrain_model/2000_2000_b128.pt")
    dqn = torch.load("2400_500.pt")

    env = Environment("data_pretrain.txt")
    # env = Environment("data_online.txt")
    fp = open("throughput_6f_pretrain.txt", "w")
    data = env.data.copy()
    action=0
    reward=0
    for d in data:
        _data = d.copy()
        d[(action^1)+2]=0
        dqn_action = select_action(dqn, d)
        if action!=dqn_action:
            cur_reward = (0.95 * _data[dqn_action+2] - _data[(dqn_action^1)+2])
        else:
            cur_reward = (_data[dqn_action+2] - _data[(dqn_action^1)+2])
        if cur_reward > 0:
            cur_reward = 0.5
        else:
            cur_reward /= (_data[action+2] if _data[action+2]>10 else 10)
        reward += cur_reward
        
        opt_thrp = _data[2] if _data[2] > _data[3] else _data[3]
        fp.write(f'{_data[2]} {_data[3]} {opt_thrp} {_data[dqn_action+2]}\n')
        
        action = dqn_action
        print("state before: ", d)
        print("action: ", action)
        print("reward:", cur_reward)
        print("#############################################################################################")
    print("total reward:", reward)
    
if __name__ == '__main__':
    main()