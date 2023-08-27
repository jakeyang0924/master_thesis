import numpy as np
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
        self.action=0
        self.training_data = self.data.copy()
        random.shuffle(self.training_data)
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

class Q_learning(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.action_dim = self.eval_net.action_dim
        self.state_dim = self.eval_net.state_dim
        self.memory_count = 0
        self.memory_capacity = 1000
        self.memory = np.zeros((self.memory_capacity, self.state_dim*2+2)) #state, action, reward, state_after
        self.target_update_count = 0
        self.target_update_it = 50
        self.gamma = 0.95
        self.epsolon = 0.1
        self.min_epsilon = 0
        self.max_epsilon = 1
        self.decay = 500
        self.lr = 0.0005
        self.batch_size = 256
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def train(self):
        if self.target_update_count == self.target_update_it:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.target_update_count=0
                
        self.target_update_count+=1

        sample_idx = np.random.choice(self.memory_capacity, self.batch_size)
        bs_memory = self.memory[sample_idx, :]
        bs_s = torch.FloatTensor(bs_memory[:, :self.state_dim])
        bs_a = torch.LongTensor(bs_memory[:, self.state_dim:self.state_dim+1])
        bs_r = torch.FloatTensor(bs_memory[:, self.state_dim+1:self.state_dim+2])
        bs_s_ = torch.FloatTensor(bs_memory[:, -self.state_dim:])

        q_eval = self.eval_net(bs_s).gather(1, bs_a)  # shape (batch, 1)
        q_next = self.target_net(bs_s_).detach()     # detach from graph, don't backpropagate
        q_target = bs_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def store(self, state, action, reward, state_after):
        idx = self.memory_count%self.memory_capacity
        self.memory[idx, :] = np.hstack((state, action, reward, state_after))
        self.memory_count+=1

        if self.memory_count >= self.memory_capacity:
            self.train()
    
    def select_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        # input only one sample
        if np.random.uniform() < self.epsolon :   # greedy
            actions_value = self.eval_net(state)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:   # random
            action = np.random.randint(0, self.action_dim)
        return action

def test(dqn):
    env = Environment("data_pretrain.txt")
    data = env.data.copy()
    action=0
    reward_sum=0
    for d in data:
        _data = d.copy()
        d[(action^1)+2]=0
        dqn_action = dqn.select_action(d)
        if action!=dqn_action:
            reward = (0.95 * _data[dqn_action+2] - _data[(dqn_action^1)+2])
        else:
            reward = (_data[dqn_action+2] - _data[(dqn_action^1)+2])
        if reward > 0:
            reward_sum += 0.5
        else:
            reward_sum += reward / (_data[dqn_action+2] if _data[dqn_action+2]>10 else 10)
        action = dqn_action

    return reward_sum


def main():
    dqn = Q_learning()
    env = Environment("data_pretrain.txt")
    fp = open("accu_rw_pretrain_epoch.txt", "w")
    fp2 = open("accu_rw_pretrain_step.txt", "w")
    fp3 = open("test_rw_pretrain.txt", "w")
    state = env.reset()
    accu_rw = 0
    for it in range(1, 4001):
        dqn.epsolon = dqn.max_epsilon - (dqn.max_epsilon - dqn.min_epsilon) * math.exp(-1.0 * it / dqn.decay)

        for _ in range(500):
            action = dqn.select_action(state)
            reward, state_after = env.step(action)
            accu_rw += reward
            
            fp2.write(str(accu_rw)+'\n')
            
            dqn.store(state, action, reward, state_after)
            state=env.next_state()

        test_rw = test(dqn)
        print(f'Ep: {it}, current epsolon: {dqn.epsolon}, current accu_rw: {accu_rw}, test_rw: {test_rw}')
        fp.write(str(accu_rw)+'\n')
        fp3.write(str(test_rw)+'\n')

        if it % 200 == 0:
            FILE = 'dqn_model_{}.pt'.format(it)
            torch.save(dqn.eval_net, FILE)
    fp.close()
    return

if __name__ == '__main__':
    main()