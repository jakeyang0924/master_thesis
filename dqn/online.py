import numpy as np
import math
import time
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
        self.history_memory_count = 0
        self.history_memory_capacity = 1000
        self.history_memory = np.zeros((self.history_memory_capacity, self.state_dim*2+2)) #state, action, reward, state_after
        self.online_memory_count = 0
        self.online_memory_capacity = 1000
        self.online_memory = np.zeros((self.online_memory_capacity, self.state_dim*2+2)) #state, action, reward, state_after
        self.target_update_count = 0
        self.target_update_iter = 50
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
        if self.target_update_count == self.target_update_iter:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.target_update_count=0
                
        self.target_update_count+=1
        online_batch_size = int(self.batch_size*0.8) if self.batch_size*0.8<=self.online_memory_count else self.online_memory_count
        history_batch_size = self.batch_size-online_batch_size
        online_batch = self.online_memory_count if self.online_memory_count<self.online_memory_capacity else self.online_memory_capacity
        sample_idx_o = np.random.choice(online_batch, online_batch_size)
        sample_idx_h = np.random.choice(self.history_memory_capacity, history_batch_size)
        bs_memory_o = self.online_memory[sample_idx_o, :]
        bs_memory_h = self.history_memory[sample_idx_h, :]
        bs_memory = np.concatenate((bs_memory_o, bs_memory_h))
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
        
    
    def history_store(self, state, action, reward, state_after):
        idx = self.history_memory_count%self.history_memory_capacity
        self.history_memory[idx, :] = np.hstack((state, action, reward, state_after))
        self.history_memory_count+=1

    def online_store(self, state, action, reward, state_after):
        idx = self.online_memory_count%self.online_memory_capacity
        self.online_memory[idx, :] = np.hstack((state, action, reward, state_after))
        self.online_memory_count+=1
    
    def select_action(self, state, prob=0.9):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        # input only one sample
        if np.random.uniform() < prob :   # greedy
            actions_value = self.eval_net(state)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:   # random
            action = np.random.randint(0, self.action_dim)
        return action
    

def reward(state, action_ori, action):
    if action!=action_ori:
        reward = (0.95 * state[action+2] - state[(action^1)+2])
    else:
        reward = (state[action+2] - state[(action^1)+2])
    if reward > 0:
        reward = 0.5
    else:
        reward /= (state[action+2] if state[action+2]>10 else 10)
    return reward

def online_store(dqn, data):
    data = [float(x) for x in data.split()]
    data0, data1 = data.copy(), data.copy()
    ori_action0, ori_action1 = 0, 1
    data0[(ori_action0^1)+2], data1[(ori_action1^1)+2] = 0, 0
    action0, action1 = dqn.select_action(data0), dqn.select_action(data1)
    reward0, reward1 = reward(data, ori_action0, action0), reward(data, ori_action1, action1)
    data0_after, data1_after = data.copy(), data.copy()
    data0_after[(action0^1)+2], data1_after[(action1^1)+2] = 0, 0
    dqn.online_store(data0, action0, reward0, data0_after)
    dqn.online_store(data1, action1, reward1, data1_after)
    return

def history_store(env, dqn, state):
    action = dqn.select_action(state)
    reward, state_after = env.step(action)
    dqn.history_store(state, action, reward, state_after)
    state=env.next_state()
    return state

def test(dqn):
    env = Environment("data_online_rand.txt")
    data = env.data.copy()
    action=0
    reward_sum=0
    for d in data:
        _data = d.copy()
        d[(action^1)+2]=0
        dqn_action = dqn.select_action(d, 1.1)
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
    model_name = "2400_500"
    dqn = Q_learning()
    dqn.eval_net = torch.load(f'{model_name}.pt')
    dqn.target_net = torch.load(f'{model_name}.pt')
    dqn.optimizer = torch.optim.Adam(dqn.eval_net.parameters(), lr=dqn.lr)
    env = Environment("data_pretrain.txt")
    state = env.reset()
    while dqn.history_memory_count < dqn.history_memory_capacity:
        state = history_store(env, dqn, state)
    
    iter_cnt = 0
    fp = open("test_rw_online2400.txt", "w")
    fp2 = open("us_per_train.txt", "w")
    print("start:")
    try:
        while True:
            iter_cnt += 1
            data = input()
            if len(data)==0:
                break
            
            # start training
            start_time = time.time()
            
            online_store(dqn, data)
            x = dqn.online_memory_count if dqn.online_memory_count<100 else 100
            for _ in range(x):
                state = history_store(env, dqn, state)
                dqn.train()
            
            end_time = time.time()
            elapsed_microseconds = (end_time - start_time) * 1e6
            fp2.write(str(elapsed_microseconds)+'\n')
            
            # test after training
            test_rw = test(dqn)
            print(f'round: {iter_cnt}, test_rw: {test_rw}')
            fp.write(str(test_rw)+'\n')
            
    except EOFError as e:
        print('EOF')
    FILE = f'{model_name}_online_rand.pt'
    torch.save(dqn.eval_net, FILE)            
    return

if __name__ == '__main__':
    main()
