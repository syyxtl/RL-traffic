#### single agent q learning for car sharing ####
#### author: shen zhiyong 
#### base model

import os
import csv
import random
import numpy as np
import pandas as pd
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets

##################################################### base model ####################################################
class BModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=1657, norm_layer=nn.BatchNorm2d):
        super(BModel, self).__init__()
        self.blinear = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(2048, output_dim),
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def get_parameters(self, base_lr):
        params_group = []
        for key, param in self.named_parameters():
            if param.requires_grad == False:
                continue
            params_group.append(param)

        params = [{'params':params_group, 'lr':base_lr}]
        return params

    def forward(self, input):
        IN = input
        OUT = self.blinear(IN)
        return OUT
        
##################################################### DQN ####################################################
class DQN(object):
    def __init__(self, root, filelist):
        # dataset
        self.root = root
        self.filelist = filelist
        # rl env
        self.start = 0
        self.end = len(filelist) - 1
        self.ACTION = self.gen_action(False)
        # model
        self.walk = 0
        self.batch_size = 128
        self.update_freq = 10   # 模型更新频率 // target network 更新频率
        self.replay_size = 2000  # 训练集大小
        self.replay_memory = deque(maxlen=self.replay_size)
        self.eval_network = BModel().cuda()
        self.target_network = BModel().cuda()

    ##################################################### datasets ####################################################
    def get_item_f(self, id):
        filename = str(id) + "_out"
        if filename not in self.filelist:
            return None
        else:
            filename = os.path.join(self.root,filename + ".txt")
            orderLabels = ["ID", "stime", "etime", "upr", "upc", "downr", "downc", "true_d", "true_v", "rewards"]
            orderSep = ","
            data = pd.read_csv(filename, names=orderLabels, sep=orderSep, index_col=False)
            lens = len(data)
            return {"data":data, "lens":lens}

    def get_item_l(self, data, id):
        lens = data["lens"]
        data = data["data"]
        if id < 0 and id >= lens: 
            return []
        else:
            # ID = data.iloc[id]["ID"]
            stime = data.iloc[id]["stime"]
            etime = data.iloc[id]["etime"]
            upr = data.iloc[id]["upr"]
            upc = data.iloc[id]["upc"]
            downr = data.iloc[id]["downr"]
            downc = data.iloc[id]["downc"]
            true_d = data.iloc[id]["true_d"]
            true_v = data.iloc[id]["true_v"]
            rewards = data.iloc[id]["rewards"]
            # return [ID, stime, etime, upr, upc, downr, downc, true_d, true_v, rewards]
            return [stime, etime, upr, upc, downr, downc, true_d, true_v, rewards]

    def gen_action(self, turnon=False):
        if turnon:
            maxACTION = 0
            for idx in range(self.start, self.end):
                data = self.get_item_f(idx)
                if data["lens"] > maxACTION:
                    maxACTION = data["lens"]
            return maxACTION
        else:
            return 1657

    ##################################################### model ####################################################
    def remember(self, s, a, next_s, r):
        self.replay_memory.append((s, a, next_s, r))

    def train_one_epoch(self, optimizer, criterion, inputs, LABEL):
        inputs = torch.tensor(inputs.clone().detach(), dtype=torch.float32)
        inputs = inputs.cuda()
        # LABEL = LABEL.cuda()
        OUT = self.eval_network(inputs)
        optimizer.zero_grad()
        total_loss = criterion(OUT, LABEL)  
        total_loss.backward()
        optimizer.step()
        return total_loss

    def train(self, optimizer, criterion, lr=0.01, gamma=0.95):
        if len(self.replay_memory) < self.replay_size:
            return
        self.walk += 1
        # update target_network with eval net_work
        if self.walk % self.update_freq == 0:
            '''
            state_dict = self.eval_network.state_dict()
            model_dict = self.target_network.state_dict()
            new_dict = {}
            for key in state_dict.keys():
                if key in model_dict.keys():
                    if model_dict[key].shape == state_dict[key].shape:
                        new_dict[key] = state_dict[key]
                    else:
                        print('size mismatch')
            self.target_network.load_state_dict(new_dict, strict=False)
            '''
            self.target_network.load_state_dict(self.eval_network.state_dict())

        # gen batch
        replay_batch = random.sample(self.replay_memory, self.batch_size)
        s_batch      = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        # update Q function
        s_batch = torch.tensor(s_batch, dtype=torch.float32)
        s_batch = s_batch.cuda()
        Q = self.eval_network(s_batch)
        next_s_batch = torch.tensor(next_s_batch, dtype=torch.float32)
        next_s_batch = next_s_batch.cuda()
        Q_next = self.target_network(next_s_batch)
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + gamma * torch.max(Q_next[i]))

        # train
        loss = self.train_one_epoch(optimizer, criterion, s_batch, Q)
    
    ##################################################### env ######################################################
    def reset(self, ):
        self.idx = 0
        self.state = [0, 0, 0]
        self.state_n = [0, 0, 0]
        return self.state

    def call_d(self, ph1, ph2): # distance
        # (row, col)
        d = abs(ph1[0]-ph2[0]) + abs(ph1[1]-ph2[1])
        return d

    def action_policy(self, dataset, state, epsilon=0.1):
        lend = dataset["lens"]
        ACTION_FULL = [i for i in range(lend)]
        state = state
        ACTION = []
        for i in range(lend):
            tdata = self.get_item_l(dataset, i)
            td = self.call_d(state, (tdata[2], tdata[3]))
            tv = tdata[7] # true_v
            if td < 5:
                ACTION.append(i)
        
        if self.idx !=0 and len(ACTION) == 0:
            return -1, "non exp"
        if self.idx == 0:
            ACTION = ACTION_FULL
        exp = ""
        if np.random.uniform() < epsilon:
            exp = "explore"
            return np.random.choice(ACTION), exp
        else:
            # for table alg
            # values_ = q_values[state[0], state[1], state[2], 0:lend]
            # return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
            exp = "exploit"
            inputs = torch.tensor(np.array([state]), dtype=torch.float32)
            inputs = inputs.cuda()
            outputs = self.eval_network(inputs).cpu().detach().numpy()
            action_all = outputs[0]
            maxx = -999999999
            action_choice = 0
            for index in ACTION:
                if action_all[index] > maxx:
                    maxx = action_all[index]
                    action_choice = index
            action = action_choice
            return action, exp
    
    def step(self, action, dataset): 
        self.idx = self.idx
        reward = 0
        DONE = False
        # [stime, etime, upr, upc, downr, downc, true_d, true_v, rewards]
        if action != -1:
            lined = self.get_item_l(dataset, action)
            self.state = [self.state[0], self.state[1], self.idx] ###### change ######
            self.idx = lined[1]
            self.state_n = [lined[4], lined[5], self.idx] ###### change ######
            reward = lined[8]
            if lined[1] < lined[0]:
                DONE = True
            if lined[1] == lined[0]:
                self.idx = self.idx + 1
        else:
            self.state = [self.state[0], self.state[1], self.idx]
            self.idx = self.idx + 1
            self.state_n = [self.state[0], self.state[1], self.idx]

        return self.state, self.state_n, reward, DONE
    
    ##################################################### training ####################################################
    def qlearning(self, SIZE, episodes=400, render=True, epsilon=0.1):
        # q_values = np.zeros(SIZE)
        ep_rewards = []
        # Qlearning begin...
        base_lr = 0.001
        optimizer = torch.optim.Adam(self.eval_network.get_parameters(base_lr), lr=base_lr, \
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = nn.SmoothL1Loss()

        for ep in range(0, episodes):
            # npy = gamme 0.9
            root = "/home/ssd1/shenzhiyong/npy/gamma095/"
            torch.save(self.eval_network.state_dict(), os.path.join(root, "eval_model_" + str(ep) + ".pth"))
            print("saved model and start episode", ep, ".........................................................")
            state = self.reset()
            done = False
            reward_sum = 0

            while done == False and self.idx < self.end:
                dataset = self.get_item_f(self.idx)
                action, exp = self.action_policy(dataset, state, epsilon)
                state, next_state, reward, done = self.step(action, dataset)
                # train
                self.remember(state, action, next_state, reward)
                self.train(optimizer, criterion)
                # if table
                # q_values[state[0], state[1], state[2], action] += learning_rate * \
                #     (reward + gamma * np.max(q_values[next_state[0], next_state[1], state[2],:]) - \
                #         q_values[state[0], state[1], state[2], action])
                self.state = self.state_n
                # for comparsion, record all the rewards, this is not necessary for QLearning algorithm
                reward_sum += reward
                print("episode:", ep, "action:", action, "for:", exp , "reward:", reward, "state:", state)

            ep_rewards.append(reward_sum)
            print("finsished episode:", ep, "ep_rewards:", reward_sum, "/")
           
            

        # Qlearning end...
        return ep_rewards

# grid [ 29.51723 103.25635 ] to [ 31.01437 104.483069 ]
# 0.01141 = 1m jd bigger
# 0.00899 = 1m wd smaller

if __name__ == '__main__':
    # dataset, 0-287_out.txt   
    dr = "out5m_req/"
    fl = [str(i)+"_out" for i in range(287)]
    # print(int((31.01437-29.51723)//0.00899+1), int((104.483069-103.25635)//0.01141+1)) #167, 108
    dqn = DQN(dr, fl)
    SIZE = (108, 167, len(fl), 1657)
    ep_rewards = dqn.qlearning(SIZE)
    print(ep_rewards)
