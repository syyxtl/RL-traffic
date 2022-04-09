import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_s import BModel
from model_s import DQN
from match_file import *
##################################################### base model ####################################################
def load_model(train_num):
    eval_network = BModel().cuda()
    state_dict = torch.load("/home/ssd1/shenzhiyong/npy/eval_model_" + str(train_num) + ".pth")
    eval_network.load_state_dict(state_dict)
    return eval_network

P = 0
D = 0
if __name__ == '__main__':
    # dataset, 0-287_out.txt   
    dr = "out5m_req/"
    fl = [str(i)+"_out" for i in range(287)]
    dqn = DQN(dr, fl)
    dqn.reset()

    model = load_model(300)

    rewards = 0
    blanks = 0
    gamma = 0.95
    Users = []
    Drivers = []
    
    for idx in range(287):
        print("===================================", idx, "===================================")
        dataset = dqn.get_item_f(idx)
        lend = dataset["lens"]
        Users = []
        for i in range(lend):
            # [ID, stime, etime, upr, upc, downr, downc, true_d, true_v, rewards]
            line = dqn.get_item_l(dataset, i)
            User = [line[2], line[3], idx, line[-1]]
            Users.append(User)
        
        matches = []
        matches_ = []
        if idx != 0:    
            for _Dstate in Drivers:
                Dstate = (_Dstate[0], _Dstate[1], _Dstate[2])
                ACTION = np.zeros(lend)
                for user in range(len(Users)):
                    td = dqn.call_d((Dstate[0], Dstate[1]), (Users[user][0], Users[user][1]))
                    if int(td) < 5:
                        ACTION[user] = 1
                if np.sum(ACTION) == 0:
                    action = 0
                    Outputs = ACTION
                else:
                    INput = torch.tensor(np.array([Dstate]), dtype=torch.float32).cuda()
                    model.eval()
                    Outputs = model(INput).cpu().detach().numpy()[0][0:lend]
                    vo = [r[-1] for r in Users]
                    Outputs = F.softmax( torch.unsqueeze(torch.tensor(Outputs), 0), dim=1).numpy()
                    Outputs = Outputs * ACTION  * np.array(vo) * 1e20
                    Outputs = torch.clip(torch.tensor(Outputs, dtype=torch.int64), 0, 1).numpy()
                    Outputs = Outputs[0].astype(int)
                matches.append(Outputs.copy())
                matches_.append(Outputs.copy())
            # for _ in range(len(matches[0]) - len(matches) ):
            #     tmp = np.zeros(lend).astype(int)
            #     matches.append(tmp)
            for i in range(len(matches)):
                for j in range(len(matches[0])):
                    if int(matches[i][j]) == int(0):
                        matches[i][j] = 999
                        matches_[i][j] = 999
            # print(('lowest cost=%d' % total_cost))

                    # action = np.argmax(Outputs) 
                    # P = dqn.get_item_l(dataset, action)
            

        Drivers = []
        
        for i in range(lend):
            # [ID, stime, etime, upr, upc, downr, downc, true_d, true_v, rewards]
            line = dqn.get_item_l(dataset, i)
            Driver = [line[4], line[5], idx, line[-1]]
            Drivers.append(Driver)

        if idx >= 2:
            m = Munkres()
            print(len(matches), len(matches[0]) )
            if len(matches) > len(matches[0]):
                indexes = m.compute(matches[0:len(matches[0])])
            else:
                indexes = m.compute(matches)
            total_cost = 0
            total_dist = 0
            for r, c in indexes:
                x = matches_[r][c]
                if int(x) != 1:
                    x = 0
                total_cost += x
                if int(x) == 1:
                    total_dist += dqn.call_d((Drivers[r][0], Drivers[r][1]), (Users[c][0], Users[c][1]))
            P += total_cost*1.0/lend
            D += total_dist*1.0/total_cost
            print(total_cost, lend, total_dist, "percent:", total_cost*1.0/lend, "distance:", total_dist*1.0/total_cost)
    
    print(('(P:%.2f, D:%.2f)' % (P*1.0/285, D*1.0/285)))