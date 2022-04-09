import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_s import BModel
from model_s import DQN


score = 0
def load_model(train_num):
    eval_network = BModel().cuda()
    state_dict = torch.load("/home/ssd1/shenzhiyong/npy/eval_model_" + str(train_num) + ".pth")
    eval_network.load_state_dict(state_dict)
    return eval_network

def test(dqn, model):
    idx = 0
    state = [0, 0, 0]
    rewards = 0
    blanks = 0
    while idx < dqn.end:
        
        dataset = dqn.get_item_f(idx)
        lend = dataset["lens"]
        ACTION_FULL = [i for i in range(lend)]
        lined = None
        if idx == 0:
            action = np.random.choice(ACTION_FULL)
            lined = dqn.get_item_l(dataset, action)
        else:
            ACTION = np.zeros(lend)
            for i in range(lend):
                tdata = dqn.get_item_l(dataset, i)
                td = dqn.call_d(state, (tdata[2], tdata[3]))
                if td < 5:
                    ACTION[i] = 1
            if np.sum(ACTION) == 0:
                action = -1
            else:
                inputs = torch.tensor(np.array([state]), dtype=torch.float32).cuda()
                model.eval()
                outputs = model(inputs).cpu().detach().numpy()[0][0:lend]
                outputs = outputs * ACTION
                action = np.argmax(outputs)
                lined = dqn.get_item_l(dataset, action)
       
        if action != -1:
            idx = lined[1]
            state = [lined[4], lined[5], idx]
            reward = lined[-1]
            if lined[1] < lined[0]:
                break
            if lined[1] == lined[0]:
                idx = idx + 1
        else:
            idx = idx + 1
            blanks = blanks + 1
            state = [state[0], state[1], idx]
            reward = 0

        rewards += reward
        
    return rewards, blanks

if __name__ == '__main__':
    # dataset, 0-287_out.txt   
    dr = "out5m_req/"
    fl = [str(i)+"_out" for i in range(287)]
    dqn = DQN(dr, fl)
    dqn.reset()
    exp = [100, 200]
    rewards_sum = []
    blanks_sum = []
    for e_ in exp:
        model = load_model(e_)
        rewards = 0
        blanks = 0
        for j in range(100):
            reward, blank = test(dqn, model)
            rewards += reward
            blanks += blank
        
        rewards_sum.append(rewards // 100)
        blanks_sum.append(blanks // 100)
        print(e_, rewards, blanks)
    
    with open("rewards.txt", "a") as wt:
        for rs in range(rewards_sum):
            wt.write("exp: " + str(exp[rs]) + "  rewards: " +  str(int(rewards_sum[rs])) + "\n"  )
        for rs in range(rewards_sum):
            wt.write("exp: " + str(exp[rs]) + "  blanks: " +  str(int(blanks_sum[rs])) + "\n"  )
    
