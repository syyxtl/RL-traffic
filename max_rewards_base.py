import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_s import BModel
from model_s import DQN


def test(dqn):
    idx = 0
    state = [0, 0, 0]
    rewards = 0
    blanks = 0
    while idx < dqn.end:
        
        dataset = dqn.get_item_f(idx)
        lend = dataset["lens"]
        ACTION_FULL = [i for i in range(lend)]
        lined = None
        reward = 0
        if idx == 0:
            action = np.random.choice(ACTION_FULL)
            lined = dqn.get_item_l(dataset, action)
            idx = lined[1]
            state = [lined[4], lined[5], idx]
            reward = lined[-1]
        else:
            ACTION = []
            for i in range(lend):
                tdata = dqn.get_item_l(dataset, i)
                td = dqn.call_d(state, (tdata[2], tdata[3]))
                if td < 5:
                    ACTION.append(i)
            if len(ACTION) != 0:
                maxx = -999999999
                action_choice = 0
                for index in ACTION:
                    if ACTION_FULL[index] > maxx:
                        maxx = ACTION_FULL[index]
                        action_choice = index
                action = action_choice
            else:
                action = -1
            # print(idx, state, action)
            if action != -1:
                lined = dqn.get_item_l(dataset, action)
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

    rewards = 0
    blanks = 0
    for i in range(100):
        reward, blank = test(dqn)
        rewards += reward
        blanks += blank
        print(rewards//(i+1), blanks//(i+1))
    print(rewards//100, blanks//100)
    # 26191