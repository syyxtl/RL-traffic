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

filelist = [str(i)+"_out" for i in range(287)]

def get_item_f(dr, id):
    filename = str(id) + "_out"
    if filename not in filelist:
        return None
    else:
        filename = os.path.join(dr,filename + ".txt")
        orderLabels = ["ID", "stime", "etime", "upr", "upc", "downr", "downc", "true_d", "true_v", "rewards"]
        orderSep = ","
        data = pd.read_csv(filename, names=orderLabels, sep=orderSep, index_col=False)
        lens = len(data)
        return {"data":data, "lens":lens}

def get_item_l(data, id):
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


if __name__ == '__main__':
    # dataset, 0-287_out.txt   
    dr = "out5m_req/"
    fl = [str(i)+"_out" for i in range(287)]
    dataset = get_item_f(dr, 0)
    lens = dataset["lens"]
    for i in range(lens):
        data = get_item_l(dataset, i)
        rewards = data[6]
        true_d = data[8]
        st = data[0]
        et = data[1]
        if st>et: 
            continue
        t = et -st
        print("@r: ",rewards, " @td: ",true_d, " @at: ", et-st, " @avr: ", round(rewards/true_d, 3))


