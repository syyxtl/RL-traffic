import time
import copy
import numpy as np
import os

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

inputs = [
	"order_20161101","order_20161102","order_20161103","order_20161104","order_20161105","order_20161106","order_20161107",
	"order_20161108","order_20161109","order_20161110","order_20161111","order_20161112","order_20161113","order_20161114",
	"order_20161115","order_20161116","order_20161117","order_20161118","order_20161119","order_20161120","order_20161121",
	"order_20161122","order_20161123","order_20161124","order_20161125","order_20161126","order_20161127","order_20161128",
	"order_20161129","order_20161130",
]

def get_filelist(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            get_filelist(newDir, Filelist) 
    return Filelist

if __name__ == '__main__':
    # dataset, 0-287_out.txt   
    root = "out5m_req/"
    gen_pr = False
    if gen_pr:
        for ins in inputs:
            print(ins, "....")
            initalp = [[0 for i in range(200)] for j in range(200)] # 108 167, price
            initalr = [[0 for i in range(200)] for j in range(200)] # 108 167, request

            path = root + ins + "/"
            lens = len(get_filelist(path, []))

            fl = [str(i)+"_out" for i in range(lens-1)]
            dqn = DQN(path, fl)

            for idx in range(lens-1):
                dataset = dqn.get_item_f(idx)
                lend = dataset["lens"]
                Users = []
                for i in range(lend):
                    # [ID, stime, etime, upr, upc, downr, downc, true_d, true_v, rewards]
                    line = dqn.get_item_l(dataset, i)
                    r, c, p = line[3], line[4], line[-1]
                    if r > 200 or c > 200:
                        continue
                    initalp[r][c] += p
                    initalr[r][c] += 1
            
            wroot = "wt/"
            with open(wroot + str(ins) + "_p.txt", "a") as wt:
                for r in range(200):
                    for c in range(200):
                        wt.write( "[" + str(r) + "," + str(c) + "," + str(initalp[r][c]) + "]," )
                        wt.write( "\n" )

            with open(wroot + str(ins) + "_r.txt", "a") as wt:
                for r in range(200):
                    for c in range(200):
                        wt.write( "[" + str(r) + "," + str(c) + "," + str(initalr[r][c]) + "]," )
                        wt.write( "\n" )
    
    resize = True
    if resize == True:
        for ins in inputs:
            data_root =  "/home/ssd3/shenzhiyong/workspace/shenzhiyong/code/RL-traffic/wt/"
            files = ins + "_r.txt"
            datas = data_root + files
            initalp = [[0 for i in range(20)] for j in range(20)]
            with open(datas, "r") as rd:
                for line in rd:
                    line = line.replace("[","").replace("]","")
                    r, c, p = line.split(",")[0], line.split(",")[1], line.split(",")[2]
                    r = int(r)
                    c = int(c)
                    p = int(float(p))
                    initalp[r//10][c//10] += p

            data_root =  "/home/ssd3/shenzhiyong/workspace/shenzhiyong/code/RL-traffic/wt_20r/"
            datas = data_root + files
            with open(datas, "a") as wt:
                for r in range(6, 18):
                    for c in range(0, 16):
                        wt.write( "[" + str(r) + "," + str(c) + "," + str(initalp[r][c]) + "]," )
                        wt.write( "\n" )

    else:
        data_rootp =  "/home/ssd3/shenzhiyong/workspace/shenzhiyong/code/RL-traffic/wt_20/"
        data_rootr =  "/home/ssd3/shenzhiyong/workspace/shenzhiyong/code/RL-traffic/wt_20r/"
        filesp = "order_20161101" + "_p.txt"
        filesr = "order_20161101" + "_r.txt"
        data_rootp_i = data_rootp + filep
        data_rootr_i = data_rootr+ filer
        inital = [[0 for i in range(20)] for j in range(20)]
        with open(datas, "a") as wt:
                for r in range(6, 18):
                    for c in range(0, 16):