import os
import csv
import numpy as np
import pandas as pd

def read_one_csv(file):
	orderLabels = ["ID", "startTime", "endTime", "upLon", "upLat", "downLon", "downLat", "rewards"]
	orderSep = ","
	data = pd.read_csv(file, names=orderLabels, sep=orderSep, index_col=False)
	lens = len(data)
	return {"data": data, "lens": lens}
	# test: lists = read_one_csv("./data/total_ride_request/order_20161101")

def read_all_csv(file_dir):
	filelist = None
	for root, dirs, files in os.walk(file_dir):
		filelist = files

	datas = []
	print("start read file...")
	for findex in range(len(filelist)):
		tpath = os.path.join(str(file_dir), str(filelist[findex]))
		print("		read the file " +  str(tpath))
		tmp = read_one_csv(tpath)
		datas.append(tmp)
	print("finished read file...")
	return datas
	# test: data = read_all_csv("./data/total_ride_request")

def getline(data, i):
	ID = data.iloc[i]["ID"]
	startTime = data.iloc[i]["startTime"]
	endTime = data.iloc[i]["endTime"]
	upLon = data.iloc[i]["upLon"]
	upLat = data.iloc[i]["upLat"]
	downLon = data.iloc[i]["downLon"]
	downLat = data.iloc[i]["downLat"]
	rewards = data.iloc[i]["rewards"]
	return [ID, startTime, endTime, upLon, upLat, downLon, downLat, rewards]
	#       0       1         2       3      4       5        6        7

def groupby(data, colume="ID"):
	return data.groupby(colume)

	
import time
def stamp_to_time(stamp):
	timeArray = time.localtime(stamp)
	t = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
	return t

# 0.01141 = 1m jd bigger
# 0.00899 = 1m wd smaller
def gen_grid(data):
	grid = np.array(data)
	mins = grid.min(0)
	maxs = grid.max(0)
	# geo grid
	geomins = [mins[3], mins[4], mins[5], mins[6]]
	geomaxs = [maxs[3], maxs[4], maxs[5], maxs[6]]
	minjd = min(geomins[0], geomins[2])
	minwd = min(geomins[1], geomins[3])
	maxjd = min(geomaxs[0], geomaxs[2])
	maxwd = min(geomaxs[1], geomaxs[3])
	print("generate grid in : [", minwd, minjd, "] to [", maxwd, maxjd, "]")
	# time grid
	# [00:00:00, 23:59:59, 00:05:00]
	return [ [minjd, maxjd, 0.01141], [minwd, maxwd, 0.00899] ]

def fid_geo_grid(grid, row, col):
	ldr, ldc = grid[0][0], grid[1][0]
	row = (row - ldr) // grid[0][2]
	col = (col - ldc) // grid[1][2]
	return int(row), int(col)

def fid_time_grid(data):
	stamps, stampe = data[1], data[2]
	stamps = stamp_to_time(stamps)
	stampe = stamp_to_time(stampe)
	hours = int(stamps[-8:-6])
	mintes = int(stamps[-5:-3])
	houre = int(stampe[-8:-6])
	mintee = int(stampe[-5:-3])
	return (hours*60+mintes)//5, (houre*60+mintee)//5

def gen_fakereq_data(data, path, root="out5m_req/", output="_out.txt"):
	root = root + path + "/"
	if not os.path.exists(root):
		os.makedirs(root)
	cont = data["data"]
	lens = data["lens"]
	grid = gen_grid(cont)
	for ind in range(lens):
		data = getline(cont, ind)
		rcs = fid_geo_grid(grid, data[3],data[4])
		rce = fid_geo_grid(grid, data[5],data[6])
		t = fid_time_grid(data)
		line = [data[0], t[0], t[1], rcs[0], rcs[1], rce[0], rce[1], data[7]]
		d, v = call_dv( (rcs[0], rcs[1], t[0]), (rce[0], rce[1], t[1]))
		# if root.split("_")[-1] == "req":
		# 	splits = t[0]
		# else:
		splits = t[1]
		with open(root + str(splits) + output, "a") as wt:
			wt.write(
				str(data[0])+","+str(t[1])+","+str(t[0])+","+ \
				str(rcs[0])+","+str(rcs[1])+","+str(rce[0])+","+\
				str(rce[1])+","+str(d)+","+str(v)+","+str(data[7])+"\n"
			)
		if ind % 2500 == 0:
			print(ind, "/", lens)
	print("finished gen req data")

def call_dv(ph1, ph2): # distance and volume
	# (row, col, time)
	d = abs(ph1[0]-ph2[0]) + abs(ph1[1]-ph2[1])
	v = abs(ph1[2]-ph2[2]) // (d+1) + 1
	return d, v 

gen_fake = True
root = "total_ride_request/"
inputs = [
	"order_20161101","order_20161102","order_20161103","order_20161104","order_20161105","order_20161106","order_20161107",
	"order_20161108","order_20161109","order_20161110","order_20161111","order_20161112","order_20161113","order_20161114",
	"order_20161115","order_20161116","order_20161117","order_20161118","order_20161119","order_20161120","order_20161121",
	"order_20161122","order_20161123","order_20161124","order_20161125","order_20161126","order_20161127","order_20161128",
	"order_20161129","order_20161130",
]
if __name__ == '__main__':
	for ins in inputs:
		path = root + ins
		data = read_one_csv(path)
		if gen_fake:
			gen_fakereq_data(data, ins)
