import os
import json
import pickle
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import ticker
from matplotlib import pyplot as plt
from matplotlib import dates as mdates


QUERY_NUM = 10

def log2action(filename):
	with open(filename) as f:
		lines = f.readlines()
	nlines = []
	for _, line in enumerate(lines):
		line = line.strip()
		lb_pos = line[1:].find('[') + 1
		if lb_pos == 0:
			nlines.append(line)
		else:
			while lb_pos > 0:
				nlines.append(line[:lb_pos].strip())
				line = line[lb_pos:]
				lb_pos = line[1:].find('[') + 1
			nlines.append(line)
	lines = nlines
	ts = []
	actions = list()
	monitors = dict()
	resources = []
	for line in lines:
		if not line.startswith('[') and len(ts) == 0:
			continue
		while line.startswith('['):
			line = line.strip()
			ts.append(float(line[1:15]))
			line = line[17:]
		if line.startswith('t'):
			thread = int(line[1:line.find(' ')])
			line = line[line.find(' ')+1:]
			if line.startswith('start sleeping') or line.startswith('end sleeping'):
				continue
			elif line.startswith('start'):
				type = 'start'
			else:
				type = 'end'
			query = int(line[line.find('id')+4:line.find(',')])
			worker_pos = line.find('max_parallel_workers_per_gather')
			worker = int(line[worker_pos+32:worker_pos+33]) if worker_pos >= 0 else -1
			actions.append((ts.pop(0), thread, type, query, worker))
		elif line.startswith('m'):
			line = line[line.find(':')+2:]
			records = line.split('; ')
			for r in records:
				items = r.split(', ')
				name = items[0]
				cpu = float(items[1][items[1].find(' ')+1: items[1].find('/')])
				cpu_limit = int(items[1][items[1].find('/')+1:])
				mem = int(items[2][items[2].find(' ')+1: items[2].find('/')])
				mem_limit = int(items[2][items[2].find('/')+1:])
				if name in monitors.keys():
					monitors[name].append((ts.pop(0), cpu, mem))
				else:
					monitors[name] = [(ts.pop(0), cpu, mem)]
		elif line.startswith('resource'):
			line = line[line.find(':')+2:]
			records = line.split(', ')
			cpu, vmem, smem = map(lambda s:float(s[s.find(' ')+1:-1]), records)
			resources.append((ts.pop(0), cpu, vmem, smem))
		elif line.startswith('division'):
			ts.pop(0)
	return actions, monitors, resources


def action2query(actions):
	min_time = actions[0][0]
	queries, raw_queries, finish_num, counter = [], [], [], 0
	for i, start in enumerate(actions):
		if start[2] == 'end':
			counter += 1
			finish_num.append((datetime.fromtimestamp(start[0]), counter))
			continue
		for ii in range(i+1, len(actions)):
			if actions[ii][3] == start[3]:
				end = actions[ii]
				break
		queries.append((start[1], start[3], start[0] - min_time, end[0] - start[0]))
		raw_queries.append((start[1], start[3], datetime.fromtimestamp(start[0]), datetime.fromtimestamp(end[0])))
	throughputs = list(map(lambda x:(x[0], x[1]/counter), finish_num))
	return queries, raw_queries, throughputs


def query2time(queries):
	q_exec_time = [[] for _ in range(QUERY_NUM)]
	for query in queries:
		q_exec_time[query[1]-1].append(query[3])
	q_mean_time = [np.mean(q) for q in q_exec_time]
	return q_exec_time, q_mean_time


def plot_gantt(df: pd.DataFrame, tdf: pd.DataFrame, save_path: str, enable_throughput: bool, enable_monitor: bool, mdf: pd.DataFrame, figsize: tuple = (16, 5)):
    if enable_monitor:
        fig, axes = plt.subplots(4, 1, figsize=(16,20), dpi=500)
        gnt, thpt, mon_cpu, mon_mem = axes
    elif enable_throughput:
        fig, axes = plt.subplots(2, 1, figsize=(16,10), dpi=500)
        gnt, thpt = axes
    else:
        fig, gnt = plt.subplots(1, 1, figsize=figsize, dpi=500)
    
    gnt.set_xlabel('Duration')
    gnt.set_ylabel('Threads')

    start_time = df['start_time'].min()
    end_time = df['end_time'].max() + timedelta(seconds=2)
    interval = int((end_time - start_time).total_seconds() // 20)
    interval = 1 if interval == 0 else interval
    # end_time = df['start_time'].min() + timedelta(seconds=70)

    # gnt.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    gnt.xaxis.set_major_formatter(lambda x,pos:'{:.0f}'.format(x*24*3600-start_time.timestamp()))
    gnt.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
    gnt.set_xlim(left=start_time, right=end_time)
    gnt.invert_yaxis()

    for _, row in df.iterrows():
        width = row['end_time'] - row['start_time']
        gnt.barh(y=row['thread'], width=width, left=row['start_time'], height=0.5 , edgecolor="black")
        # gnt.barh(y=row['thread'], width=width, left=row['start_time'], height=0.5 , label='query{}'.format(row['unique_id']), edgecolor="black")
        xcenters = row['start_time'] + width / 2
        gnt.text(x=xcenters, y=row['thread'], s=str(row['unique_id']), ha='center', va='center')
    
    if enable_throughput:
        thpt.xaxis.set_major_formatter(lambda x,pos:'{:.0f}'.format(x*24*3600-start_time.timestamp()))
        thpt.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
        thpt.set_xlim(left=start_time, right=end_time)
        thpt.set_xlabel('Duration')
        thpt.set_ylabel('Throughput')
        thpt.plot(tdf['time'].tolist(), tdf['throughput'].tolist())

    if enable_monitor:
        # mon.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        mon_cpu.xaxis.set_major_formatter(lambda x,pos:'{:.0f}'.format(x*24*3600-start_time.timestamp()))
        mon_cpu.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
        mon_cpu.set_xlim(left=start_time, right=end_time)
        mon_cpu.set_xlabel('Duration')
        mon_cpu.set_ylabel('CPU Usage')
        for k, v in mdf.items():
            mon_cpu.plot(v['time'], v['cpu'], label=k)
        mon_cpu.legend()

        mon_mem.xaxis.set_major_formatter(lambda x,pos:'{:.0f}'.format(x*24*3600-start_time.timestamp()))
        mon_mem.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
        mon_mem.set_xlim(left=start_time, right=end_time)
        mon_mem.set_xlabel('Duration')
        mon_mem.set_ylabel('Memory Usage')
        for k, v in mdf.items():
            mon_mem.plot(v['time'], v['mem'], label=k)
        mon_mem.legend()

    if save_path:
        plt.savefig(save_path, format='jpg', pad_inches=0)