import json, numpy as np, pickle, sys
sys.path.append(".")
from tqdm import tqdm
from utils.log_analyzer import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from modes.args.train_on_clusters import args
from scipy.cluster.hierarchy import dendrogram
from configparser import ConfigParser


QUERY_SCALE = 5
QUERY_NUM = 99 * QUERY_SCALE
N_CLUSTERS = 99
# Get config
conf = ConfigParser()
conf.read(f'modes/config/train_on_clusters.ini', encoding='utf-8')

def get_actions():
	logdir = f'../../scheduler/log/{conf["database"]["host"]}/tpcds1X/query_scale=5/random_worker/max_worker=58'
	actions = []
	# for logpath in ['outs/tmp.out']:
	for logfile in tqdm(os.listdir(logdir)[:100]):
		if not logfile.endswith('.log'):
			continue
		logpath = os.path.join(logdir, logfile)
		actions += log2action(logpath)[0]
	return actions

def get_ov_t():
	actions = get_actions()
	counter = 0
	con_queries = []
	ov_ij = [[[] for __ in range(QUERY_NUM+1)] for _ in range(QUERY_NUM+1)]
	ti_j = [[[] for __ in range(QUERY_NUM+1)] for _ in range(QUERY_NUM+1)]
	query_times = np.array([[-1] * (QUERY_NUM + 1)] * (len(actions) // (QUERY_NUM * 2) + 1), dtype=float)
	for i, action in enumerate(tqdm(actions)):
		if i % (2 * QUERY_NUM) == 0:
			counter += 1
		if action[2] == 'start':
			con_queries.append(action)
		elif action[2] == 'end':
			qids = [a[3] for a in con_queries]
			finish_index = qids.index(action[3])
			query_times[counter][action[3]] = action[0] - con_queries[finish_index][0]
			for i, cq in enumerate(con_queries):
				if cq[3] != action[3]:
					ov = action[0] - max(cq[0], con_queries[finish_index][0])
					ov_ij[min(cq[3], action[3])][max(cq[3], action[3])].append(ov)
					ti_j[cq[3]][action[3]].append(query_times[counter][action[3]])
					ti_j[action[3]][cq[3]].append(-counter)
			del con_queries[finish_index]
		else:
			raise Exception()
	return ov_ij, ti_j, query_times

def get_ov_sim():
	# ov_ij, ti_j, query_times = get_ov_t()
	# with open('query_cluster/ov_t_qt.pickle', 'wb') as f:
	# 	pickle.dump({'ov_ij': ov_ij, 'ti_j': ti_j, 'query_times': query_times}, f)

	with open('query_cluster/ov_t_qt.pickle', 'rb') as f:
		ov_t_qt = pickle.load(f)
	ov_ij, ti_j, query_times = ov_t_qt['ov_ij'], ov_t_qt['ti_j'], ov_t_qt['query_times']

	n, empty = 10, 0
	mean_query_times = np.mean(query_times[1:], axis=0)
	sim_ij = np.zeros((QUERY_NUM + 1, QUERY_NUM + 1), dtype=np.float32)
	for qi in tqdm(range(1, QUERY_NUM + 1)):
		for qj in range(qi + 1, QUERY_NUM + 1):
			if len(ov_ij[qi][qj]) == 0:
				empty += 1
			else:
				for i in range(len(ov_ij[qi][qj])):
					if ti_j[qi][qj][i] < 0:
						ti_j[qi][qj][i] = query_times[-ti_j[qi][qj][i]][qj]
					if ti_j[qj][qi][i] < 0:
						ti_j[qj][qi][i] = query_times[-ti_j[qj][qi][i]][qi]
					o_ij = ov_ij[qi][qj][i] / ti_j[qi][qj][i]
					o_ji = ov_ij[qi][qj][i] / ti_j[qj][qi][i]
					a_ij = 1 - ti_j[qi][qj][i] / mean_query_times[qj]
					a_ji = 1 - ti_j[qj][qi][i] / mean_query_times[qi]
					sim_ij[qi][qj] += (o_ij * a_ij * np.sqrt(mean_query_times[qj]) + \
						o_ji * a_ji * np.sqrt(mean_query_times[qi])) / \
						(np.sqrt(mean_query_times[qi]) + np.sqrt(mean_query_times[qj]))
					# sim_ij[qi][qj] += ((o_ij + n - 1) * a_ij * np.sqrt(mean_query_times[qj]) + \
					# 	(o_ji + n - 1) * a_ji * np.sqrt(mean_query_times[qi])) / \
					# 	(np.sqrt(mean_query_times[qi]) + np.sqrt(mean_query_times[qj]))
					# sim_ij[qi][qj] += (ov_ij[qi][qj][i] + n - 1) * \
					# 	(a_ij * np.sqrt(mean_query_times[qj]) + a_ji * np.sqrt(mean_query_times[qi])) / \
					# 	(np.sqrt(mean_query_times[qi]) + np.sqrt(mean_query_times[qj]))
				sim_ij[qi][qj] = sim_ij[qi][qj] / len(ov_ij[qi][qj])
				# sim_ij[qi][qj] = sim_ij[qi][qj] / (n * len(ov_ij[qi][qj]))
	np.savez('query_cluster/sim_array.npz', sim_ij)
	print(empty / (QUERY_NUM * QUERY_NUM - QUERY_NUM))
	return sim_ij

def get_ov_cluster():
	sim_ij = np.load('query_cluster/sim_array.npz')['arr_0'][1:, 1:][:QUERY_NUM, :QUERY_NUM]
	# sim_ij = get_ov_sim()[1:, 1:]
	sim_max, sim_min = sim_ij.max(), sim_ij.min()
	for i in tqdm(range(QUERY_NUM)):
		# sim_ij[i][i] = np.finfo(np.float32).min
		for j in range(0, i):
			if sim_ij[j][i] > 0:
				sim_ij[i][j] = sim_ij[j][i]
			else:
				sim_ij[i][j] = sim_ij[j][i] = sim_min - 1
	dis_ij = -sim_ij + sim_max
				
	model = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average')
	cluster_result = model.fit_predict(dis_ij)

	long_qpos, long_qposs = np.array([2, 5, 14, 21, 59], dtype=np.int64) - 1, np.array([], dtype=np.int64)
	for i in range(QUERY_SCALE):
		long_qposs = np.concatenate((long_qposs, long_qpos + i * 99))
	query_workers = np.zeros((QUERY_NUM, ), dtype=np.int64)
	query_workers[long_qposs] = 1
	np.savez(f'query_cluster/result{QUERY_SCALE}X{N_CLUSTERS}_oc.npz', cluster_result, query_workers)

def get_costs_worker():
	with open('query_cluster/costs_worker.json', 'r') as f:
		costs_worker = json.load(f)
	for k, _ in costs_worker.items():
		for i, l in enumerate(costs_worker[k]):
			costs_worker[k][i] = np.mean(l)
	return costs_worker

def get_worker_cluster():
	costs_worker = get_costs_worker()
	keys = list(costs_worker.keys())
	query_features = np.array([[costs_worker[keys[0]][i] - costs_worker[keys[1]][i], (costs_worker[keys[0]][i] - costs_worker[keys[1]][i]) / 
							 costs_worker[keys[0]][i]] for i in range(len(costs_worker[keys[0]]))])
	query_features = normalize(query_features, axis=0)
	if args.pretrain_enable_embedding:
		query_embedding = np.load('nets/embeddings10.npy')
		query_features = np.concatenate((query_features, query_embedding), axis=1)
	cluster_result = AgglomerativeClustering(n_clusters=99).fit_predict(query_features)
	cluster_features = np.array([np.mean(query_features[cluster_result == i], axis=0) for i in range(99)])
	np.savez('query_cluster/cluster_result_emb_wc.npz', cluster_result, cluster_features)

def main():
	get_ov_cluster()
	# get_worker_cluster()

main()