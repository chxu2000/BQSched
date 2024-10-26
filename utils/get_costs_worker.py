import pandas as pd, psycopg2, time, json
from args import args
from tqdm import tqdm
from configparser import ConfigParser


# Get config
conf = ConfigParser()
conf.read('config.ini', encoding='utf-8')
queries_path = f'envs/scheduler/cache/{conf["database"]["host"]}/{conf["database"]["database"]}/queries{"" if int(conf["database"]["query_scale"]) == 1 else conf["database"]["query_scale"] + "X"}.csv'
queries = pd.read_csv(queries_path).values.tolist()

workers = [0, 2]
costs_worker = dict()
for w in workers:
	costs_worker[f'max_parallel_workers_per_gather={w}'] = [[] for _ in range(len(queries))]

conn_string = 'host={} port={} dbname={} user={} password={}'.format(
	conf['database']['host'], conf['database']['port'], conf['database']['database'], 
	conf['database']['user'], conf['database']['password']
)
conn = psycopg2.connect(conn_string)
conn.set_session(autocommit=True)
cursor = conn.cursor()

for _ in range(5):
	for w in workers:
		cursor.execute(f'set max_parallel_workers_per_gather={w};')
		for i, q in enumerate(tqdm(queries)):
			start_time = time.time()
			try:
				cursor.execute(q[3])
			except Exception as e:
				pass
			end_time = time.time()
			costs_worker[f'max_parallel_workers_per_gather={w}'][i].append(end_time - start_time)
	with open('query_cluster/costs_worker.json', 'w') as f:
		json.dump(costs_worker, f, indent=4)

conn.close()