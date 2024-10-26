import os, sys, time, json, pickle, numpy as np, psutil, torch
from datetime import datetime
from threading import Event
from concurrent.futures import ThreadPoolExecutor, wait
from envs.scheduler.database.database import Database
from configparser import ConfigParser
from envs.scheduler.query_list import QueryList
from envs.scheduler.table_list import TableList
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from simulator.net.transformer import ConcurrentQueryFormer, SharedQueryFormer


def get_config(path):
    conf = ConfigParser()
    conf.read(path, encoding='utf-8')
    return conf


def print_time(hint, file=None):
    if file is None:
        # print(f'[{time.time():.3f}] {hint}', flush=True)
        pass
    else:
        print(f'[{time.time():.3f}] {hint}', file=file, flush=True)


def thread_query(i, conn, query, runtime_log=None):
    err_flag = False
    if query[0][0] == -1:
        time.sleep(query[0][1])
    else:
        with conn.cursor() as cursor:
            cursor = conn.cursor()
            if query[1]:
                cursor.execute(query[1])
            print_time("t%d start executing query (id: %s, filename: %s, prefix: %s)" % (i, query[0][0], query[0][1], query[1]), file=runtime_log)
            try:
                cursor.execute(query[0][3])
            except Exception as e:
                print_time(e, file=runtime_log)
                if e.pgcode != '22012':
                    err_flag = True
            print_time("t%d end executing query (id: %s, filename: %s, postfix: %s)" % (i, query[0][0], query[0][1], query[2]), file=runtime_log)
            if query[2]:
                cursor.execute(query[2])
    return i, query[0][0], err_flag, time.time()


def thread_sleep(i, t):
    time.sleep(t)
    return i, -1


def thread_monitor(conn, event, interval=1, runtime_log=None):
    cursor = conn.cursor()
    while not event.is_set():
        cursor.execute('select * from PGXC_RESPOOL_RESOURCE_INFO where rpname=\'default_pool\';')
        res = list(cursor.fetchall())
        hint = 'monitor: '
        for stat in res:
            hint += f'{stat[0]}, cpu {stat[11]}/{stat[12]}, mem {stat[13]}/{stat[15]}; '
        hint = hint[:-2]
        print_time(hint, file=runtime_log)
        time.sleep(interval)
    cursor.close()


def get_local_resource():
    return {
        "cpu_times_percent": psutil.cpu_times_percent(),
        "virtual_memory": psutil.virtual_memory(),
        "swap_memory": psutil.swap_memory()
    }


def get_remote_resource(conn_ssh):
    command = """def rpcexecute():
        import psutil
        return {
            "cpu_times_percent": psutil.cpu_times_percent(),
            "virtual_memory": psutil.virtual_memory(),
            "swap_memory": psutil.swap_memory()
        }"""
    conn_ssh.execute(command)
    remote_exec = conn_ssh.namespace['rpcexecute']
    return remote_exec()


class QueryScheduler():

    def __init__(self, conf, strategy=None, max_worker=None, qid_list=None, query_num=None, runtime_log=None,
                 enable_resource=False, conn_ssh=None, local_resource=False, args=None):
        self.args = args
        self.conf = conf
        self.qid_list = qid_list
        self.strategy = self.conf['scheduler']['strategy'] if strategy is None else strategy
        self.max_worker = self.conf.getint('scheduler', 'max_worker') if max_worker is None else max_worker
        self.enable_monitor = self.conf.getboolean('scheduler', 'enable_monitor')
        self.use_simulator = self.conf.getboolean('database', 'use_simulator')
        self.simulator_model = self.conf['database']['simulator_model']
        self.database = Database(self.conf)  # 存放元数据啥的
        self.conn_list = []
        for _ in range(self.max_worker):
            self.conn_list.append(self.database.connect() if not self.use_simulator else None)
        if self.enable_monitor and not self.use_simulator:
            self.monitor_conn = self.database.connect(database='postgres')
        # printtime("Finish Initialize Connections")
        self.queries = sorted(self.database.get_queries(), key=lambda x:x[0])
        if not query_num is None:
            self.queries = self.queries[:query_num]
        self.query_num = len(self.queries)
        self.query_scale = self.conf.getint('database', 'query_scale')
        self.query_cluster = self.conf.getboolean('database', 'query_cluster')
        # self.table_list = TableList(self.database.conn) if not self.use_simulator else None
        # self.stdout_bak = sys.stdout
        # flog_dir = 'log/{}/{}/{}/max_worker={}'.format(self.conf['database']['host'], self.conf['database']['database'], self.strategy, self.max_worker)
        # if not os.path.exists(flog_dir):
        #     os.makedirs(flog_dir)
        # self.flog = open('{}/{}.log'.format(flog_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), 'w')
        # self.flog = open('query_exec.log', 'w')
        # sys.stdout = self.flog
        self.runtime_log = runtime_log
        if self.use_simulator:
            if self.simulator_model == 'lgbm':
                self.one_hot_enc = OneHotEncoder().fit([[i] * self.max_worker for i in range(100)])
                with open(self.conf['database']['scaler_path'], 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(self.conf['database']['clf_model_path'], 'rb') as f:
                    self.clf_model = pickle.load(f)
                with open(self.conf['database']['reg_model_path'], 'rb') as f:
                    self.reg_model = pickle.load(f)
            elif self.simulator_model == 'con_query_former':
                self.time_scale = self.conf.getint('database', 'time_scale')
                self.enable_cluster_embedding = self.conf.getboolean('database', 'enable_cluster_embedding')
                self.use_rank_loss = self.conf.getboolean('database', 'use_rank_loss')
                self.con_query_former = torch.load(f'envs/scheduler/cache/{self.conf["database"]["host"]}/{self.conf["database"]["database"]}/cqf_w{self.conf["scheduler"]["max_worker"]}.pth')
                if self.enable_cluster_embedding and self.query_scale > 1:
                    self.cluster_query_num = self.query_num // self.query_scale
                    self.query_embeddings = np.eye(self.cluster_query_num).astype('float32') # One-hot Encoding
                    self.cluster_embeddings = np.eye(self.query_scale).astype('float32')
                else:
                    self.query_embeddings = np.eye(self.query_num).astype('float32') # One-hot Encoding
                with open(f'envs/scheduler/cache/{self.conf["database"]["host"]}/{self.conf["database"]["database"]}/base_costs{self.query_scale}.json') as f:
                    self.base_costs = json.load(f)
        self.enable_resource = enable_resource
        self.conn_ssh = conn_ssh
        self.local_resource = local_resource
        self.rel_clamp_scale = self.conf.getfloat('scheduler', 'rel_clamp_scale')

    def __del__(self):
        # for conn in self.conn_list:
        #     conn.close()
        # if self.enable_monitor:
        #     self.monitor_conn.close()
        # sys.stdout = self.stdout_bak
        # self.flog.close()
        pass

    def predict(self):
        poss = self.executing_qposs + [-1] * (self.max_worker - len(self.executing_queries))
        qids = self.executing_queries + [0] * (self.max_worker - len(self.executing_queries))
        if self.simulator_model == 'lgbm':
            qids = np.asarray(self.one_hot_enc.transform([qids]).todense())
            time_lasts = np.array([[self.current_time - self.query_time_info['start'][pos] for pos in poss]])
            query_costs = np.array([[self.query_list.costs[pos] for pos in poss]])
            X = np.concatenate([qids, time_lasts, query_costs], axis=1)
            X = self.scaler.transform(X)
            finish_index = min(self.clf_model.predict(X)[0], len(self.executing_queries)-1)
            finish_time = max(self.reg_model.predict(X)[0], 0.01)
        elif self.simulator_model == 'con_query_former':
            if self.enable_cluster_embedding and self.query_scale > 1:
                query_embeddings = np.array([np.concatenate([self.query_embeddings[(i-1)%self.cluster_query_num], self.cluster_embeddings[(i-1)//self.cluster_query_num]]) 
                                             if i > 0 else np.zeros((self.cluster_query_num+self.query_scale, )).astype('float32') for i in qids])
            else:
                query_embeddings = np.array([self.query_embeddings[i-1] if i > 0 else np.zeros((self.query_num, )).astype('float32') for i in qids])
            time_lasts = np.array([[round(self.current_time - self.query_time_info['start'][pos], 3) if pos > -1 else 0] for pos in poss]).astype('float32')
            base_list =[]
            for i in qids:
                if i > 0:
                    if i > self.query_list.base_query_num:
                        i -= self.query_num - self.query_list.base_query_num
                    key = str(i) + " " + str((self.query_worker[i]+2 if self.conf['database']['database'].startswith('tpcds') else self.query_worker[i]+3) 
                                             if self.query_worker[i] > 0 else self.query_worker[i] + 1)
                    try:
                        base_list.append([self.base_costs[key][6]])
                    except Exception:
                        key = str(i) + " " + str(1)
                        base_list.append([self.base_costs[key][6]])
                else:
                    base_list.append([200])
            base_costs = np.array(base_list).astype('float32')
            workers = np.array([[((self.query_worker[i]+2 if self.conf['database']['database'].startswith('tpcds') else self.query_worker[i]+3) 
                                  if self.query_worker[i] > 0 else self.query_worker[i] + 1) if i > 0 else 0] for i in qids]).astype('float32')
            query_embeddings, time_lasts, workers, base_costs = torch.tensor(query_embeddings).to('cuda'), \
                torch.tensor(time_lasts).to('cuda'), torch.tensor(workers).to('cuda'), torch.tensor(base_costs).to('cuda')
            query_embeddings, time_lasts, workers, base_costs = query_embeddings.unsqueeze(0), \
                time_lasts.unsqueeze(0), workers.unsqueeze(0), base_costs.unsqueeze(0)
            if type(self.con_query_former).__name__ == ConcurrentQueryFormer.__name__:
                output_time, output_index = self.con_query_former(query_embeddings, time_lasts*self.time_scale, workers, base_costs*self.time_scale)
            elif type(self.con_query_former).__name__ == SharedQueryFormer.__name__:
                output_time, output_index = self.con_query_former(None, query_embeddings, time_lasts*self.time_scale, workers, base_costs*self.time_scale)
            finish_index = min(output_time.argmin().item() if self.use_rank_loss else output_index.argmax().item(), len(self.executing_queries)-1)
            finish_time = max((output_time[finish_index] / self.time_scale).item(), 0.01)
        qid = self.executing_queries[finish_index]
        idx = self.query_conn_id[qid]
        self.current_time += finish_time
        return idx, qid

    def preprocess(self):
        self.start_time, self.current_time, self.last_end_time = -1, -1, -1
        self.pending_queries = [q[0] for q in self.queries]
        self.executing_queries = []
        self.executed_queries = []
        self.pending_qposs = list(range(self.query_num))
        self.executing_qposs = []
        self.executed_qposs = []
        self.scaled_executing_queries = []
        self.scaled_executed_queries = []
        self.current_cluster = 1
        self.query_time_info = {
            'start': [-1] * self.query_num,
            'end': [-1] * self.query_num,
            'relative': [-1] * self.query_num,
            'relative_clamp': [-1] * self.query_num,
        }
        self.query_worker, self.query_conn_id = dict(), dict()
        self.relative_time, self.relative_time_clamp, self.total_time = 1e-10, 1e-10, 1e-10
        self.next_qpos = -1
        self.next_worker = -1
        self.cost = -1
        self.mean_tailing = -1
        self.err_flag = False
        self.resource_status = {}
        self.info_aux = {
            'observations': {
                "query_status": [],
                "time_last": [],
            },
            'finish_qposs': [],
            'finish_times': [],
        }
    
    def start_query(self, qid, worker, conn_id, q_start_time=None):
        qpos = self.query_list.id_to_pos[qid]
        self.pending_queries.remove(qid)
        self.executing_queries.append(qid)
        self.pending_qposs.remove(qpos)
        self.executing_qposs.append(qpos)
        # if (qid - 1) // self.query_list.original_query_num == self.query_list.query_scale - 1:
        #     self.scaled_executing_queries.append((qid - 1) % self.query_list.original_query_num + 1)
        if self.query_list.query_scale > 1 and self.query_cluster:
            scaled_qid = (qid - 1) % self.query_list.original_query_num + 1
            if self.current_cluster < self.query_list.query_scale and len(self.executed_queries) + len(self.executing_queries) \
                == self.current_cluster * self.query_list.original_query_num:
                self.scaled_executing_queries, self.scaled_executed_queries = [], []
                self.current_cluster += 1
            else:
                self.scaled_executing_queries.append(scaled_qid)
        assert self.use_simulator or not (q_start_time is None), 'q_start_time must be provided when not use_simulator'
        self.query_time_info['start'][qpos] = q_start_time if not self.use_simulator else self.current_time
        self.query_worker[qid] = worker
        self.query_conn_id[qid] = conn_id

        if len(self.executing_qposs) == self.max_worker or len(self.executed_qposs) > 0:
            query_status = np.zeros((self.args.query_num*self.query_scale,), dtype=np.int64)
            query_status[np.array(self.executing_qposs, dtype=np.int64)] = \
                1 if not self.args.enable_worker else [1 + self.query_worker[i] for i in self.executing_queries]
            query_status[np.array(self.executed_qposs, dtype=np.int64)] = \
                1 + self.args.max_worker
            self.info_aux['observations']['query_status'].append(query_status)
            self.info_aux['observations']['time_last'].append(self.get_time_last())
    
    def finish_query(self, qid, q_end_time=None):
        qpos = self.query_list.id_to_pos[qid]
        self.executing_queries.remove(qid)
        self.executed_queries.append(qid)
        self.executing_qposs.remove(qpos)
        self.executed_qposs.append(qpos)
        # if (qid - 1) // self.query_list.original_query_num == self.query_list.query_scale - 1:
        #     self.scaled_executing_queries.remove((qid - 1) % self.query_list.original_query_num + 1)
        #     self.scaled_executed_queries.append((qid - 1) % self.query_list.original_query_num + 1)
        if self.query_list.query_scale > 1 and self.query_cluster:
            if (qid - 1) // self.query_list.original_query_num == self.current_cluster - 1:
                scaled_qid = (qid - 1) % self.query_list.original_query_num + 1
                self.scaled_executing_queries.remove(scaled_qid)
                self.scaled_executed_queries.append(scaled_qid)
        assert self.use_simulator or not (q_end_time is None), 'q_end_time must be provided when not use_simulator'
        self.query_time_info['end'][qpos] = q_end_time if not self.use_simulator else self.current_time
        assert not self.query_time_info['end'][qpos] is None, 'Query end time not provided'
        self.query_time_info['relative'][qpos] = self.query_list.costs[qpos] / \
            (self.query_time_info['end'][qpos] - self.query_time_info['start'][qpos])
        self.relative_time += self.query_time_info['relative'][qpos] * self.query_list.costs[qpos]
        self.query_time_info['relative_clamp'][qpos] = min(max(self.query_list.costs[qpos] / (self.query_time_info['end'][qpos] - self.query_time_info['start'][qpos]), 
                                                               1 / self.rel_clamp_scale), self.rel_clamp_scale)
        self.relative_time_clamp += self.query_time_info['relative_clamp'][qpos] * self.query_list.costs[qpos]
        self.total_time += self.query_list.costs[qpos]

        self.info_aux['finish_qposs'].append([qpos])
        if len(self.info_aux['finish_times']) == 0:
            self.info_aux['finish_times'].append([self.query_time_info['end'][qpos] - self.start_time])
        else:
            self.info_aux['finish_times'].append([self.query_time_info['end'][qpos] - self.query_time_info['end'][self.info_aux['finish_qposs'][-2][0]]])
        if len(self.pending_qposs) == 0 and len(self.executed_qposs) < self.args.query_num:
            query_status = np.zeros((self.args.query_num,), dtype=np.int64)
            query_status[np.array(self.executing_qposs, dtype=np.int64)] = \
                1 if not self.args.enable_worker else [1 + self.query_worker[i] for i in self.executing_queries]
            query_status[np.array(self.executed_qposs, dtype=np.int64)] = \
                1 + self.args.max_worker
            self.info_aux['observations']['query_status'].append(query_status)
            self.info_aux['observations']['time_last'].append(self.get_time_last())
    
    def get_throughput(self):
        return len(self.executed_queries) / (time.time() if not self.use_simulator else self.current_time - self.start_time)
    
    def get_relative_time(self):
        # relative, total = 1e-10, 1e-10
        # for i in range(self.query_num):
        #     if self.query_time_info['relative'][i] != -1:
        #         relative += self.query_time_info['relative'][i] * self.query_list.costs[i]
        #         total += self.query_list.costs[i]
        # return relative / total
        return self.relative_time / self.total_time
    
    def get_relative_time_clamp(self):
        return self.relative_time_clamp / self.total_time
    
    def get_absolute_time(self):
        return self.total_time / self.query_list.total_cost
    
    def get_time_last(self):
        cur_time = self.current_time if self.use_simulator else time.time()
        time_last = np.zeros((self.query_num))
        time_last[self.executing_qposs] = [cur_time - self.query_time_info['start'][qpos] for qpos in self.executing_qposs]
        return time_last
    
    def get_cluster_time_last(self, qposs):
        cur_time = self.current_time if self.use_simulator else time.time()
        return cur_time - min([self.query_time_info['start'][qpos] for qpos in qposs])
    
    def get_time_diff(self):
        cur_time = self.current_time if self.use_simulator else time.time()
        result = cur_time - self.last_end_time
        self.last_end_time = cur_time
        return result
    
    def schedule_with_yield(self):
        self.preprocess()
        # printtime("******Start Scheduler")
        self.query_list = QueryList(self.database.conn, self.queries, None, self.conf)
        # with ThreadPoolExecutor(max_workers=self.max_worker if not self.enable_monitor else self.max_worker+1) as t:
        t = ThreadPoolExecutor(max_workers=self.max_worker if not self.enable_monitor else self.max_worker+1) if not self.use_simulator else None
        finish_event = Event() if not self.use_simulator else None
        if self.enable_monitor and not self.use_simulator:
            t.submit(thread_monitor, self.monitor_conn, finish_event, 5, self.runtime_log)
        obj_list = []
        for qid in range(self.max_worker):
            if not self.use_simulator and self.enable_resource and (not self.conn_ssh is None or self.local_resource):
                self.resource_status = None if self.use_simulator else (get_local_resource() if self.local_resource else get_remote_resource(self.conn_ssh))
                print_time(f'resource status: cpu {100-self.resource_status["cpu_times_percent"].idle:.2f}%, vmem {self.resource_status["virtual_memory"].percent:.2f}%, smem {self.resource_status["swap_memory"].percent:.2f}%',
                           self.runtime_log)
            yield 0
            if qid == 0:
                self.start_time = time.time()
                self.current_time = self.start_time
                self.last_end_time = self.start_time
                self.query_list.start_time = self.start_time
            cur_state = {
                'conn_id': qid,
                'cur_queries': self.executing_queries
            }
            if self.query_list.query_scale > 1 and self.query_cluster:
                query = self.query_list.get_next_query(strategy=self.strategy, id=self.next_qpos+self.query_list.original_query_num*(self.current_cluster-1), worker=self.next_worker, cur_state=cur_state)
            else:
                query = self.query_list.get_next_query(strategy=self.strategy, id=self.next_qpos, worker=self.next_worker, cur_state=cur_state)
            obj = t.submit(thread_query, qid, self.conn_list[qid], query, self.runtime_log) if not self.use_simulator else None
            if not self.use_simulator:
                self.start_query(query[0][0], self.next_worker, qid, time.time())
            else:
                self.current_time = time.time()
                self.start_query(query[0][0], self.next_worker, qid, self.current_time)
            obj_list.append(obj)
        self.current_time = time.time()
        while (not self.query_list.empty() and not self.err_flag):
            if not self.use_simulator:
                time.sleep(0.01)
                for future in obj_list:
                    if future.done():
                        idx, qid, self.err_flag, q_end_time = future.result()
                        if self.err_flag:
                            break
                        if qid != -1:
                            self.finish_query(qid, q_end_time)
                        obj_list.remove(future)
                        if not self.query_list.empty():
                            if self.enable_resource and (not self.conn_ssh is None or self.local_resource):
                                self.resource_status = None if self.use_simulator else (get_local_resource() if self.local_resource else get_remote_resource(self.conn_ssh))
                                print_time(f'resource status: cpu {100-self.resource_status["cpu_times_percent"].idle:.2f}%, vmem {self.resource_status["virtual_memory"].percent:.2f}%, smem {self.resource_status["swap_memory"].percent:.2f}%',
                                           self.runtime_log)
                            yield 0
                            cur_state = {
                                'conn_id': idx,
                                'cur_queries': self.executing_queries
                            }
                            if self.query_list.query_scale > 1 and self.query_cluster:
                                query = self.query_list.get_next_query(strategy=self.strategy, id=self.next_qpos+self.query_list.original_query_num*(self.current_cluster-1), worker=self.next_worker, cur_state=cur_state)
                            else:
                                query = self.query_list.get_next_query(strategy=self.strategy, id=self.next_qpos, worker=self.next_worker, cur_state=cur_state)
                            obj = t.submit(thread_query, idx, self.conn_list[idx], query, self.runtime_log)
                            if query[0][0] != -1:
                                self.start_query(query[0][0], self.next_worker, idx, time.time())
                            obj_list.append(obj)
            else:
                idx, qid = self.predict()
                self.finish_query(qid)
                if not self.query_list.empty():
                    yield 0
                    cur_state = {
                        'conn_id': idx,
                        'cur_queries': self.executing_queries
                    }
                    query = self.query_list.get_next_query(strategy=self.strategy, id=self.next_qpos, worker=self.next_worker, cur_state=cur_state)
                    if query[0][0] != -1:
                        self.start_query(query[0][0], self.next_worker, idx)
        if not self.use_simulator:
            _, not_done = wait(obj_list, timeout=3000)
            end_time = time.time()
            # if len(not_done) > 0:
            #     print('len(not_done)', len(not_done))
            if not self.err_flag and len(not_done) == 0:
                for future in obj_list:
                    _, qid, self.err_flag, q_end_time = future.result()
                    if self.err_flag:
                        break
                    if qid != -1:
                        self.finish_query(qid, q_end_time)
            finish_event.set()
            t.shutdown(wait=False)
        else:
            while len(self.executing_queries) > 0:
                idx, qid = self.predict()
                self.finish_query(qid)
            end_time = self.current_time

        # printtime("******Finish Scheduler")
        time_last = end_time - self.start_time if self.use_simulator or (not self.err_flag and len(not_done) == 0) else -1
        if not self.use_simulator:
            print("Time last %.2f" % (time_last), flush=True)
            print("Time last %.2f" % (time_last), file=self.runtime_log, flush=True)
        else:
            print("Time last %.2f, actual %.2f" % (time_last, time.time() - self.start_time), flush=True)
        self.cost = time_last
        # assert min(self.query_time_info['end']) > 0, 'Terminate when queries still executing'
        if min(self.query_time_info['end']) < 0:
            print(f'Terminate when queries still executing (time last={time_last:.3f})')
        last_end_times = sorted(self.query_time_info['end'], reverse=True)[:self.max_worker]
        self.mean_tailing = np.mean(last_end_times) - last_end_times[-1]
        yield 1

if __name__ == '__main__':
    conf = get_config('config.ini')
    host, database = conf['database']['host'], conf['database']['database']
    result_path = 'result.json'
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            result = json.load(f)
    else:
        result = dict()
    if host not in result.keys():
        result[host] = dict()
    if database not in result[host].keys():
        result[host][database] = dict()
    # for strategy in ['max_cost_first', 'max_cost_first_dop', 'max_cost_first_gather']:
    for strategy in ['random']:
        if strategy not in result[host][database].keys():
            result[host][database][strategy] = dict()
        for max_worker in range(10, 100, 10):
            key_worker = f'max_worker={max_worker}'
            if key_worker not in result[host][database][strategy].keys():
                result[host][database][strategy][key_worker] = []
            for _ in range(5):
                time_last = -1
                while time_last == -1:
                    qs = QueryScheduler(conf=conf, strategy=strategy, max_worker=max_worker)
                    time_last = qs.schedule()
                    del qs
                result[host][database][strategy][key_worker].append(time_last)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=4)