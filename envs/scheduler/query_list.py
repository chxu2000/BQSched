import os, time, json, random, math, re
import numpy as np
import pandas as pd
from envs.scheduler.context import context


def add_query_hint(s, a, symbol=';'):
    sym_poss = [m.start() for m in re.finditer(symbol, s)]
    sym_num = len(sym_poss)
    for i in range(sym_num):
        s = s[:sym_poss[i]] + a + s[sym_poss[i]:]
        sym_poss = [m.start() for m in re.finditer(symbol, s)]
    return s


class QueryList():

    def __init__(self, conn, queries, table_list, conf, vary_conn_num):
        self.conf = conf
        self.connection = conn
        self.queries = queries
        self.raw_queries = [query[3] for query in self.queries]
        self.ids = [query[0] for query in self.queries]
        self.clean_queries = []
        self.tables = []
        self.plans = []
        self.costs = []
        self.long_queries = [4, 23, 30, 11, 81]
        self.unexecuted_ids = self.ids.copy()
        self.id_to_pos = dict(zip(self.ids, list(range(len(self.ids)))))
        self.table_list = table_list
        self.max_worker = int(self.conf['scheduler']['max_worker'])
        self.cache_prefix = 'envs/scheduler/cache/{}/{}/'.format(self.conf['database']['host'], self.conf['database']['database'])
        # self.context = context(conn, table_list, self.max_worker)
        self.start_time = None
        self.vary_conn_num = vary_conn_num
        self.build_list()

    def __len__(self):
        return len(self.raw_queries)

    def empty(self):
        return len(self.unexecuted_ids) == 0
    
    def executed_query_num(self):
        return len(self.ids) - len(self.unexecuted_ids)

    def build_list(self):
        if not os.path.exists(self.cache_prefix):
            os.makedirs(self.cache_prefix)
        self.build_clean_queries()
        # self.build_plans()
        # self.extract_tables()
        self.build_costs()
        self.build_dops()
        self.build_mems()
        self.build_cgroups()
        # self.build_conn_queries()
        self.build_worker_masks()
        self.build_mem_masks()
        self.build_worker_mem_masks()

    def build_clean_queries(self):
        for rawq in self.raw_queries:
            sub_queries = []
            while True:
                semi_pos = rawq.find(';')
                if semi_pos == -1:
                    break
                sub_queries.append(rawq[:semi_pos + 1])
                rawq = rawq[semi_pos + 1:]
            self.clean_queries.append(sub_queries)

    def build_plans(self):
        plans_path = self.cache_prefix + 'plans.json'
        if os.path.exists(plans_path):
            with open(plans_path) as f:
                self.plans = json.load(f)
        else:
            cur = self.connection.cursor()
            for sub_queries in self.clean_queries:
                plans = []
                for query in sub_queries:
                    cur.execute('explain (format json)\n' + query)
                    plans += list(cur.fetchall())
                self.plans.append(plans)
            cur.close()
            with open(plans_path, 'w') as f:
                json.dump(self.plans, f)

    def extract_tables_from_plan(self,plan):
        tables = []
        if 'Relation Name' in plan and plan['Node Type'] != 'Index Scan':
            tables.append(plan['Relation Name'])
        if 'Plans' in plan:
            for p in plan['Plans']:
                tables += self.extract_tables_from_plan(p)
        return tables

    def extract_tables(self):
        for plans in self.plans:
            tables = []
            for plan in plans:
                tables += self.extract_tables_from_plan(plan[0][0]['Plan'])
            self.tables.append(list(set(tables)))

    def build_costs(self):
        self.query_scale = int(self.conf['database']['query_scale'])
        costs_path = self.cache_prefix + f'costs{"" if self.query_scale == 1 else self.query_scale}.json'
        if os.path.exists(costs_path):
            with open(costs_path) as f:
                total_costs = json.load(f)
            # if self.vary_conn_num and not (f'max_worker={self.max_worker}' in total_costs):
            #     query_costs = [np.mean(clist) if len(clist) > 0 else -1 for clist in total_costs[f'max_worker=20']]
            # else:
            #     query_costs = [np.mean(clist) if len(clist) > 0 else -1 for clist in total_costs[f'max_worker={self.max_worker}']]
            try:
                query_costs = [np.mean(clist) if len(clist) > 0 else -1 for clist in total_costs[f'max_worker={self.max_worker}']]
            except Exception as e:
                # print(f"RuntimeWarning: {repr(e)}, using 'max_worker=20' as the key")
                if 'max_worker=20' in total_costs:
                    query_costs = [np.mean(clist) if len(clist) > 0 else -1 for clist in total_costs[f'max_worker=20']]
                else:
                    query_costs = [np.mean(clist) if len(clist) > 0 else -1 for clist in total_costs[f'max_worker=16']]
            # query_costs = query_costs * math.ceil(len(self.ids) / len(query_costs)) # For duplicate queries
            self.base_query_num = len(query_costs)
            if self.query_scale == 1:
                self.original_query_num = len(self.ids)
                self.costs = [query_costs[(i-1)%self.base_query_num] for i in self.ids]
                # if self.original_query_num <= self.base_query_num:
                #     self.costs = [query_costs[i-1] for i in self.ids]
                # else:
                #     self.costs = query_costs + query_costs[-(self.original_query_num - self.base_query_num):]
            else:
                self.original_query_num = int(len(self.ids) / self.query_scale)
                self.costs = [query_costs[i-1] for i in self.ids]
                # self.costs = [query_costs[(id - 1) % self.original_query_num] for _, id in enumerate(self.ids)]
        else:
            for i in self.ids:
                if i in self.long_queries:
                    self.costs.append(10)
                else:
                    self.costs.append(2)
        self.total_cost = sum(self.costs)

    def build_worker_masks(self):
        costs_worker_path = self.cache_prefix + 'costs_worker.json'
        if os.path.exists(costs_worker_path):
            with open(costs_worker_path) as f:
                costs_worker = json.load(f)
            costs_worker_list = []
            for key, value in costs_worker.items():
                value = [np.mean(clist) if len(clist) > 0 else -1 for clist in value]
                try:
                    value = [value[(i-1)%self.base_query_num] for i in self.ids]
                except:
                    value = [value[(i-1)%(self.base_query_num//self.query_scale)] for i in self.ids[:self.original_query_num]]
                # if self.original_query_num <= len(value):
                #     value = [value[i-1] for i in self.ids[:self.original_query_num]]
                # else:
                #     value = value + value[-(self.original_query_num - len(value)):]
                costs_worker_list.append(value)
            self.worker_masks = [[True for _ in range(len(costs_worker_list[0]))]] + [[False for __ in range(len(costs_worker_list[0]))] for _ in range(len(costs_worker_list) - 1) ]
            for i in range(1, len(costs_worker_list)):
                for j in range(self.original_query_num):
                    if (costs_worker_list[i][j] < costs_worker_list[0][j] * (1 - float(self.conf['scheduler']['rel_improve']))) and \
                        ((costs_worker_list[0][j] - costs_worker_list[i][j]) > float(self.conf['scheduler']['abs_improve'])):
                        self.worker_masks[i][j] = True
            for i in range(len(self.worker_masks)):
                self.worker_masks[i] *= self.query_scale
            costs_worker_np = np.array(costs_worker_list)
            # self.soft_worker_masks = np.sqrt(costs_worker_np[0, :]/costs_worker_np[1, :]/2)   # 1
            self.soft_worker_masks = costs_worker_np[0, :]/costs_worker_np[1, :]-np.sqrt(2)     # 2

    def build_mem_masks(self):
        costs_mem_path = self.cache_prefix + 'costs_mem.json'
        if os.path.exists(costs_mem_path):
            with open(costs_mem_path) as f:
                costs_mem = json.load(f)
            costs_mem_list = []
            for key, value in costs_mem.items():
                value = [np.mean(clist) if len(clist) > 0 else -1 for clist in value]
                try:
                    value = [value[(i-1)%self.base_query_num] for i in self.ids]
                except:
                    value = [value[(i-1)%(self.base_query_num//self.query_scale)] for i in self.ids[:self.original_query_num]]
                # if self.original_query_num <= len(value):
                #     value = [value[i-1] for i in self.ids[:self.original_query_num]]
                # else:
                #     value = value + value[-(self.original_query_num - len(value)):]
                costs_mem_list.append(value)
            self.mem_masks = [[True for _ in range(len(costs_mem_list[0]))]] + [[False for __ in range(len(costs_mem_list[0]))] for _ in range(len(costs_mem_list) - 1) ]
            for i in range(1, len(costs_mem_list)):
                for j in range(self.original_query_num):
                    if (costs_mem_list[i][j] < costs_mem_list[0][j] * (1 - float(self.conf['scheduler']['rel_improve']))) and \
                        ((costs_mem_list[0][j] - costs_mem_list[i][j]) > float(self.conf['scheduler']['abs_improve'])):
                        self.mem_masks[i][j] = True
            for i in range(len(self.mem_masks)):
                self.mem_masks[i] *= self.query_scale
            costs_mem_np = np.array(costs_mem_list)
            self.soft_mem_masks = costs_mem_np[0, :]/costs_mem_np[1, :]-np.sqrt(2)     # 2
    
    def build_worker_mem_masks(self):
        if hasattr(self, 'worker_masks') and hasattr(self, 'mem_masks'):
            assert len(self.worker_masks[0]) == len(self.mem_masks[0]), 'worker_masks and mem_masks must have the same base query number'
            self.worker_mem_masks = [
                [self.worker_masks[i][k] and self.mem_masks[j][k] for k in range(len(self.worker_masks[0]))]
                for j in range(len(self.mem_masks))
                for i in range(len(self.worker_masks))
            ]
    
    def build_dops(self):
        dops_path = self.cache_prefix + 'dops.json'
        if os.path.exists(dops_path):
            with open(dops_path) as f:
                self.dops = json.load(f)
    
    def build_mems(self):
        mems_path = self.cache_prefix + 'mems.json'
        if os.path.exists(mems_path):
            with open(mems_path) as f:
                self.mems = json.load(f)
    
    def build_cgroups(self):
        cgroups_path = self.cache_prefix + 'cgroups.json'
        if os.path.exists(cgroups_path):
            with open(cgroups_path) as f:
                self.cgroups = json.load(f)
    
    def build_conn_queries(self):
        conn_queries_path = self.cache_prefix + 'conn_queries.json'
        if os.path.exists(conn_queries_path):
            with open(conn_queries_path) as f:
                self.conn_queries = json.load(f)[f'max_worker={self.max_worker}']

    def get_next_query(self, strategy, id=None, worker=None, mem=None, cur_state=None):
        prefix, postfix = '', ''
        if 'index' in strategy:
            chosen_id = self.ids[id]
            # if self.query_scale > 1:
            #     while chosen_id not in self.unexecuted_ids and chosen_id <= len(self.ids):
            #         chosen_id += self.original_query_num
            #     assert chosen_id <= len(self.ids), 'Selecting executed query'
            if worker != -1:
                if self.conf['database']['port'] == '8000':
                    # prefix += f'set query_dop={1 if worker==0 else -(worker+1)};'
                    prefix += f'set query_dop={0 if worker==0 else worker+2};'
                elif self.conf['database']['port'] == '1433':
                    if worker > 0:
                        self.queries[self.id_to_pos[chosen_id]][1] = add_query_hint(self.queries[self.id_to_pos[chosen_id]][1], f' (MAXDOP {0 if worker == 0 else worker + 1})', '\.')
                        self.queries[self.id_to_pos[chosen_id]][3] = add_query_hint(self.queries[self.id_to_pos[chosen_id]][3], f' OPTION (MAXDOP {0 if worker == 0 else worker + 1})')
                else:
                    if self.conf["database"]["database"].startswith("imdb"):
                        prefix += f'set max_parallel_workers_per_gather={worker+1 if worker>0 else worker};'    # 02
                        # prefix += f'set max_parallel_workers_per_gather={worker+2};'    # 23
                    else:
                        prefix += f'set max_parallel_workers_per_gather={(worker+1 if self.conf["database"]["database"].startswith("tpcds") else worker+2) if worker>0 else worker};'
            if mem is not None and mem != -1:
                work_mem = "\'16MB\'" if mem > 0 else "\'4MB\'"
                prefix += f'set work_mem={work_mem};'
            if 'worker' in strategy:
                prefix += f'set max_parallel_workers_per_gather={2 if chosen_id in self.long_queries else 0};'
        elif strategy in ['random']:
            chosen_id = random.choice(self.unexecuted_ids)
        elif strategy in ['table_greedy', 'table_size_greedy', 'table_size_buffer']:
            cur_tables = []
            for qid in cur_state['cur_queries']:
                cur_tables += self.tables[self.id_to_pos[qid]]
            cur_tables = set(cur_tables)
            max_id, max_inter = -1, -1
            for qid in self.unexecuted_ids:
                inter_tables = set.intersection(cur_tables, self.tables[self.id_to_pos[qid]])
                if strategy == 'table_greedy':
                    inter_score = len(inter_tables)
                elif strategy == 'table_size_greedy':
                    inter_score = 0
                    for table in inter_tables:
                        inter_score += self.table_list.get_table_size(table)
                elif strategy == 'table_size_buffer':
                    query_tables = self.tables[self.id_to_pos[qid]]
                    query_cost = self.costs[self.id_to_pos[qid]]
                    inter_score = self.context.get_similarity_by_query(query_tables, query_cost)
                if inter_score > max_inter:
                    max_id, max_inter = qid, inter_score
            chosen_id = max_id
        elif 'cost_first' in strategy:
            unexecuted_costs = [self.costs[self.id_to_pos[i]] for i in self.unexecuted_ids]
            if 'max_cost_first' in strategy:
                if 'sleep' in strategy:
                    if cur_state['conn_id'] >= 5 and time.time() - self.start_time <= 8:
                        return (-1, 1), prefix, postfix
                chosen_id = self.unexecuted_ids[random.choice(np.where(unexecuted_costs == max(unexecuted_costs))[0])]
                if 'worker' in strategy:
                    prefix += f'set max_parallel_workers_per_gather={2 if chosen_id in self.long_queries else 0};'
                else:
                    if 'dop' in strategy or 'gather' in strategy:
                        dop = self.dops[self.id_to_pos[chosen_id]]
                        if dop != 1:
                            prefix += f'set query_dop={dop};'
                            postfix += f'set query_dop=1;'
                    if 'mem' in strategy or 'gather' in strategy:
                        mem = self.mems[self.id_to_pos[chosen_id]]
                        if mem != 0:
                            prefix += f'set query_mem=\'{mem}MB\';'
                            postfix += f'set query_mem=0;'
                    if 'cg' in strategy or 'gather' in strategy:
                        cgroup = self.cgroups[self.id_to_pos[chosen_id]]
                        if cgroup != 'Medium':
                            prefix += f'set cgroup_name=\'{cgroup}\';'
                            postfix += f'set cgroup_name=\'DefaultClass:Medium\';'
            elif 'min_cost_first' in strategy:
                chosen_id = self.unexecuted_ids[random.choice(np.where(unexecuted_costs == min(unexecuted_costs))[0])]
            else:
                pass
        elif strategy in ['fixed_conn_queries']:
            if len(self.conn_queries[str(cur_state['conn_id'])]) > 0:
                chosen_id = self.conn_queries[str(cur_state['conn_id'])].pop(0)
            else:
                chosen_id = random.choice(self.unexecuted_ids)
                for key, value in self.conn_queries.items():
                    if chosen_id in value:
                        self.conn_queries[key].remove(chosen_id)
                        break
        else:
            pass
        self.unexecuted_ids.remove(chosen_id)
        return self.queries[self.id_to_pos[chosen_id]], prefix, postfix