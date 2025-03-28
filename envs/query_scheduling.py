import gym, time, numpy as np, json
from gym import spaces
from math import sqrt
from sklearn.preprocessing import normalize
from envs.scheduler.query_scheduler import *

from rpyc.utils.zerodeploy import DeployedServer
from plumbum import SshMachine

import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


class QuerySchedulingEnv(gym.Env):

    def __init__(self, reward_type='cost', args=None, conf=None, embedding_path='nets/embeddings.npy', runtime_log=None):
        # Observations are dictionaries with the queries' status.
        # Each status is encoded as 0/1/2 for pending/executing/executed.
        # self.observation_space = spaces.Dict(
        #     {
        #         "query_status": spaces.MultiDiscrete(np.ones((99,), dtype=np.int64) * 2),
        #     }
        # )
        self.args = args
        self.conf = conf
        self.reward_type = reward_type
        self.query_scale = self.conf.getint('database', 'query_scale')
        self.query_cluster = self.conf.getboolean('database', 'query_cluster')
        self.vertical_cluster = self.conf.getboolean('database', 'vertical_cluster')
        self.overlap_cluster = self.conf.getboolean('database', 'overlap_cluster')
        self.worker_cluster = self.conf.getboolean('database', 'worker_cluster')
        self.vary_conn_num = 'vary_conn_num' in self.conf['scheduler'] and self.conf.getboolean('scheduler', 'vary_conn_num')
        self.max_conn_num = self.conf.getint('scheduler', 'max_worker')
        if self.vary_conn_num:
            self.min_conn_num = self.conf.getint('scheduler', 'min_worker')
        self.compress_action_c = 'compress_action_c' in self.conf['scheduler'] and self.conf.getboolean('scheduler', 'compress_action_c')
        if self.overlap_cluster or self.worker_cluster:
            cluster_result = np.load(self.conf['database']['cluster_result_path'])
            self.cluster_result = cluster_result['arr_0']
            self.cluster_num = int(max(self.cluster_result) + 1)
        if self.overlap_cluster:
            if args.enable_worker:
                with open(f'envs/scheduler/cache/{self.conf["database"]["host"]}/{self.conf["database"]["database"]}/costs_worker{self.query_scale}.json', 'r') as f:
                    costs_worker = json.load(f)
                costs_worker_list = []
                for _, value in costs_worker.items():
                    value = [np.mean(clist) if len(clist) > 0 else -1 for clist in value]
                    costs_worker_list.append(value)
                self.query_masks = np.array([[True] * len(costs_worker_list[0])] + [[False] * len(costs_worker_list[0])] * (len(costs_worker_list) - 1))
                for i in range(1, len(costs_worker_list)):
                    for j in range(len(costs_worker_list[0])):
                        if (costs_worker_list[i][j] < costs_worker_list[0][j] * (1 - float(self.conf['scheduler']['rel_improve']))) and \
                            ((costs_worker_list[0][j] - costs_worker_list[i][j]) > float(self.conf['scheduler']['abs_improve'])):
                            self.query_masks[i][j] = True
                self.cluster_masks = np.array([[True] * self.cluster_num] + [[False] * self.cluster_num] * (len(costs_worker_list) - 1))
                for i in range(1, len(costs_worker_list)):
                    for j in range(self.cluster_num):
                        if sum(self.query_masks[i][np.where(self.cluster_result==j)[0]]) > 0:
                            self.cluster_masks[i][j] = True
            else:
                self.query_workers = cluster_result['arr_1']
        if self.worker_cluster:
            cluster_features = cluster_result['arr_1']
            self.cluster_masks = np.array([[True for _ in range(self.cluster_num)], 
                                           np.logical_or(cluster_features[:, 0] > self.conf.getfloat('scheduler', 'abs_threshold'), 
                                                         cluster_features[:, 1] > self.conf.getfloat('scheduler', 'rel_threshold'))], dtype=bool)
        self.query_num = args.query_num if self.query_cluster else args.query_num * self.query_scale
        self.action_num = args.query_num if self.vertical_cluster else (self.cluster_num if self.overlap_cluster or self.worker_cluster else self.query_num)
        self.baseline = args.init_baseline
        self.enable_embedding = args.enable_embedding
        self.enable_resource = args.enable_resource
        self.local_resource = args.local_resource
        self.enable_time_last = args.enable_time_last
        self.conn_ssh = None
        self.enable_worker = args.enable_worker
        self.max_worker = args.max_worker
        self.action_flatten = args.action_flatten
        self.runtime_log = runtime_log

        assert not (self.enable_embedding and self.enable_resource), 'enable_embedding and resource not supported in the meanwhile'
        assert self.enable_worker or self.max_worker == 1, 'when disable worker feature, max worker must be 1'
        assert self.enable_worker or not self.action_flatten, 'when disable worker feature, action flatten must be False'
        assert not (self.query_cluster and self.vertical_cluster), 'query_cluster and vertical_cluster not supported in the meanwhile'
        assert not (self.query_cluster and self.overlap_cluster), 'query_cluster and overlap_cluster not supported in the meanwhile'
        assert not (self.vertical_cluster and self.overlap_cluster), 'vertical_cluster and overlap_cluster not supported in the meanwhile'
        # assert not (self.enable_worker and self.overlap_cluster), 'enable_worker and overlap_cluster not supported in the meanwhile'
        assert not (self.query_cluster and self.worker_cluster), 'query_cluster and worker_cluster not supported in the meanwhile'
        assert not (self.vertical_cluster and self.worker_cluster), 'vertical_cluster and worker_cluster not supported in the meanwhile'
        assert not (self.overlap_cluster and self.worker_cluster), 'overlap_cluster and worker_cluster not supported in the meanwhile'
        if self.overlap_cluster or self.worker_cluster:
            assert len(self.cluster_result) == args.query_num * self.query_scale, 'length of cluster result not correct'

        if self.enable_embedding:
            # Initialize query embeddings
            self.query_embeddings = np.load(embedding_path)
            self.observation_space = spaces.Dict(
                {
                    "query_status": spaces.MultiDiscrete(np.ones((self.query_num,), dtype=np.int64) * (2 + self.max_worker)),
                    "query_embeddings": spaces.Box(min(0, self.query_embeddings.min()), self.query_embeddings.max(), (329,)),
                }
            )
        elif self.enable_resource:
            if self.local_resource:
                result = get_local_resource()
            else:
                mach = SshMachine(self.conf['database']['host'], user="root", password="123456")
                self.server = DeployedServer(mach)
                self.conn_ssh = self.server.classic_connect()
                result = get_remote_resource(self.conn_ssh)
            self.observation_space = spaces.Dict(
                {
                    "query_status": spaces.MultiDiscrete(np.ones((self.query_num,), dtype=np.int64) * (2 + self.max_worker)),
                    "cpu_times_percent": spaces.Box(0, 100, (10,)),
                    "virtual_memory": spaces.Box(0, result['virtual_memory'][0], (9,)),
                    "swap_memory": spaces.Box(0, result['swap_memory'][0], (4,)),
                }
            )
        elif self.enable_time_last:
            # if self.overlap_cluster:
            #     self.observation_space = spaces.Dict(
            #         # {
            #         #     "query_status": spaces.MultiDiscrete(np.ones((self.query_num,), dtype=np.int64) * (2 + self.max_worker)),
            #         #     "time_last": spaces.Box(0, self.baseline, (self.query_num,)),
            #         #     "cluster_status": spaces.MultiDiscrete(np.ones((self.cluster_num,), dtype=np.int64) * (2 + self.max_worker)),
            #         #     "cluster_time_last": spaces.Box(0, self.baseline, (self.cluster_num,)),
            #         # }
            #         {
            #             "query_status": spaces.MultiDiscrete(np.ones((self.cluster_num,), dtype=np.int64) * (2 + self.max_worker)),
            #             "time_last": spaces.Box(0, self.baseline, (self.cluster_num,)),
            #         }
            #     )
            # else:
            self.observation_space = spaces.Dict(
                {
                    "query_status": spaces.MultiDiscrete(np.ones((self.query_num,), dtype=np.int64) * (2 + self.max_worker)),
                    "time_last": spaces.Box(0, self.baseline, (self.query_num,)),
                }
            )
        elif self.vary_conn_num:
            self.observation_space = spaces.Dict(
                {
                    "query_status": spaces.MultiDiscrete(np.ones((self.query_num,), dtype=np.int64) * (2 + self.max_worker)),
                    "conn_status": spaces.MultiDiscrete([self.max_conn_num+1]*(2 if self.compress_action_c else 3)),
                }
            )
        else:
            self.observation_space = spaces.MultiDiscrete(np.ones((self.query_num,), dtype=np.int64) * (2 + self.max_worker))

        if self.enable_worker:
            if self.action_flatten:
                self.action_space = spaces.Discrete(self.action_num * self.max_worker)
            else:
                self.action_space = spaces.MultiDiscrete(np.array([self.action_num, self.max_worker]))
        else:
            self.action_space = spaces.Discrete(self.action_num)
        
        if self.vary_conn_num:
            self.action_worker_num = self.action_space.n
            if self.compress_action_c:
                self.action_space = spaces.Discrete(self.action_worker_num + 1)
            else:
                self.action_space = spaces.Discrete(self.action_worker_num + 3)
            # self.action_space = spaces.Dict(
            #     {
            #         "connection_action": spaces.Discrete(3),
            #         "query_action": self.action_space
            #     }
            # )


    def _get_obs(self):
        # return {"query_status": self._query_status}
        if self.enable_embedding:
            # executing_query_mask = np.logical_and(self._query_status>0, self._query_status<1+self.max_worker)
            # return {
            #     "query_status": self._query_status,
            #     "query_embeddings": np.zeros((329,)) if not max(executing_query_mask) \
            #         else self.query_embeddings[executing_query_mask].mean(axis=0)
            # }
            return {
                "query_status": self._query_status,
                "query_embeddings": self.query_embeddings.mean(axis=0)
            }
        elif self.enable_resource:
            resource_status = self._scheduler.resource_status
            return {
                "query_status": self._query_status,
                "cpu_times_percent": np.array(resource_status['cpu_times_percent']),
                "virtual_memory": np.array([resource_status['virtual_memory'][1], *resource_status['virtual_memory'][3:]]),
                "swap_memory": np.array([*resource_status['swap_memory'][1:3], *resource_status['swap_memory'][4:]]),
            }
        elif self.enable_time_last:
            # if self.overlap_cluster:
            #     query_time_last = self._scheduler.get_time_last()
            #     cluster_status, cluster_time_last = np.zeros((self.cluster_num, ), dtype=np.int64), np.zeros((self.cluster_num, ), dtype=query_time_last.dtype)
            #     for i in range(self.cluster_num):
            #         cluster_qposs = np.where(self.cluster_result==i)[0]
            #         cluster_query_status = self._query_status[cluster_qposs]
            #         min_status, max_status = cluster_query_status.min(), cluster_query_status.max()
            #         if min_status == self.max_worker + 1:
            #             cluster_status[i] = self.max_worker + 1
            #         elif max_status > 0:
            #             cluster_status[i] = max_status
            #             cluster_time_last = self._scheduler.get_cluster_time_last(cluster_qposs)
            #     # return {
            #     #     "query_status": self._query_status,
            #     #     "time_last": query_time_last,
            #     #     "cluster_status": cluster_status,
            #     #     "cluster_time_last": cluster_time_last
            #     # }
            #     return {
            #         "query_status": cluster_status,
            #         "time_last": cluster_time_last
            #     }
            # else:
            return {
                "query_status": self._query_status,
                "time_last": self._scheduler.get_time_last()
            }
        elif self.vary_conn_num:
            return {
                "query_status": self._query_status,
                "conn_status": [
                    len(self._scheduler.running_conn_ids),
                    len(self._scheduler.idle_conn_ids)
                ] if self.compress_action_c else [
                    len(self._scheduler.inactive_conn_ids),
                    len(self._scheduler.running_conn_ids),
                    len(self._scheduler.idle_conn_ids)
                ]
            }
        else:
            return self._query_status

    # def _get_info(self):
    #     return {"query_status": self._query_status}

    def reset(self):
        try:
            del self._scheduler.conn_list
            self._last_scheduler = self._scheduler
            # Delete old scheduler
            # del self._scheduler
        except:
            pass
        # Initialize scheduler, generator, return, and query status
        self._scheduler = QueryScheduler(conf=self.conf, runtime_log=self.runtime_log, enable_resource=self.enable_resource,
                                         conn_ssh=self.conn_ssh, local_resource=self.local_resource, args=self.args)
        self._scheduler_gen = self._scheduler.schedule_with_yield()
        self._scheduler_status = next(self._scheduler_gen)
        self._query_status = np.zeros((self.query_num,), dtype=np.int64)
        if self.overlap_cluster or self.worker_cluster:
            self._cluster_status = np.zeros((self.cluster_num, ), dtype=np.int64)

        observation = self._get_obs()
        # info = self._get_info()

        self.baseline_sum = 0

        return observation

    def step(self, action):
        # Parse action
        if self.vary_conn_num and action < (1 if self.compress_action_c else 3):
            self._scheduler.action_c = 1 if self.compress_action_c else action
            self._scheduler_status = next(self._scheduler_gen)
        else:
            if self.vary_conn_num:
                action -= (1 if self.compress_action_c else 3)
            if self.enable_worker:
                if self.action_flatten:
                    action_q = action % self.action_num
                    action_w = action // self.action_num
                else:
                    action_q = action[0]
                    action_w = action[1]
            else:
                action_q = action
            # Invalid action
            if not self.overlap_cluster and not self.worker_cluster and (self._scheduler.query_list.query_scale == 1 or not self._scheduler.query_cluster) \
                and self._query_status[action_q] != 0:
                # Negative reward and early termination
                return self._get_obs(), -1e4, True, {}
            self._scheduler.next_qpos = action_q
            if self.enable_worker:
                self._scheduler.next_worker = action_w
            # Execute the query and get return value from generator
            if self.vertical_cluster:
                for i in range(self.query_scale):
                    self._scheduler.next_qpos = action_q + i * self.action_num
                    self._scheduler_status = next(self._scheduler_gen)
            elif self.overlap_cluster or self.worker_cluster:
                assert self._cluster_status[action_q] == 0, 'selected cluster already executed'
                self._cluster_status[action_q] = 1
                cluster_queries = np.where(self.cluster_result == action_q)[0]
                for qpos in cluster_queries:
                    self._scheduler.next_qpos = qpos
                    if self.overlap_cluster:
                        if self.enable_worker:
                            if self.query_masks[action_w][qpos]:
                                self._scheduler.next_worker = action_w
                            else:
                                self._scheduler.next_worker = 0
                        # else:
                        #     self._scheduler.next_worker = self.query_workers[qpos]
                    self._scheduler_status = next(self._scheduler_gen)
            else:
                self._scheduler_status = next(self._scheduler_gen)
        # Update query status
        if self._scheduler_status == 1:
            self._query_status = np.ones((self.query_num,), dtype=np.int64) * (1 + self.max_worker)
        else:
            self._query_status = np.zeros((self.query_num,), dtype=np.int64)
            if not self._scheduler.query_cluster or self._scheduler.query_list.query_scale == 1:
                self._query_status[np.array(self._scheduler.executing_qposs, dtype=np.int64)] = \
                    1 if not self.enable_worker else [1 + self._scheduler.query_worker[i] for i in self._scheduler.executing_queries]
                self._query_status[np.array(self._scheduler.executed_qposs, dtype=np.int64)] = \
                    1 + self.max_worker
            else:
                self._query_status[np.array([self._scheduler.query_list.id_to_pos[(qid - 1) % self.query_num + 1] for qid in self._scheduler.executing_queries], dtype=np.int64)] = \
                    1 if not self.enable_worker else [1 + self._scheduler.query_worker[i] for i in self._scheduler.executing_queries]
                self._query_status[np.array([self._scheduler.query_list.id_to_pos[qid] for qid in self._scheduler.scaled_executed_queries], dtype=np.int64)] = \
                    1 + self.max_worker

        # An episode is done if the scheduler generator returns 1
        terminated = self._scheduler_status == 1
        if self.reward_type == 'cost':
            reward = -self._scheduler.cost if terminated else 0 # Sparse rewards
        elif self.reward_type == 'cost_with_baseline':
            reward = self.baseline - self._scheduler.cost if terminated else 0  # Sparse rewards
        elif self.reward_type == 'throughput':
            reward = self.query_num / self._scheduler.cost if terminated else \
                self._scheduler.get_throughput() / 990  # Sparse rewards with shaping
        elif self.reward_type == 'relative_time':
            reward = self.query_num / self._scheduler.cost if terminated else \
                self._scheduler.get_relative_time() / 100   # Sparse rewards with shaping
        elif self.reward_type == 'relative_time2':
            reward = self.query_num / self._scheduler.cost if terminated else \
                self._scheduler.get_relative_time() / 200   # Sparse rewards with shaping
        elif self.reward_type == 'relative_time_with_baseline':
            reward = self.baseline - self._scheduler.cost if terminated else \
                self._scheduler.get_relative_time() / 100   # Sparse rewards with shaping
        elif self.reward_type == 'delayed_relative_time_with_baseline':
            reward = self.baseline - self._scheduler.cost if terminated else 0  # Sparse rewards with shaping
        elif self.reward_type == 'relative_time_clamp_with_baseline':
            reward = self.baseline - self._scheduler.cost if terminated else \
                self._scheduler.get_relative_time_clamp() / 100 # Sparse rewards with shaping
        elif self.reward_type == 'relative_time_with_scaled_baseline':
            reward = self.baseline - self._scheduler.cost if terminated else \
                self._scheduler.get_relative_time() / (100 * sqrt(self._scheduler.query_list.query_scale))   # Sparse rewards with shaping
        elif self.reward_type == 'relative_time_with_baseline_and_trailing_punishment':
            reward = self.baseline - self._scheduler.cost - self._scheduler.mean_tailing if terminated else \
                self._scheduler.get_relative_time() / 100   # Sparse rewards with shaping
        elif self.reward_type == 'relative_absolute_time':
            reward = self.query_num / self._scheduler.cost if terminated else \
                (self._scheduler.get_relative_time() + \
                 self._scheduler.get_absolute_time()) / 100 # Sparse rewards with shaping
        elif self.reward_type == 'time_difference':
            reward = self.baseline / self.action_num - self._scheduler.get_time_diff()
        elif self.reward_type == 'relative_time_with_time_difference':
            reward = self.baseline / self.action_num - self._scheduler.get_time_diff() + self._scheduler.get_relative_time() / 100
        elif self.reward_type == 'relative_time_with_scaled_time_difference':
            reward = self.baseline / self.action_num - self._scheduler.get_time_diff() + self._scheduler.get_relative_time() / \
                (100 * sqrt(self._scheduler.query_list.query_scale))
        else:
            raise Exception(f'unrecognized reward type {self.reward_type}')
        observation = self._get_obs()
        # info = self._get_info()

        if terminated:
            return observation, reward, terminated, self._scheduler.info_aux
        else:
            return observation, reward, terminated, {}

    def render(self):
        pass

    def close(self):
        del self._scheduler
        if self.enable_resource:
            self.server.close()

    def valid_action_mask(self):
        if self.vary_conn_num:
            if self.compress_action_c:
                if len(self._scheduler.running_conn_ids) < self.min_conn_num:
                    conn_mask = [False]
                else:
                    conn_mask = [True]
                if len(self._scheduler.idle_conn_ids) == 0:
                    return np.array(conn_mask + [False] * self.action_worker_num)
                if self.enable_worker:
                    if self.action_flatten:
                        worker_mask = np.logical_and(self._scheduler.query_list.worker_masks[:self.max_worker], self._query_status==0).reshape(-1)
                        return np.concatenate([conn_mask, worker_mask])
            else:
                conn_mask = [True] * 3
                if self._scheduler.last_action_c == 2 or len(self._scheduler.inactive_conn_ids) == 0 or len(self._scheduler.idle_conn_ids) > 0:
                    conn_mask[0] = False
                if len(self._scheduler.running_conn_ids) == 0 or len(self._scheduler.idle_conn_ids) > 0:
                    conn_mask[1] = False
                if self._scheduler.last_action_c == 0 or len(self._scheduler.idle_conn_ids) == 0 or \
                    (len(self._scheduler.running_conn_ids) + len(self._scheduler.idle_conn_ids)) <= self.min_conn_num:
                    conn_mask[2] = False
                if len(self._scheduler.idle_conn_ids) == 0:
                    return np.array(conn_mask + [False] * self.action_worker_num)
                if self.enable_worker:
                    if self.action_flatten:
                        worker_mask = np.logical_and(self._scheduler.query_list.worker_masks[:self.max_worker], self._query_status==0).reshape(-1)
                        return np.concatenate([conn_mask, worker_mask])
        elif self.args.enable_soft_masking:
            return np.concatenate([self._query_status==0, self._scheduler.query_list.soft_worker_masks])
        elif self.vertical_cluster:
            if self.enable_worker:
                if self.action_flatten:
                    return np.logical_and(self._scheduler.query_list.worker_masks[:self.max_worker], (self._query_status==0))[:, :self.action_num].reshape(-1)
                else:
                    return list((self._query_status == 0)[:self.action_num]) + [True] * self.max_worker
            else:
                return (self._query_status == 0)[:self.action_num]
        elif self.overlap_cluster or self.worker_cluster:
            if self.enable_worker:
                if self.action_flatten:
                    return np.logical_and(self.cluster_masks[:self.max_worker], self._cluster_status==0).reshape(-1)
                else:
                    return list(self._cluster_status == 0) + [True] * self.max_worker
            else:
                return self._cluster_status == 0
        elif self._scheduler.query_list.query_scale == 1 or not self._scheduler.query_cluster:
            if self.enable_worker:
                if self.action_flatten:
                    return np.logical_and(self._scheduler.query_list.worker_masks[:self.max_worker], self._query_status==0).reshape(-1)
                else:
                    return list(self._query_status == 0) + [True] * self.max_worker
            else:
                return self._query_status == 0
        else:
            action_mask = np.array([True] * len(self._query_status))
            action_mask[np.array([self._scheduler.query_list.id_to_pos[qid] for qid in self._scheduler.scaled_executing_queries], dtype=np.int64)] = False
            action_mask[np.array([self._scheduler.query_list.id_to_pos[qid] for qid in self._scheduler.scaled_executed_queries], dtype=np.int64)] = False
            if self.enable_worker:
                if self.action_flatten:
                    return np.logical_and(self._scheduler.query_list.worker_masks[:self.max_worker], action_mask).reshape(-1)
                else:
                    return list(action_mask) + [True] * self.max_worker
            else:
                return action_mask


class MultiAgentQuerySchedulingEnv(QuerySchedulingEnv, AECEnv):

    def __init__(self, reward_type='cost', args=None, conf=None, embedding_path='nets/embeddings.npy', runtime_log=None):

        QuerySchedulingEnv.__init__(self, reward_type=reward_type, args=args, conf=conf, embedding_path=embedding_path, runtime_log=runtime_log)

        self.possible_agents = ["connection_controller", "query_selector"]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self._action_spaces = {
            "connection_controller": spaces.Discrete(2),
            "query_selector": spaces.Discrete(self.action_worker_num)
        }
        self._observation_spaces = {
            agent: self.observation_space for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def _get_maobs(self, observation=None):
        if observation is None:
            observation = self._get_obs()
        return {
            "connection_controller": {
                "observation": observation,
                "action_mask": [self.valid_action_mask[0], len(self._scheduler.running_conn_ids) < self.max_conn_num]
            },
            "query_selector": {
                "observation": observation,
                "action_mask": self.valid_action_mask[1:]
            }
        }
    
    def observe(self, agent):
        return self._get_maobs()[agent]
    
    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        observation = QuerySchedulingEnv.reset(self)
        # self.observations = {agent: observation for agent in self.agents}
        self.observations = self._get_maobs(observation)
        self.num_moves = 0
        self.agent_selection = "query_selector"

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        if self.agent_selection == "connection_controller":
            if action == 0:
                observation, reward, terminated, info = QuerySchedulingEnv.step(self, 0)
                self.agent_selection = "connection_controller"
            elif action == 1:
                observation, reward, terminated, info = self._get_obs(), 0, self._scheduler_status == 1, {}
                self.agent_selection = "query_selector"
            else:
                raise Exception(f"Invalid action {action} from agent {self.agent_selection}")
        elif self.agent_selection == "query_selector":
            observation, reward, terminated, info = QuerySchedulingEnv.step(self, action+1)
            if len(self._scheduler.running_conn_ids) < self.min_conn_num:
                self.agent_selection = "query_selector"
            else:
                self.agent_selection = "connection_controller"
        else:
            raise Exception(f"Invalid agent {self.agent_selection}")

        self._cumulative_rewards[agent] = 0

        # self.observations = {
        #     agent: observation for agent in self.agents
        # }
        self.observations = self._get_maobs(observation)
        self._clear_rewards()
        self.rewards[agent] = reward
        self._accumulate_rewards()
        self.terminations = {
            agent: terminated for agent in self.agents
        }
        self.truncations = {
            agent: False for agent in self.agents
        }
        self.infos = {
            agent: info for agent in self.agents
        }


class MultiTaskQuerySchedulingEnv(gym.Env):
    def __init__(self, envs, sample_strategy="round_robin"):
        self.sample_strategy = sample_strategy
        self.num_tasks = len(envs)
        self.active_task_index = None
        self.query_cluster = False

        self.query_num, self.query_nums = 0, []
        self.task_envs = []
        for env in envs:
            assert not env.query_cluster
            query_num = env.observation_space['query_status'].shape[0]
            if query_num > self.query_num:
                self.query_num = query_num
                self.observation_space = env.observation_space
                self.action_space = env.action_space
            self.query_nums.append(query_num)
            self.task_envs.append(env)
        
        observation_spaces = self.observation_space.spaces
        observation_spaces["task_id"] = spaces.Discrete(self.num_tasks)
        self.observation_space = spaces.Dict(observation_spaces)
        self.steps_per_run = sum([env.action_num for env in self.task_envs]) if sample_strategy == "round_robin" else None
    
    def sample_task(self):
        if self.sample_strategy == "round_robin":
            if self.active_task_index is None:
                return 0
            return (self.active_task_index + 1) % self.num_tasks
            # return 0
        else:
            raise ValueError(f"Not supported value {self.sample_strategy} for sample_strategy")
    
    def process_obs(self, observation):
        observation["task_id"] = self.active_task_index
        if self.env.action_num < self.query_num:
            observation["query_status"] = np.concatenate([
                observation["query_status"],
                np.array([self.observation_space["query_status"].nvec.max()-1] * (self.query_num-self.env.action_num), dtype=observation["query_status"].dtype)
            ])
            observation["time_last"] = np.concatenate([
                observation["time_last"],
                np.array([0.] * (self.query_num-self.env.action_num), dtype=observation["time_last"].dtype)
            ])
        return observation
    
    def reset(self):
        self.active_task_index = self.sample_task()
        self.env = self.task_envs[self.active_task_index]
        observation = self.env.reset()
        observation = self.process_obs(observation)
        return observation
    
    def step(self, action):
        if self.env.action_num < self.query_num:
            action_q, action_w = action % self.query_num, action // self.query_num
            action = action_q + action_w * self.env.action_num
        observation, reward, terminated, info = self.env.step(action)
        observation = self.process_obs(observation)
        if "observations" in info:
            obs_len = len(info['observations']['query_status'])
            info["observations"]["task_id"] = [np.array([self.active_task_index])] * obs_len
            if self.env.action_num < self.query_num:
                for i in range(obs_len):
                    info["observations"]["query_status"][i] = np.concatenate([
                        info["observations"]["query_status"][i],
                        np.array([self.observation_space["query_status"].nvec.max()-1] * (self.query_num-self.env.action_num), dtype=info["observations"]["query_status"][i].dtype)
                    ])
                    info["observations"]["time_last"][i] = np.concatenate([
                        info["observations"]["time_last"][i],
                        np.array([0.] * (self.query_num-self.env.action_num), dtype=info["observations"]["time_last"][i].dtype)
                    ])
        return observation, reward, terminated, info
    
    def render(self):
        pass

    def close(self):
        for env in self.task_envs:
            env.close()
    
    def valid_action_mask(self):
        valid_action_mask = self.env.valid_action_mask()
        if self.env.action_num < self.query_num:
            valid_action_mask = valid_action_mask.reshape(-1, self.env.action_num)
            valid_action_mask = np.concatenate([
                valid_action_mask,
                np.zeros((valid_action_mask.shape[0], self.query_num-self.env.action_num), dtype=valid_action_mask.dtype)
            ], axis=-1)
            valid_action_mask = valid_action_mask.reshape(-1)
        return valid_action_mask