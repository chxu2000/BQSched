import sys
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.ppo_mask.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from sb3_contrib.ppo_mask import MaskablePPO

from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

from copy import deepcopy
from envs.scheduler.query_scheduler import QueryScheduler
from utils.log_analyzer import *


SelfPretrainMaskablePPO = TypeVar("SelfPretrainMaskablePPO", bound="PretrainMaskablePPO")

class PretrainMaskablePPO(MaskablePPO):

    def __init__(
        self,
        policy: Union[str, Type[MaskableActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model
        )
    
    def log2buffer(
        self,
        args,
        conf,
        runtime_log_path: Union[None, str] = None,
    ):
        tconf = deepcopy(conf)
        tconf['database']['use_simulator'] = 'True'
        sim_scheduler = QueryScheduler(tconf)
        actions = log2action(runtime_log_path)[0]
        actions = actions[:args.pretrain_total_steps] if args.pretrain_total_steps > 0 else actions

        
        buffer_cls = MaskableDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else MaskableRolloutBuffer
        pretrain_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        return pretrain_buffer

    def pretrain(
        self,
        args,
        conf,
        runtime_log_path: str = "logs/tpcds/mppg_rsrtbs_qsched1X_wf_1/21.runtime.out",
	):
        pretrain_buffer = self.log2buffer(args, conf, runtime_log_path)
        print('test')
