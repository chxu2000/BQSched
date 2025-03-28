from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from stable_baselines3.common import logger
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, \
    Schedule
from stable_baselines3.common.utils import get_schedule_fn, \
    update_learning_rate
from stable_baselines3.ppo import PPO
from sb3_contrib.ppo_mask import MaskablePPO
import torch as th
from torch import distributions as td
from torch.nn import functional as F

from .policies import MaskableAuxActorCriticPolicy
from nets.policies import MaskableAuxActorCriticPolicySN

from gym import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from copy import deepcopy


class MaskablePPG(MaskablePPO):
    """
    Phasic Policy Gradient algorithm (PPG) (with PPO clip version)
    This version does not support a different number of policy and value
    optimization phases.
    Paper: https://arxiv.org/abs/2009.04416
    Code: This implementation borrows code from Stable Baselines 3
    (PPO from https://github.com/DLR-RM/stable-baselines3)
    and the OpenAI implementation
    (https://github.com/openai/phasic-policy-gradient)
    Introduction to PPO (which PPG improves upon):
    https://spinningup.openai.com/en/latest/algorithms/ppo.html
    :param policy: The policy model to use (AuxMlpPolicy, AuxCnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can
        be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param aux_learning_rate: The learning rate for the auxiliary optimizer,
        it can be a function of the current progress remaining (from 1 to 0).
        If None, use the same function as learning_rate.
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of
        environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the
        advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param aux_batch_size: Auxiliary minibatch size
    :param n_policy_iters: Number of policy phase optimization iterations
    :param n_epochs: Number of epochs when optimizing the surrogate loss
        (policy and value epochs are always the same)
    :param n_aux_epochs: Number of epochs for the auxiliary phase
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for
        Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current
        progress remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is
        passed (default), no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param beta_clone: Trade-off between optimizing auxiliary objective and
        original policy
    :param vf_true_coef: Non-auxiliary value function coefficient for the joint
        loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent
        Exploration (gSDE) instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when
        using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213
        (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None,
        no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when
        passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy
        on creation. Note that in the PPG paper, no activation function and a
        flat model was used.
    :param use_paper_parameters: Whether to overwrite the other parameters with
        those from the PPG paper
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the
        creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[MaskableAuxActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,  # 5e-4 in paper
        aux_learning_rate: Union[None, float, Schedule] = None,
        n_steps: int = 2048,  # 256 in paper
        batch_size: Optional[int] = 64,  # 8 in paper
        aux_batch_size: Optional[int] = 32,  # 4 in paper
        n_policy_iters: int = 10,  # 32 in paper
        n_epochs: int = 10,
        n_aux_epochs: int = 2,  # 6 in paper
        gamma: float = 0.99,  # 0.999 in paper
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,  # 0.01 in paper
        vf_coef: float = 0.5,
        beta_clone: float = 1.0,
        vf_true_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        use_paper_parameters: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        aux_cost_threshold: Union[None, float] = None,
        aux_least_episodes: int = 10,
        aux_value_phase: bool = True,
        aux_time_phase: Union[None, bool] = None,
        aux_time_learning_rate: Union[None, float, Schedule] = None,
        beta_clone_time: float = 1.0,
        aux_value_time: bool = False,
        aux_time_ind: bool = False,
        aux_vf_distance: bool = False,
        task_adaptive: bool = False,
        value_adaptive: bool = False,
        entropy_adaptive: bool = False,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        if aux_learning_rate is None:
            self.aux_learning_rate = learning_rate
        else:
            self.aux_learning_rate = aux_learning_rate
        self.aux_batch_size = aux_batch_size
        self.n_policy_iters = n_policy_iters
        self._curr_n_policy_iters = 0
        self.n_aux_epochs = n_aux_epochs
        self.beta_clone = beta_clone
        self.vf_true_coef = vf_true_coef
        self._n_aux_updates = 0
        self.aux_cost_threshold = aux_cost_threshold
        self.aux_least_episodes = aux_least_episodes
        self.buffer_index_start = 0
        self.aux_value_phase = aux_value_phase
        self.aux_time_phase = aux_time_phase
        if not self.aux_time_phase is None:
            self.policy_kwargs["aux_time_phase"] = self.aux_time_phase
            if aux_time_learning_rate is None:
                self.aux_time_learning_rate = learning_rate
            else:
                self.aux_time_learning_rate = aux_time_learning_rate
            self.aux_value_time = aux_value_time
            self.policy_kwargs["aux_value_time"] = self.aux_value_time
            self.aux_time_ind = aux_time_ind
            self.policy_kwargs["aux_time_ind"] = self.aux_time_ind
            self.aux_vf_distance = aux_vf_distance
        self.task_adaptive = task_adaptive
        self.value_adaptive = value_adaptive
        self.entropy_adaptive = entropy_adaptive

        if use_paper_parameters:
            self._set_paper_parameters()

        self._name2coef = {
            "pol_distance": beta_clone,
            "vf_true": vf_true_coef,
        }

        if self.aux_time_phase:
            self._name2coef_time = {
                "pol_distance": beta_clone_time,
                "vf_distance": beta_clone_time,
            }

        if _init_setup_model:
            self._setup_model()

    def _set_paper_parameters(self):
        if self.env.num_envs != 64:
            print("Warning: Paper uses 64 environments. "
                  "Change this if you want to have the same setup.")

        self.learning_rate = 5e-4
        self.aux_learning_rate = 5e-4
        self.n_steps = 256
        self.batch_size = 8
        self.aux_batch_size = 4
        self.n_policy_iters = 32
        self.n_epochs = 1
        self.n_aux_epochs = 6
        self.gamma = 0.999
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.clip_range_vf = None
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.beta_clone = 1.0
        self.vf_true_coef = 1.0
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.policy_kwargs["activation_fn"] = th.nn.Identity

    def _setup_model(self) -> None:
        self.aux_lr_schedule = get_schedule_fn(self.aux_learning_rate)
        self.policy_kwargs["aux_lr_schedule"] = self.aux_lr_schedule
        if self.aux_time_phase:
            self.aux_time_lr_schedule = get_schedule_fn(self.aux_time_learning_rate)
            self.policy_kwargs["aux_time_lr_schedule"] = self.aux_time_lr_schedule

        super()._setup_model()

        self._post_setup_model()

    def _post_setup_model(self) -> None:
        buffer_size = self.n_steps * self.n_envs * self.n_policy_iters
        if type(self.rollout_buffer.observation_space) is spaces.Dict:
            self.obs_keys = self.rollout_buffer.observations.keys()
            self._observations_buffer = {}
            for key in self.obs_keys:
                self._observations_buffer[key] = np.empty_like(
                    self.rollout_buffer.observations[key],
                    shape=(buffer_size,) + (self.rollout_buffer.observations[key].shape[-1],)
                )
        else:
            self._observations_buffer = np.empty_like(
                self.rollout_buffer.observations,
                shape=(buffer_size,) + self.rollout_buffer.obs_shape,
            )
        self._masks_buffer = np.empty_like(
            self.rollout_buffer.action_masks,
            dtype=bool,
            shape=(buffer_size, self.rollout_buffer.action_space.n),
        )
        self._returns_buffer = np.empty_like(
            self.rollout_buffer.returns,
            shape=(buffer_size, 1),
        )
        if self.aux_time_phase:
            self._observations_buffer_time = deepcopy(self._observations_buffer)
            # self._masks_buffer_time = deepcopy(self._masks_buffer)
            self._returns_buffer_time = deepcopy(self._returns_buffer)
            self._index_buffer_time = deepcopy(self._returns_buffer).astype(np.int32)
            self._time_buffer_time = deepcopy(self._returns_buffer)

    def _update_learning_rate(
            self,
            optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer],
    ) -> None:
        super()._update_learning_rate(optimizers)
        self.logger.record("train/aux_learning_rate",
                      self.aux_lr_schedule(self._current_progress_remaining))
        update_learning_rate(
            self.policy.aux_optimizer,
            self.aux_lr_schedule(self._current_progress_remaining))
        if self.aux_time_phase:
            update_learning_rate(
                self.policy.aux_time_optimizer,
                self.aux_time_lr_schedule(self._current_progress_remaining))

    def _excluded_save_params(self) -> List[str]:
        exclude = super()._excluded_save_params()
        exclude.extend([
            "_curr_n_policy_iters",
            "_observations_buffer",
            "_returns_buffer",
        ])
        return exclude

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, var_list = super()._get_torch_save_params()
        state_dicts.append("policy.aux_optimizer")
        return state_dicts, var_list
    
    def set_env(self, env, n_steps=None, force_reset=True):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.policy.observation_space = env.observation_space
        self.policy.action_space = env.action_space
        try:
            self.policy.features_extractor.query_num = env.observation_space['query_status'].shape[0]
        except Exception as e:
            self.policy.features_extractor.query_num = env.observation_space.shape[0]
        self.policy.action_dist.action_dim = env.action_space.n
        super().set_env(env, force_reset)

        if n_steps is not None:
            self.n_steps = n_steps
        buffer_cls = MaskableDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else MaskableRolloutBuffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        
        self._post_setup_model()
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        This method is largely identical to the implementation found in the parent class.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :param use_masking: Whether or not to use invalid action masks during training
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()
        if self.aux_time_phase:
            for key in self.obs_keys:
                self._observations_buffer_time[key].fill(0)
            self._returns_buffer_time.fill(0)
            self._index_buffer_time.fill(0)
            self._time_buffer_time.fill(0)

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        rollout_infos = []
        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # This is the only change related to invalid action masking
                if use_masking:
                    action_masks = get_action_masks(env)

                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            # self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # rollout_buffer.add(
            #     self._last_obs,
            #     actions,
            #     rewards,
            #     self._last_episode_starts,
            #     values,
            #     log_probs,
            #     action_masks=action_masks,
            # )
            rollout_infos.append({
                'infos': infos,
                '_last_obs': self._last_obs,
                'actions': actions,
                'rewards': rewards,
                '_last_episode_starts': self._last_episode_starts,
                'values': values,
                'log_probs': log_probs,
                'action_masks': action_masks
            })
            self._last_obs = new_obs
            self._last_episode_starts = dones

            for idx, done in enumerate(dones):
                sched_env = env.envs[0].env.env
                if not hasattr(sched_env, "baseline"):
                    sched_env = sched_env.env
                if (
                    done
                    and rewards[0] <= sched_env.baseline
                ):
                    if type(self.rollout_buffer.observations) == dict:
                        rollout_start_pos = self._curr_n_policy_iters * len(next(iter(self.rollout_buffer.observations.values())))
                    else:
                        rollout_start_pos = self._curr_n_policy_iters * len(self.rollout_buffer.observations)
                    start_pos = rollout_start_pos + rollout_buffer.pos
                    for info in rollout_infos:
                        if sched_env.reward_type == 'delayed_relative_time_with_baseline':
                            scheduler = sched_env._last_scheduler
                            qpos = info['actions'][0][0] % sched_env.action_num
                            try:
                                if self.policy.features_extractor.enable_cl:
                                    for qqpos in th.where(self.policy.features_extractor.cluster_result==qpos)[0]:
                                        info['rewards'][0] += scheduler.query_time_info['relative'][qqpos] * scheduler.query_list.costs[qqpos] / scheduler.total_time
                                else:
                                    info['rewards'][0] += scheduler.query_time_info['relative'][qpos] * scheduler.query_list.costs[qpos] / scheduler.total_time
                            except Exception as _:
                                info['rewards'][0] += scheduler.query_time_info['relative'][qpos] * scheduler.query_list.costs[qpos] / scheduler.total_time
                        ###
                        self._update_info_buffer(info['infos'])
                        rollout_buffer.add(
                            info['_last_obs'],
                            info['actions'],
                            info['rewards'],
                            info['_last_episode_starts'],
                            info['values'],
                            info['log_probs'],
                            action_masks=info['action_masks'],
                        )
                    end_pos = rollout_start_pos + rollout_buffer.pos
                    if self.aux_time_phase:
                        for key in self.obs_keys:
                            self._observations_buffer_time[key][start_pos:end_pos] = rollout_infos[-1]['infos'][0]['observations'][key]
                        self._index_buffer_time[start_pos:end_pos] = rollout_infos[-1]['infos'][0]['finish_qposs']
                        self._time_buffer_time[start_pos:end_pos] = rollout_infos[-1]['infos'][0]['finish_times']
                    rollout_infos = []
                elif (
                    done
                    and rewards[0] > sched_env.baseline
                ):
                    query_num = len(sched_env._scheduler.query_list.ids)
                    self.num_timesteps -= env.num_envs * query_num
                    n_steps -= query_num
                    callback.n_calls -= query_num
                    rollout_infos = []

        with th.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # super().train()
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            if self.task_adaptive:
                task_weightss = []
                with th.no_grad():
                    _, _, entropy = self.policy.evaluate_actions(
                        {key: th.tensor(obs, device=self.device) for (key, obs) in self.rollout_buffer.observations.items()},
                        th.tensor(self.rollout_buffer.actions, device=self.device),
                        action_masks=th.tensor(self.rollout_buffer.action_masks, device=self.device),
                        normalize_entropy=True,
                        query_nums=th.tensor(self.env.envs[0].unwrapped.query_nums, device=self.device)
                    )
                    mean_entropy = np.array([
                        entropy[np.where(self.rollout_buffer.observations["task_id"]==i)[0]].mean().item()
                        for i in range(self.env.envs[0].unwrapped.num_tasks)
                    ])
                    task_weights = mean_entropy / sum(mean_entropy) * self.env.envs[0].unwrapped.num_tasks
                    task_weightss.append(task_weights)
                    print(f"Train epoch={epoch}, task weights={task_weights}")
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2)
                if self.task_adaptive:
                    with th.no_grad():
                        task_ids = rollout_data.observations["task_id"].cpu().numpy().squeeze()
                        task_weights_ = th.tensor(task_weights[task_ids.astype(np.int64)], device=self.device)
                    policy_loss = policy_loss * task_weights_
                policy_loss = policy_loss.mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                if self.task_adaptive and self.value_adaptive:
                    value_loss = F.mse_loss(rollout_data.returns, values_pred, reduction="none")
                    value_loss = value_loss * task_weights_
                    value_loss = value_loss.mean()
                else:
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    if self.task_adaptive and self.entropy_adaptive:
                        entropy_loss = -th.mean(-log_prob * task_weights_)
                    else:
                        entropy_loss = -th.mean(-log_prob)
                else:
                    if self.task_adaptive and self.entropy_adaptive:
                        entropy_loss = -th.mean(entropy * task_weights_)
                    else:
                        entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if self.task_adaptive:
            task_weightss = np.array(task_weightss)
            for i in range(self.env.envs[0].unwrapped.num_tasks):
                self.logger.record(f"train/task_weight{i}", task_weightss[:, i].mean())

        # Auxiliary phase
        if self.aux_cost_threshold is None:
            if type(self.rollout_buffer.observation_space) is spaces.Dict:
                buffer_index_start = \
                    self._curr_n_policy_iters * len(next(iter(self.rollout_buffer.observations.values())))
                buffer_index_end = \
                    buffer_index_start + len(next(iter(self.rollout_buffer.observations.values())))
                for key in self.obs_keys:
                    self._observations_buffer[key][buffer_index_start:buffer_index_end] = \
                        self.rollout_buffer.observations[key]
            else:
                buffer_index_start = \
                    self._curr_n_policy_iters * len(self.rollout_buffer.observations)
                buffer_index_end = \
                    buffer_index_start + len(self.rollout_buffer.observations)
                self._observations_buffer[buffer_index_start:buffer_index_end] = \
                    self.rollout_buffer.observations
            self._masks_buffer[buffer_index_start:buffer_index_end] = \
                self.rollout_buffer.action_masks
            self._returns_buffer[buffer_index_start:buffer_index_end] = \
                self.rollout_buffer.returns

            self._curr_n_policy_iters += 1
            if self._curr_n_policy_iters < self.n_policy_iters:
                return

            if type(self.rollout_buffer.observation_space) is spaces.Dict:
                indices = np.arange(len(next(iter(self._observations_buffer.values()))))
                # In the paper, these are re-calculated after updating the policy
                old_pds = np.empty(len(next(iter(self._observations_buffer.values()))), dtype=object)
                if self.aux_time_phase:
                    old_pds_time = np.empty(len(next(iter(self._observations_buffer_time.values()))), dtype=object)
            else:
                indices = np.arange(len(self._observations_buffer))
                # In the paper, these are re-calculated after updating the policy
                old_pds = np.empty(len(self._observations_buffer), dtype=object)
                if self.aux_time_phase:
                    old_pds_time = np.empty(len(self._observations_buffer_time), dtype=object)
        else:
            sched_env = self.env.envs[0].env.env
            if not hasattr(sched_env, "baseline"):
                sched_env = sched_env.env
            episode_num = int(sum(self.rollout_buffer.episode_starts.reshape(-1)))
            episode_length = int(self.rollout_buffer.buffer_size / episode_num)
            selected_episodes = list(filter(lambda i:self.rollout_buffer.rewards[(i+1)*episode_length-1] > \
                                   sched_env.baseline - self.aux_cost_threshold, range(episode_num)))
            if len(selected_episodes) > 0:
                selected_index = [i * episode_length + j for j in range(episode_length) for i in selected_episodes]
                buffer_index_start = self.buffer_index_start
                buffer_index_end = buffer_index_start + len(selected_index)
                self._observations_buffer[buffer_index_start:buffer_index_end] = \
                    self.rollout_buffer.observations[selected_index]
                self._masks_buffer[buffer_index_start:buffer_index_end] = \
                    self.rollout_buffer.action_masks[selected_index]
                self._returns_buffer[buffer_index_start:buffer_index_end] = \
                    self.rollout_buffer.returns[selected_index]
                self.buffer_index_start = buffer_index_end

            self._curr_n_policy_iters += 1
            if self._curr_n_policy_iters < self.n_policy_iters or self.buffer_index_start / \
                episode_length < self.aux_least_episodes:
                return

            indices = np.arange(self.buffer_index_start)
            # In the paper, these are re-calculated after updating the policy
            old_pds = np.empty(self.buffer_index_start, dtype=object)
            if self.aux_time_phase:
                old_pds_time = np.empty(self.buffer_index_start, dtype=object)

        # aux_time_phase
        if self.aux_time_phase:
            with th.no_grad():
                start_idx = 0
                while start_idx < len(indices):
                    if self.use_sde:
                        self.policy.reset_noise(self.aux_batch_size)

                    batch_indices = indices[
                        start_idx:start_idx + self.aux_batch_size]
                    if type(self.rollout_buffer.observation_space) is spaces.Dict:
                        obs = {}
                        for key in self.obs_keys:
                            obs[key] = self._observations_buffer_time[key][batch_indices]
                    else:
                        obs = self._observations_buffer_time[batch_indices]
                    # mask = self._masks_buffer[batch_indices]
                    # Convert to pytorch tensor
                    if type(self.rollout_buffer.observation_space) is spaces.Dict:
                        obs_tensor = {}
                        for key in self.obs_keys:
                            obs_tensor[key] = th.as_tensor(obs[key]).to(self.policy.device)
                    else:
                        obs_tensor = th.as_tensor(obs).to(self.policy.device)
                    # mask_tensor = th.as_tensor(mask).to(self.policy.device)
                    if self.aux_value_time:
                        _, values, _ = self.policy.forward(obs_tensor)
                        self._returns_buffer_time[start_idx:start_idx+self.aux_batch_size] = values.cpu().numpy()
                    else:
                        if isinstance(self.policy, MaskableAuxActorCriticPolicySN):
                            distribution, _, latent_vf, _, _ = self.policy.forward_policy(obs_tensor)
                        else:
                            distribution, _, latent_vf = self.policy.forward_policy(obs_tensor)
                        old_pds_time[start_idx // self.aux_batch_size] = \
                            distribution.distribution
                        if self.aux_vf_distance:
                            values = self.policy.value_net(latent_vf)
                            self._returns_buffer_time[start_idx:start_idx+self.aux_batch_size] = values.cpu().numpy()
                    
                    start_idx += self.aux_batch_size
            
            unscaled_aux_time_losses = []
            aux_time_losses = []
            for _ in range(self.n_aux_epochs):
                start_idx = 0
                while start_idx < len(indices):
                    if self.use_sde:
                        self.policy.reset_noise(self.aux_batch_size)

                    batch_indices = indices[
                        start_idx:start_idx + self.aux_batch_size]
                    if type(self.rollout_buffer.observation_space) is spaces.Dict:
                        obs = {}
                        for key in self.obs_keys:
                            obs[key] = self._observations_buffer_time[key][batch_indices]
                    else:
                        obs = self._observations_buffer_time[batch_indices]
                    if type(self.rollout_buffer.observation_space) is spaces.Dict:
                        obs_tensor = {}
                        for key in self.obs_keys:
                            obs_tensor[key] = th.as_tensor(obs[key]).to(self.policy.device)
                    else:
                        obs_tensor = th.as_tensor(obs).to(self.policy.device)
                    old_pds_batch = old_pds_time[start_idx // self.aux_batch_size]

                    distribution, value, aux_time = self.policy.forward_aux_time(obs_tensor)
                    itarg = self._index_buffer_time[batch_indices]
                    itarg = th.as_tensor(itarg).to(self.policy.device)
                    ttarg = self._time_buffer_time[batch_indices]
                    ttarg = th.as_tensor(ttarg).to(self.policy.device)

                    name2loss_time = {}
                    if self.aux_value_time:
                        vtarg = self._returns_buffer_time[batch_indices]
                        vtarg = th.as_tensor(vtarg).to(self.policy.device)
                        name2loss_time["vf_distance"] = 0.5 * F.mse_loss(value, vtarg)
                    else:
                        aux_time = aux_time[th.arange(aux_time.shape[0]), itarg.squeeze(), :]
                        name2loss_time["pol_distance"] = td.kl_divergence(
                            old_pds_batch, distribution.distribution).mean()
                        if self.aux_vf_distance:
                            vtarg = self._returns_buffer_time[batch_indices]
                            vtarg = th.as_tensor(vtarg).to(self.policy.device)
                            name2loss_time["vf_distance"] = 0.5 * F.mse_loss(value, vtarg)
                    name2loss_time["time_aux"] = 0.5 * F.mse_loss(aux_time, ttarg)

                    unscaled_losses = {}
                    losses = {}
                    loss = 0
                    for name in name2loss_time.keys():
                        unscaled_loss = name2loss_time[name]
                        coef = self._name2coef_time.get(name, 1)

                        scaled_loss = unscaled_loss * coef
                        unscaled_losses[name] = \
                            unscaled_loss.detach().cpu().numpy()
                        losses[name] = scaled_loss.detach().cpu().numpy()
                        loss += scaled_loss
                    unscaled_aux_time_losses.append(unscaled_losses)
                    aux_time_losses.append(losses)

                    self.policy.aux_time_optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                self.max_grad_norm)
                    self.policy.aux_time_optimizer.step()
                    start_idx += self.aux_batch_size

        if self.aux_value_phase:
            with th.no_grad():
                start_idx = 0
                while start_idx < len(indices):
                    if self.use_sde:
                        self.policy.reset_noise(self.aux_batch_size)

                    batch_indices = indices[
                        start_idx:start_idx + self.aux_batch_size]
                    if type(self.rollout_buffer.observation_space) is spaces.Dict:
                        obs = {}
                        for key in self.obs_keys:
                            obs[key] = self._observations_buffer[key][batch_indices]
                    else:
                        obs = self._observations_buffer[batch_indices]
                    mask = self._masks_buffer[batch_indices]
                    # Convert to pytorch tensor
                    if type(self.rollout_buffer.observation_space) is spaces.Dict:
                        obs_tensor = {}
                        for key in self.obs_keys:
                            obs_tensor[key] = th.as_tensor(obs[key]).to(self.policy.device)
                    else:
                        obs_tensor = th.as_tensor(obs).to(self.policy.device)
                    mask_tensor = th.as_tensor(mask).to(self.policy.device)
                    if isinstance(self.policy, MaskableAuxActorCriticPolicySN):
                        distribution, _, _, _, _ = self.policy.forward_policy(obs_tensor, mask_tensor)
                    else:
                        distribution, _, _ = self.policy.forward_policy(obs_tensor, mask_tensor)
                    old_pds[start_idx // self.aux_batch_size] = \
                        distribution.distribution
                    
                    start_idx += self.aux_batch_size

            unscaled_aux_losses = []
            aux_losses = []
            for _ in range(self.n_aux_epochs):
                start_idx = 0
                while start_idx < len(indices):
                    if self.use_sde:
                        self.policy.reset_noise(self.aux_batch_size)

                    batch_indices = indices[
                        start_idx:start_idx + self.aux_batch_size]
                    if type(self.rollout_buffer.observation_space) is spaces.Dict:
                        obs = {}
                        for key in self.obs_keys:
                            obs[key] = self._observations_buffer[key][batch_indices]
                    else:
                        obs = self._observations_buffer[batch_indices]
                    mask = self._masks_buffer[batch_indices]
                    if type(self.rollout_buffer.observation_space) is spaces.Dict:
                        obs_tensor = {}
                        for key in self.obs_keys:
                            obs_tensor[key] = th.as_tensor(obs[key]).to(self.policy.device)
                    else:
                        obs_tensor = th.as_tensor(obs).to(self.policy.device)
                    mask_tensor = th.as_tensor(mask).to(self.policy.device)
                    old_pds_batch = old_pds[start_idx // self.aux_batch_size]

                    distribution, value, aux = self.policy.forward_aux(
                        obs_tensor, mask_tensor)
                    vtarg = self._returns_buffer[batch_indices]
                    vtarg = th.as_tensor(vtarg).to(self.policy.device)

                    name2loss = {}
                    name2loss["pol_distance"] = td.kl_divergence(
                        old_pds_batch, distribution.distribution).mean()
                    name2loss["vf_aux"] = 0.5 * F.mse_loss(aux, vtarg)
                    name2loss["vf_true"] = 0.5 * F.mse_loss(value, vtarg)

                    unscaled_losses = {}
                    losses = {}
                    loss = 0
                    for name in name2loss.keys():
                        unscaled_loss = name2loss[name]
                        coef = self._name2coef.get(name, 1)

                        scaled_loss = unscaled_loss * coef
                        unscaled_losses[name] = \
                            unscaled_loss.detach().cpu().numpy()
                        losses[name] = scaled_loss.detach().cpu().numpy()
                        loss += scaled_loss
                    unscaled_aux_losses.append(unscaled_losses)
                    aux_losses.append(losses)

                    self.policy.aux_optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                self.max_grad_norm)
                    self.policy.aux_optimizer.step()
                    start_idx += self.aux_batch_size

        self._n_aux_updates += self.n_aux_epochs
        self._curr_n_policy_iters = 0
        self.buffer_index_start = 0

        self.logger.record("train/n_aux_updates", self._n_aux_updates,
                      exclude="tensorboard")
        if self.aux_value_phase:
            for name in name2loss.keys():
                self.logger.record(f"train/unscaled_aux_{name}_loss", np.mean(
                    [entry[name] for entry in unscaled_aux_losses]))
                self.logger.record(f"train/aux_{name}_loss", np.mean(
                    [entry[name] for entry in aux_losses]))

        if self.aux_time_phase:
            for name in name2loss_time.keys():
                self.logger.record(f"train/unscaled_aux_time_{name}_loss", np.mean(
                    [entry[name] for entry in unscaled_aux_time_losses]))
                self.logger.record(f"train/aux_time_{name}_loss", np.mean(
                    [entry[name] for entry in aux_time_losses]))


    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MaskablePPG",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> "MaskablePPG":

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            use_masking=use_masking,
            progress_bar=progress_bar,
        )
