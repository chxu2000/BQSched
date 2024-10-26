import warnings, math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device
from torch import nn
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.options import get_options
from utils import torch_load_cpu, load_problem, move_to
from nets.attention_model import FeaturesExtractor, AttentionExtractor
from nets.mha_encoder import EncoderLayer


def soft_masking(distribution, masks):
    # MaskableCategoricalDistribution
    assert distribution.distribution is not None, "Must set distribution parameters"
    # MaskableCategorical
    if masks is not None:
        device = distribution.distribution.logits.device
        
        logits_len = distribution.distribution.logits.shape[1]
        masks, soft_masks = masks[:, :logits_len//2], masks[:, logits_len//2:]
        if not type(masks) == th.Tensor:
            masks, soft_masks = th.Tensor(masks), th.Tensor(soft_masks)
        masks = th.cat([masks, masks], dim=-1)
        # soft_masks = th.as_tensor(th.cat([th.ones_like(soft_masks), soft_masks], dim=-1), # 1
        soft_masks = th.as_tensor(th.cat([th.zeros_like(soft_masks), soft_masks], dim=-1),  # 2
                                  dtype=distribution.distribution.logits.dtype, device=device)
        # logits = torch.where(distribution.distribution._original_logits > 0,
        #                      distribution.distribution._original_logits * soft_masks,
        #                      distribution.distribution._original_logits / soft_masks)     # 1
        logits = distribution.distribution._original_logits + soft_masks    # 2

        distribution.distribution.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(distribution.distribution.logits.shape)
        HUGE_NEG = th.tensor(-1e8, dtype=distribution.distribution.logits.dtype, device=device)

        logits = th.where(distribution.distribution.masks, logits, HUGE_NEG)
    else:
        distribution.distribution.masks = None
        logits = distribution.distribution._original_logits

    # Reinitialize with updated logits
    Categorical.__init__(distribution.distribution, logits=logits)

    # self.probs may already be cached, so we must force an update
    distribution.distribution.probs = logits_to_probs(distribution.distribution.logits)


class MaskableActorCriticPolicyMHA(MaskableActorCriticPolicy):
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        preprocess_obs: bool = True,
        hidden_dim: int = 256,
        embedding_path: str = 'nets/embeddings.npy',
    ):
        self.hidden_dim = hidden_dim
        self.embedding_path = embedding_path
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            preprocess_obs,
        )

    def extract_latent_pi(
        self,
        features: th.Tensor,
    ) -> th.Tensor:
        features_concat = th.cat([features[:, 0, :].unsqueeze(1).repeat(1, features.shape[1] - 1, 1), 
                                  features[:, 1:, :]], dim=2)
        return features_concat

    def extract_latent_vf(
        self,
        features: th.Tensor,
    ) -> th.Tensor:
        return features[:, 0, :]
    
    def extract_latents(
        self,
        features: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        return self.extract_latent_pi(features), self.extract_latent_vf(features)
    
    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.extract_latents(features)
        values = self.value_net(latent_vf).squeeze(1)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        del self.pi_features_extractor, self.vf_features_extractor

        query_num = self.observation_space.shape[0]
        status_num = self.observation_space[0].n
        # Initialize query embeddings
        embeddings = list(np.load(self.embedding_path))
        self.query_embeddings = th.FloatTensor(np.array(embeddings * math.ceil(query_num/len(embeddings)))
                                               [:query_num])[None, :, :].to('cuda')
        embedding_dim = self.query_embeddings.shape[-1]
        # self.status_embed = nn.Embedding(status_num, embedding_dim)
        self.pre_embed = nn.Sequential(
            nn.Linear(embedding_dim + status_num, self.hidden_dim),
            self.activation_fn()
        )
        self.mha_encoder = EncoderLayer(self.hidden_dim, 128, 0.1, 0.1, 4)
        self.post_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_fn()
        )
        self.action_net = nn.Linear(self.hidden_dim * 2, self.action_space.n // query_num)
        self.value_net = nn.Linear(self.hidden_dim, 1)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits_layered = self.action_net(latent_pi)
        action_logits = th.cat([action_logits_layered[:, :, i] for i in range(action_logits_layered.shape[2])], dim=1)
        return self.action_dist.proba_distribution(action_logits=action_logits)
    
    def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.
        :param obs: Observation
        :return: the output of the features extractor(s)
        """
        embeddings = self.pre_embed(torch.cat([self.query_embeddings.repeat(obs.shape[0], 1, 1), 
                                               nn.functional.one_hot(obs.to(dtype=th.int64), num_classes=self.observation_space[0].n)], dim=2))
        # embeddings = self.query_embeddings + self.status_embed(obs.to(dtype=th.int64))
        features = self.post_embed(self.mha_encoder(torch.cat([embeddings.mean(dim=1)[:, None, :], embeddings], dim=1)))
        return features
    
    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.extract_latents(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf).squeeze(1)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.extract_latent_pi(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.extract_latent_vf(features)
        return self.value_net(latent_vf).squeeze(1)


class MaskableAttentionActorCriticPolicy(MaskableActorCriticPolicy):
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
    
    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        embeddings = self.extract_features(obs)
        action_logits, value_logits = self.attention_extractor(embeddings)
        # Evaluate the values for the given observations
        values = self.value_net(value_logits)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # =================================
        # if obs.max().item() == 0:
        #     print('action_logits', action_logits)
        # =================================
        return actions, values, log_prob

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        del self.pi_features_extractor, self.vf_features_extractor

        opts = get_options()
        # Set the device
        opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

        # Figure out what's the problem
        problem = load_problem(opts.problem)

        self.embeddings = None
        self.features_extractor = FeaturesExtractor(
			embedding_dim=opts.embedding_dim,
			hidden_dim=opts.hidden_dim,
			problem=problem,
			n_encode_layers=opts.n_encode_layers,
			mask_inner=False,
			mask_logits=False,
			normalization=opts.normalization,
			tanh_clipping=opts.tanh_clipping,
			checkpoint_encoder=opts.checkpoint_encoder,
			shrink_size=opts.shrink_size
		).to(opts.device)
        self.obs_emb = nn.Embedding(3, opts.embedding_dim)
        self.attention_extractor = AttentionExtractor(
			embedding_dim=opts.embedding_dim,
			hidden_dim=opts.hidden_dim,
			problem=problem,
			n_encode_layers=opts.n_encode_layers,
			mask_inner=False,
			mask_logits=False,
			normalization=opts.normalization,
			tanh_clipping=opts.tanh_clipping,
			checkpoint_encoder=opts.checkpoint_encoder,
			shrink_size=opts.shrink_size
		).to(opts.device)

        # Generate new training data for each epoch
        training_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
        training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
        for batch in training_dataloader:
            self.input = move_to(batch, opts.device)

        self.value_net = nn.Linear(99, 1)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.
        :param obs: Observation
        :return: the output of the features extractor(s)
        """
        if self.embeddings is None or obs.shape[0] != 1 or obs.max().item() == 0:
            self.embeddings = self.features_extractor(torch.cat([self.input] * obs.shape[0], axis=0))
        return self.embeddings + self.obs_emb(obs.to(dtype=th.int64))
    
    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        embeddings = self.extract_features(obs)
        action_logits, value_logits = self.attention_extractor(embeddings)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(value_logits)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        embeddings = self.extract_features(obs)
        action_logits = self.attention_extractor.forward_actor(embeddings)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        embeddings = self.extract_features(obs)
        value_logits = self.attention_extractor.forward_critic(embeddings)
        return self.value_net(value_logits)


class MaskableActorCriticPolicySN(MaskableActorCriticPolicy):
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        preprocess_obs: bool = True,
        cluster_num: int = None,
        soft_masking: bool = False,
    ):
        self.cluster_num = cluster_num
        self.soft_masking = soft_masking
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            preprocess_obs,
        )

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        # features = super().extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor.forward_actor(features[:, 1:, :]), self.mlp_extractor.forward_critic(features[:, 0, :])
        features_pi, features_vf = super().extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor.forward_actor(features_pi), self.mlp_extractor.forward_critic(features_vf)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            if self.soft_masking:
                soft_masking(distribution, action_masks)
            else:
                distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        del self.pi_features_extractor, self.vf_features_extractor

        if type(self.observation_space) == spaces.dict.Dict:
            self.query_num = self.observation_space['query_status'].shape[0] if self.cluster_num is None else self.cluster_num
            self.status_num = self.observation_space['query_status'][0].n
        else:
            self.query_num = self.observation_space.shape[0] if self.cluster_num is None else self.cluster_num
            self.status_num = self.observation_space[0].n

        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.n // self.query_num)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits_layered = self.action_net(latent_pi)
        # action_logits = action_logits_layered.reshape(action_logits_layered.shape[0], -1)
        # action_logits = th.cat([action_logits_layered[:, :, i] for i in range(action_logits_layered.shape[2])], dim=1)
        action_logits = action_logits_layered.transpose(1, 2).contiguous().view(action_logits_layered.shape[0], -1)
        return self.action_dist.proba_distribution(action_logits=action_logits)
    
    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # features = super().extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor.forward_actor(features[:, 1:, :]), self.mlp_extractor.forward_critic(features[:, 0, :])
        features_pi, features_vf = super().extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor.forward_actor(features_pi), self.mlp_extractor.forward_critic(features_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            if self.soft_masking:
                soft_masking(distribution, action_masks)
            else:
                distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        # features = super().extract_features(obs)
        # latent_pi = self.mlp_extractor.forward_actor(features[:, 1:, :])
        features_pi, _ = super().extract_features(obs)
        # print('features_pi', features_pi[:, :80, :])
        latent_pi = self.mlp_extractor.forward_actor(features_pi)
        # print('latent_pi', latent_pi[:, :80, :])
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            if self.soft_masking:
                soft_masking(distribution, action_masks)
            else:
                distribution.apply_masking(action_masks)
        return distribution
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        # features = super().extract_features(obs)
        # latent_vf = self.mlp_extractor.forward_critic(features[:, 0, :])
        _, features_vf = super().extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features_vf)
        return self.value_net(latent_vf)


class MaskableMultiInputActorCriticPolicySN(MaskableActorCriticPolicySN):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space (Tuple)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param features_extractor_class: Uses the CombinedExtractor
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        preprocess_obs: bool = True,
        cluster_num: int = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            preprocess_obs,
            cluster_num,
        )

class MaskableAuxActorCriticPolicySN(MaskableActorCriticPolicySN):
    def __init__(self, *args, **kwargs):
        aux_lr_schedule = kwargs.pop('aux_lr_schedule')
        self.aux_time_phase = kwargs.pop('aux_time_phase')
        aux_time_lr_schedule = kwargs.pop('aux_time_lr_schedule') if self.aux_time_phase else None
        self.emb_agg = kwargs.pop('emb_agg')
        self.aux_value_time = kwargs.pop('aux_value_time')
        self.aux_time_ind = kwargs.pop('aux_time_ind')
        assert not (self.aux_value_time and self.aux_time_ind)
        super().__init__(*args, **kwargs)
        lr_schedule = kwargs.get('lr_schedule', args[2])
        self._build(lr_schedule, aux_lr_schedule, aux_time_lr_schedule)

    def _build(
        self,
        lr_schedule,
        aux_lr_schedule=None,
        aux_time_lr_schedule=None,
    ):
        if aux_lr_schedule is None:
            return
        if self.aux_time_phase and aux_time_lr_schedule is None:
            return

        super()._build(lr_schedule)
        if self.emb_agg:
            init_dim = self.mlp_extractor.latent_dim_pi * self.query_num
            self.aux_head = nn.Sequential(
                nn.Linear(init_dim, init_dim // 10),
                nn.Tanh(),
                nn.Linear(init_dim // 10, 1),
            )
        else:
            self.aux_head = th.nn.Linear(self.mlp_extractor.latent_dim_pi, 1)
        self.aux_head.apply(lambda x: self.init_weights(x, gain=1))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.aux_optimizer = self.optimizer_class(
            self.parameters(), lr=aux_lr_schedule(1),
            **self.optimizer_kwargs)
        
        if self.aux_time_phase:
            if self.aux_time_ind:
                device = get_device(self.device)
                aux_net: List[nn.Module] = []
                last_layer_dim_aux = self.features_dim
                if isinstance(self.net_arch, dict):
                    aux_layers_dims = self.net_arch.get("pi", [])
                else:
                    aux_layers_dims = self.net_arch
                for curr_layer_dim in aux_layers_dims:
                    aux_net.append(nn.Linear(last_layer_dim_aux, curr_layer_dim))
                    aux_net.append(self.activation_fn())
                    last_layer_dim_aux = curr_layer_dim
                self.aux_net = nn.Sequential(*aux_net).to(device)
            if self.aux_value_time:
                self.aux_time_head = th.nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
            else:
                self.aux_time_head = th.nn.Linear(self.mlp_extractor.latent_dim_pi, 1)
            self.aux_time_head.apply(lambda x: self.init_weights(x, gain=1))
            self.aux_time_optimizer = self.optimizer_class(
                self.parameters(), lr=aux_time_lr_schedule(1),
                **self.optimizer_kwargs)

    def forward_policy(
        self,
        obs: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
        enable_va: bool = False,
        enable_aux: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in the actor network

        :param obs: Observation
        :param action_masks: Action masks to apply to the action distribution
        :return: action, latent policy vector and latent value vector
        """
        features_pi, features_vf = super().extract_features(obs)
        latent_pi, latent_vf, latent_va, latent_aux = self.mlp_extractor.forward_actor(features_pi), \
            self.mlp_extractor.forward_critic(features_vf), \
            self.mlp_extractor.forward_actor(features_vf) if enable_va and (not self.emb_agg) else None, \
            self.aux_net(features_pi) if enable_aux and self.aux_time_ind else None
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            if self.soft_masking:
                soft_masking(distribution, action_masks)
            else:
                distribution.apply_masking(action_masks)
        return distribution, latent_pi, latent_vf, latent_va, latent_aux
        # Preprocess the observation if needed
        # features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        #     latent_pi = self.mlp_extractor.forward_actor(pi_features)
        #     latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # # Evaluate the values for the given observations
        # distribution = self._get_action_dist_from_latent(latent_pi)
        # if action_masks is not None:
        #     distribution.apply_masking(action_masks)
        # return distribution, latent_pi, latent_vf

    def forward_aux_time(
        self,
        obs: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks, including the auxiliary value network

        :param obs: Observation
        :return: action, true value and auxiliary value
        """
        distribution, latent_pi, latent_vf, _, latent_aux = \
            self.forward_policy(obs, action_masks, enable_aux=True)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        if self.aux_time_ind:
            aux_times = self.aux_time_head(latent_aux)
        elif self.aux_value_time:
            aux_times = self.aux_time_head(latent_vf)
        else:
            aux_times = self.aux_time_head(latent_pi)
        return distribution, values, aux_times

    def forward_aux(
        self,
        obs: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks, including the auxiliary value network

        :param obs: Observation
        :return: action, true value and auxiliary value
        """
        distribution, latent_pi, latent_vf, latent_va, _ = \
            self.forward_policy(obs, action_masks, enable_va=True)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        if self.emb_agg:
            aux_values = self.aux_head(latent_pi.view(latent_pi.shape[0], -1))
        else:
            aux_values = self.aux_head(latent_va)
        return distribution, values, aux_values

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in the actor and critic networks

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        distribution, _, latent_vf, _, _ = self.forward_policy(obs, action_masks)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                # dummy lr schedule, not needed for loading policy alone
                aux_lr_schedule=self._dummy_schedule,
            )
        )
        return data


class MaskableMultiInputAuxActorCriticPolicySN(MaskableAuxActorCriticPolicySN):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            # features_extractor_class=CombinedExtractor
        )


def list_product(l):
    result = 1
    for e in l:
        result *= e
    return result

class HierarchicalMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        preprocess_obs: bool = True,
        hierarchical_mode: int = 1,
        layer_shapes: List[int] = None,
    ):
        assert list_product(layer_shapes) == action_space.n, 'product of layer shapes must equal to action_space.n'
        self.hierarchical_mode = hierarchical_mode
        self.layer_shapes = layer_shapes
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            preprocess_obs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)
        if self.hierarchical_mode == 1:
            self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, sum(self.layer_shapes))
        elif self.hierarchical_mode == 2:
            self.action_net = nn.Sequential(
                *[nn.Linear(self.mlp_extractor.latent_dim_pi + (sum(self.layer_shapes[:i]) if i > 0 else 0), layer_shape) 
                  for i, layer_shape in enumerate(self.layer_shapes)]
            )
            # self.action_net_high = nn.Linear(self.mlp_extractor.latent_dim_pi, self.layer_shapes[0])
            # self.action_net_low = nn.Linear(self.mlp_extractor.latent_dim_pi + self.layer_shapes[0], self.layer_shapes[1])
        else:
            raise Exception('Not supported hierarchical mode')
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        if self.hierarchical_mode == 1:
            layer_logits = th.split(self.action_net(latent_pi), self.layer_shapes, dim=1)
            for i in range(len(self.layer_shapes)):
                if i == 0:
                    action_logits = layer_logits[:][i]
                else:
                    action_logits = th.einsum('ni,nj->nij', [action_logits, layer_logits[:][i]])
                    action_logits = action_logits.view(action_logits.shape[0], -1)
            # action_logits = th.einsum('ni,nj->nij', th.split(self.action_net(latent_pi), self.layer_shapes, dim=1))
            # action_logits = action_logits.view(action_logits.shape[0], -1)
        elif self.hierarchical_mode == 2:
            layer_logits = []
            for i in range(len(self.layer_shapes)):
                layer_logits.append(self.action_net[i](th.cat([latent_pi] + layer_logits, dim=1)))
                if i == 0:
                    action_logits = layer_logits[-1]
                else:
                    action_logits = th.einsum('ni,nj->nij', [action_logits, layer_logits[-1]])
                    action_logits = action_logits.view(action_logits.shape[0], -1)
            # action_logits_high = self.action_net_high(latent_pi)
            # action_logits_low = self.action_net_low(th.cat([latent_pi, action_logits_high], dim=1))
            # action_logits = th.einsum('ni,nj->nij', [action_logits_high, action_logits_low])
            # action_logits = action_logits.view(action_logits.shape[0], -1)
        return self.action_dist.proba_distribution(action_logits=action_logits)