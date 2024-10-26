from typing import Dict, List, Tuple, Type, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many


class FeaturesExtractor(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(FeaturesExtractor, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_qsched = problem.NAME == 'qsched'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        assert problem.NAME == "qsched", "Unsupported problem: {}".format(problem.NAME)
        node_dim = 329  # query embedding

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

    def forward(self, input):

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self.init_embed(input))
        else:
            embeddings, _ = self.embedder(self.init_embed(input))

        # state = self.problem.make_state(input)

        return embeddings

        # return self.init_embed(input)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_qsched = problem.NAME == 'qsched'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        assert problem.NAME == "tsp" or problem.NAME == "qsched", "Unsupported problem: {}".format(problem.NAME)
        step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        
        # # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        # self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        # self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # # self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        # assert embedding_dim % n_heads == 0
        # # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        # self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.project_out = nn.Linear(embedding_dim*99, 99)

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        # if self.mask_inner:
        #     assert self.mask_logits, "Cannot mask inner without masking logits"
        #     compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        # if self.mask_logits:
        #     logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed):

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _get_logits(self, embeddings):

        # # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        # fixed = self._precompute(embeddings)

        # # Compute query = context node embedding
        # query = fixed.context_node_projected

        # # Compute keys and values for the nodes
        # glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        # # Compute the mask
        # # mask = state.get_mask()

        # # Compute logits (unnormalized log_p)
        # logits, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, None)

        # return logits

        embeddings = embeddings.view(-1, self.embedding_dim*99)
        return self.project_out(embeddings)


class AttentionExtractor(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        n_encode_layers=2,
        tanh_clipping=10.,
        mask_inner=True,
        mask_logits=True,
        normalization='batch',
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None
    ) -> None:
        super().__init__()
        self.policy_net = AttentionModel(
			embedding_dim,
			hidden_dim,
			problem,
			n_encode_layers=n_encode_layers,
			mask_inner=mask_inner,
			mask_logits=mask_logits,
			normalization=normalization,
			tanh_clipping=tanh_clipping,
            n_heads=n_heads,
			checkpoint_encoder=checkpoint_encoder,
			shrink_size=shrink_size
		)
        self.value_net = AttentionModel(
			embedding_dim,
			hidden_dim,
			problem,
			n_encode_layers=n_encode_layers,
			mask_inner=mask_inner,
			mask_logits=mask_logits,
			normalization=normalization,
			tanh_clipping=tanh_clipping,
            n_heads=n_heads,
			checkpoint_encoder=checkpoint_encoder,
			shrink_size=shrink_size
		)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net._get_logits(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net._get_logits(features)

