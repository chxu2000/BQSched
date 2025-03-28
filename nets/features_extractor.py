import gym
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from nets.graph_encoder import GraphAttentionEncoder, GraphAttentionEncoderWithMask
from nets.mha_encoder import EncoderLayer
from nets.queryformer import QueryFormer
from nets.model.database_util import Batch

from stable_baselines3.common.preprocessing import get_flattened_obs_dim

class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


class LinearExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 128, embedding_path: str = 'nets/embeddings.npy'):
        query_num = observation_space.shape[0]
        super().__init__(observation_space, query_num * features_dim)

        # Load query embeddings
        embeddings = list(np.load(embedding_path))
        self.query_embeddings = th.FloatTensor(np.array(embeddings * math.ceil(query_num/len(embeddings)))[:query_num])[None, :, :].to('cuda')
        node_dim = self.query_embeddings.shape[2]
        self.init_embed = nn.Linear(node_dim, features_dim)
        # Initialize embedders
        self.status_embedder = nn.Embedding(observation_space[0].n, embedding_dim=features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n_batch = observations.shape[0]
        status_embeddings = self.status_embedder(observations.to(dtype=th.int64))
        query_embeddings = self.init_embed(self.query_embeddings.repeat(n_batch, 1, 1))
        return (query_embeddings + status_embeddings).view(n_batch, -1)


class AttentionExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256, embedding_path: str = 'nets/embeddings.npy',
                 embedding_type: str = 'fixed', n_heads: int = 4, n_layers: int = 1, normalization: str = 'batch',
                 dim_reduction: str = 'multilayer_perceptron', pre_embed: bool = False, reduced_features_dim: int = 32,
                 train_qf_data: Batch = None, costsw: dict = None, enable_tl: bool = False, pe_dim: int = None,
                 share_features: bool = True, con_query_num: int = None, cluster_result = None, mask_done: bool = False,
                 feature_align: bool = False, info_agg: bool = False):
        super().__init__(observation_space, features_dim)
        self.embedding_type, self.dim_reduction, self.pre_embed, self.reduced_features_dim, self.pe_dim, self.share_features, self.con_query_num, self.mask_done, self.feature_align, self.info_agg = \
            embedding_type, dim_reduction, pre_embed, reduced_features_dim, pe_dim, share_features, con_query_num, mask_done, feature_align, info_agg
        self.enable_qf, self.enable_cw, self.enable_tl, self.enable_pe, self.enable_cq = not train_qf_data is None, not costsw is None, enable_tl, not self.pe_dim is None, not self.con_query_num is None
        self.enable_cqe, self.enable_cl = self.enable_cq and self.reduced_features_dim <= 64, not cluster_result is None
        if self.enable_tl:
            self.query_num, self.status_num = observation_space['query_status'].shape[0], observation_space['query_status'][0].n + 1
        else:
            self.query_num, self.status_num = observation_space.shape[0], observation_space[0].n
        if self.enable_cw:
            original_query_num = train_qf_data.attn_bias.shape[0]
            if not self.enable_cl:
                try:
                    self.costs_worker = [[costsw[f'{i} 1'][2] / costsw[f'{i} 1'][3], costsw[f'{i} 3'][2] / costsw[f'{i} 3'][3] \
                                          if f'{i} 3' in costsw.keys() else costsw[f'{i} 1'][2] / costsw[f'{i} 1'][3]] for i in range(1, original_query_num+1)]
                except:
                    try:
                        self.costs_worker = [[np.mean(costsw["max_parallel_workers_per_gather=0"][i]), \
                                              np.mean(costsw["max_parallel_workers_per_gather=2"][i])] for i in range(0, original_query_num)]
                    except:
                        self.costs_worker = [[np.mean(costsw["max_parallel_workers_per_gather=2"][i]), \
                                              np.mean(costsw["max_parallel_workers_per_gather=3"][i])] for i in range(0, original_query_num)]
            else:
                self.costs_worker = [[costsw[f'{i} 1'][2] / costsw[f'{i} 1'][3], costsw[f'{i} 3'][2] / costsw[f'{i} 3'][3] \
                                      if f'{i} 3' in costsw.keys() else costsw[f'{i} 1'][2] / costsw[f'{i} 1'][3]] for i in range(1, original_query_num+1)]
                self.costs_worker *= self.query_num // original_query_num
        if self.enable_qf:
            self.train_qf_data = train_qf_data
            self.train_qf_data.attn_bias = self.train_qf_data.attn_bias.to('cuda')
            self.train_qf_data.rel_pos = self.train_qf_data.rel_pos.to('cuda')
            self.train_qf_data.x = self.train_qf_data.x.to('cuda')
            self.train_qf_data.heights = self.train_qf_data.heights.to('cuda')
            self.query_former = QueryFormer(emb_size=32, ffn_dim=32, head_size=8, dropout=0.1,
                                            attention_dropout_rate=0.2, n_layers=2, use_sample=True,
                                            use_hist=True, bin_number=50, pred_hid=256)
            self.query_embeddings = None
            node_dim = self.query_former.hidden_dim
        else:
            # Initialize query embeddings
            embeddings = list(np.load(embedding_path))
            self.query_embeddings = th.FloatTensor(np.array(embeddings * math.ceil(self.query_num/len(embeddings)))
                                                   [:self.query_num])[None, :, :].to('cuda')
            node_dim = self.query_embeddings.shape[2]
        if self.enable_pe:
            self.pe_embedder = nn.Embedding(self.query_num, self.pe_dim)
        if self.enable_cl:
            self.cluster_result = th.tensor(cluster_result, dtype=th.int64, device='cuda')
            self.cluster_num = int(max(self.cluster_result) + 1)
        # self.init_embed = nn.Linear(node_dim, features_dim)
        # self.init_embed = nn.Sequential(
        #     nn.Linear(node_dim, features_dim),
        #     nn.Tanh(),
        # )
        if self.embedding_type == 'learnable':
            self.query_embeddings = nn.Parameter(self.query_embeddings)
        # Initialize encoders
        if self.pre_embed:
            self.pre_embedder = GraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=features_dim,
                n_layers=n_layers,
                normalization=normalization
            )
        # self.status_encoder = nn.Embedding(observation_space[0].n, embedding_dim=features_dim)
        # self.status_embed = nn.Linear(features_dim + self.status_num, features_dim)
        init_dim = node_dim + self.status_num
        if self.enable_cw: init_dim += 1
        if self.enable_pe: init_dim += self.pe_dim
        if self.reduced_features_dim < init_dim // 2:
            self.status_embed = nn.Sequential(
                nn.Linear(init_dim, self.reduced_features_dim * 2),
                nn.Tanh(),
                nn.Linear(self.reduced_features_dim * 2, self.reduced_features_dim),
                nn.Tanh(),
            )
        else:
            self.status_embed = nn.Sequential(
                nn.Linear(init_dim, self.reduced_features_dim),
                nn.Tanh(),
            )
        if self.dim_reduction == 'super_node':
            if self.info_agg:
                self.super_node_encoder_con = nn.Embedding(1, embedding_dim=self.reduced_features_dim)
                self.super_node_encoder_pen = nn.Embedding(1, embedding_dim=self.reduced_features_dim)
            else:
                self.super_node_encoder = nn.Embedding(1, embedding_dim=self.reduced_features_dim)
        if info_agg:
            self.embedder = GraphAttentionEncoderWithMask(
                n_heads=n_heads,
                embed_dim=self.reduced_features_dim,
                n_layers=n_layers,
                normalization=normalization
            )
        else:
            self.embedder = GraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=self.reduced_features_dim,
                n_layers=n_layers,
                normalization=normalization
            )
        # self.embedder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=reduced_features_dim,
        #         nhead=n_heads,
        #         batch_first=True,
        #     ), num_layers=n_layers
        # )
        if self.dim_reduction == 'multilayer_perceptron':
            # self.single_query_embed = nn.Sequential(
            #     nn.Linear(features_dim, features_dim // 2),
            #     nn.Tanh(),
            #     nn.Linear(features_dim // 2, reduced_features_dim),
            #     nn.Tanh(),
            # )
            self.global_embed = nn.Sequential(
                nn.Linear(self.query_num * self.reduced_features_dim, self.query_num * self.reduced_features_dim // 4),
                nn.Tanh(),
                nn.Linear(self.query_num * self.reduced_features_dim // 4, features_dim),
                nn.Tanh(),
            )
        elif self.dim_reduction == 'super_node':
            # self.obs_embed = nn.Sequential(
            #     nn.Linear(self.query_num * self.status_num, reduced_features_dim),
            #     nn.Tanh(),
            # )
            if self.share_features:
                self.output_embed = nn.Sequential(
                    nn.Linear(self.reduced_features_dim, features_dim),
                    nn.Tanh(),
                )
            else:
                if self.enable_cqe:
                    features_pi_dim = self.reduced_features_dim * (2 + self.con_query_num - 1)
                    self.output_embed_pi = nn.Sequential(
                        nn.Linear(features_pi_dim, features_pi_dim // 2),
                        nn.Tanh(),
                        nn.Linear(features_pi_dim // 2, features_dim),
                        nn.Tanh(),
                    )
                elif self.info_agg:
                    self.output_embed_pi = nn.Sequential(
                        nn.Linear(self.reduced_features_dim * 3, features_dim),
                        nn.Tanh(),
                        nn.Linear(features_dim, features_dim),
                        nn.Tanh(),
                    )
                else:
                    self.output_embed_pi = nn.Sequential(
                        # nn.Linear(self.reduced_features_dim * 2 + (self.con_query_num - 1) * 4 if self.enable_cq else self.reduced_features_dim * 2, features_dim),
                        nn.Linear(self.reduced_features_dim * 2 + self.con_query_num * 3 if self.enable_cq else self.reduced_features_dim * 2, features_dim),
                        nn.Tanh(),
                    )
                if not self.feature_align:
                    if self.info_agg:
                        features_vf_dim = self.reduced_features_dim * 2
                    else:
                        features_vf_dim = self.reduced_features_dim * 2 + self.query_num * (self.status_num + self.enable_cw)
                    self.output_embed_vf = nn.Sequential(
                        nn.Linear(features_vf_dim, features_dim),
                        nn.Tanh(),
                        nn.Linear(features_dim, features_dim),
                        nn.Tanh(),
                    ) if features_vf_dim / features_dim < 2 else (nn.Sequential(
                        nn.Linear(features_vf_dim, features_vf_dim // 2),
                        nn.Tanh(),
                        nn.Linear(features_vf_dim // 2, features_dim),
                        nn.Tanh(),
                    ) if features_vf_dim / features_dim < 20 else nn.Sequential(
                        nn.Linear(features_vf_dim, features_vf_dim // 5),
                        nn.Tanh(),
                        nn.Linear(features_vf_dim // 5, features_dim),
                        nn.Tanh(),
                    ))
        self.encoded_embeddings = None

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print('observations', observations)
        n_batch = observations['query_status'].shape[0] if self.enable_tl else observations.shape[0]
        # if self.encoded_embeddings is None or n_batch != 1 or observations.max().item() == 0:
        #     if self.pre_embed:
        #         self.encoded_embeddings, _ = self.pre_embedder(self.init_embed(self.query_embeddings))
        #     else:
        #         self.encoded_embeddings = self.init_embed(self.query_embeddings)
        if self.enable_qf and (self.query_embeddings is None or n_batch != 1 or self.query_embeddings.shape[0] != self.query_num):
            self.query_embeddings = self.query_former(self.train_qf_data.attn_bias, self.train_qf_data.rel_pos, self.train_qf_data.x,
                                                      self.train_qf_data.heights)
        # print('self.query_embeddings', self.query_embeddings)
        query_status_layered = observations['query_status'].view(n_batch, self.query_num, self.status_num - 1)
        if self.enable_tl:
            status_emb = th.cat([query_status_layered, observations['time_last'].view(n_batch, self.query_num, 1)], dim=2)
        else:
            status_emb = observations.view(n_batch, self.query_num, self.status_num)
        # if self.enable_cw:
        #     len_cw = len(self.costs_worker)
        #     if self.query_num <= len_cw:
        #         costs_worker = th.tensor([[self.costs_worker[j][1] if status_emb[i][j][2] else self.costs_worker[j][0] for j in range(self.query_num)]
        #                                    for i in range(n_batch)]).view(n_batch, self.query_num, 1).to('cuda')
        #     else:
        #         inc_num = self.query_num - len_cw
        #         range_j = list(range(len_cw)) + list(range(len_cw))[-inc_num:]
        #         costs_worker = th.tensor([[self.costs_worker[j][1] if status_emb[i][j][2] else self.costs_worker[j][0] for j in range_j]
        #                                 for i in range(n_batch)]).view(n_batch, self.query_num, 1).to('cuda')
        #         self.query_embeddings = th.cat([self.query_embeddings, self.query_embeddings[-inc_num:]], dim=0)
        inc_num = self.query_num - self.query_embeddings.shape[0]
        if self.query_num < self.query_embeddings.shape[0]:
            # self.query_embeddings = self.query_embeddings[:self.query_num]
            status_emb = th.cat([status_emb, th.tensor([[[1, 0, 0, 0, 0]]], dtype=status_emb.dtype, device=status_emb.device).repeat(n_batch, -inc_num, 1)], dim=1)
        elif self.enable_cl and self.query_num >= self.query_embeddings.shape[0] * 2:
            self.query_embeddings = self.query_embeddings.repeat(self.query_num // self.query_embeddings.shape[0], 1)[:self.query_num]
        elif self.query_num > self.query_embeddings.shape[0]:
            inc_num_, query_embeddings = inc_num, self.query_embeddings
            while inc_num_ > self.query_embeddings.shape[0]:
                query_embeddings = th.cat([query_embeddings, self.query_embeddings], dim=0)
                inc_num_ -= self.query_embeddings.shape[0]
            self.query_embeddings = th.cat([query_embeddings, self.query_embeddings[-inc_num_:]], dim=0)
        # print('status_emb', status_emb)
        status_embeddings = th.cat([self.query_embeddings.repeat(n_batch, 1, 1), status_emb], dim=2)
        original_query_num = len(self.costs_worker)
        inc_num = self.query_num - original_query_num
        if self.enable_cw:
            if self.query_num <= original_query_num:
                # costs_worker = th.tensor([[self.costs_worker[j][1] if status_emb[i][j][2] else self.costs_worker[j][0] for j in range(self.query_num)]
                #                           for i in range(n_batch)]).view(n_batch, self.query_num, 1).to('cuda')
                costs_worker = th.tensor([[self.costs_worker[j][1] if status_emb[i][j][2] else self.costs_worker[j][0] for j in range(original_query_num)]
                                          for i in range(n_batch)]).view(n_batch, original_query_num, 1).to(device='cuda', dtype=self.query_embeddings.dtype)
            else:
                inc_num_, range_j = inc_num, list(range(original_query_num))
                while inc_num_ > original_query_num:
                    range_j = range_j + list(range(original_query_num))
                    inc_num_ -= original_query_num
                range_j = range_j + list(range(original_query_num))[-inc_num_:]
                costs_worker = th.tensor([[self.costs_worker[j][1] if status_emb[i][j][2] else self.costs_worker[j][0] for j in range_j]
                                          for i in range(n_batch)]).view(n_batch, self.query_num, 1).to(device='cuda', dtype=self.query_embeddings.dtype)
            status_embeddings = th.cat([status_embeddings, costs_worker], dim=2)
        # print('costs_worker', costs_worker)
        if self.enable_pe:
            if self.query_num <= original_query_num:
                # positions = th.arange(0, self.query_num, dtype=th.int64, device='cuda').view(1, -1).repeat(n_batch, 1)
                positions = th.arange(0, original_query_num, dtype=th.int64, device='cuda').view(1, -1).repeat(n_batch, 1)
            else:
                positions = th.cat([th.arange(0, original_query_num, dtype=th.int64, device='cuda'),
                                    th.arange(original_query_num - inc_num, original_query_num, dtype=th.int64, device='cuda')],
                                    dim=0).view(1, -1).repeat(n_batch, 1)
            status_embeddings = th.cat([status_embeddings, self.pe_embedder(positions)], dim=2)
        # print('positions', positions)
        encoded_status_embeddings = self.status_embed(status_embeddings)
        # print('encoded_status_embeddings', encoded_status_embeddings)

        if self.dim_reduction == 'super_node' and self.info_agg:
            super_node_embedding_con = self.super_node_encoder_con.weight.unsqueeze(0).repeat(n_batch, 1, 1)
            super_node_embedding_pen = self.super_node_encoder_pen.weight.unsqueeze(0).repeat(n_batch, 1, 1)
            total_embeddings = th.cat([super_node_embedding_con, super_node_embedding_pen, encoded_status_embeddings], dim=1)

            indices_con = th.where((query_status_layered[:, :, 0] == 0) & (query_status_layered[:, :, -1] == 0))
            indices_pen = th.where(query_status_layered[:, :, 0] == 1)
            mask = th.ones((n_batch, total_embeddings.shape[1], total_embeddings.shape[1]), dtype=th.bool, device='cuda')
            mask[:, 0, 0] = 0       # sn_con2sn_con
            mask[:, 1, 1] = 0       # sn_pen2sn_pen
            for batch in range(n_batch):
                batch_indices_con = indices_con[1][indices_con[0]==batch] + 2
                mask[batch, 0, batch_indices_con] = 0   # sn_con2con
                mask[batch, batch_indices_con.unsqueeze(0), batch_indices_con.unsqueeze(1)] = 0     # con2con
                batch_indices_pen = indices_pen[1][indices_pen[0]==batch] + 2
                mask[batch, 1, batch_indices_pen] = 0   # sn_pen2pen
                # mask[batch, batch_indices_pen.unsqueeze(0), batch_indices_pen.unsqueeze(1)] = 0     # pen2pen

            output = self.embedder(total_embeddings, mask=mask)
            if type(output) != th.Tensor: output = output[0]
            output_vf = output[:, :2, :].view(n_batch, -1)
            output_pi = th.cat([output[:, :2, :].view(n_batch, 1, -1).repeat(1, output.shape[1], 1), output], dim=2)

            return self.output_embed_pi(output_pi[:, 2:2+self.query_num, :]), self.output_embed_vf(output_vf)
        elif self.dim_reduction == 'super_node':
            super_node_embedding = self.super_node_encoder.weight.unsqueeze(0).repeat(n_batch, 1, 1)
            if self.mask_done:
                indices = th.where(query_status_layered[:, :, -1] == 1)
                mask = th.ones_like(encoded_status_embeddings)
                mask[indices[0], indices[1], :] = 0
                encoded_status_embeddings = encoded_status_embeddings * mask
            total_embeddings = th.cat([super_node_embedding, encoded_status_embeddings], dim=1)
            output = self.embedder(total_embeddings)
            if type(output) != th.Tensor: output = output[0]
            if self.share_features:
                return self.output_embed(output[:, 0, :])
            else:
                # output = th.cat([output[:, 0:1, :].repeat(1, output.shape[1], 1),
                #                  th.cat([self.obs_embed(status_emb.view(n_batch, -1)).view(n_batch, 1, -1), output[:, 1:, :]], dim=1)], dim=2)
                # output = self.output_embed(output)
                query_status_raw = observations['query_status'].reshape(n_batch, self.query_num, -1).argmax(dim=2)
                cq_index_row, cq_index_col = th.where((query_status_raw==1)|(query_status_raw==2))
                if self.enable_cqe:
                    con_queries = th.zeros((n_batch, 1, (self.con_query_num - 1) * self.reduced_features_dim), device='cuda')
                    for i, count in enumerate(th.bincount(cq_index_row)):
                        col_index = cq_index_col[cq_index_row == i]
                        con_queries[i][0][:self.reduced_features_dim*col_index.shape[0]] = output[i, col_index+1, :].view(-1)
                    output = th.cat([output[:, 0:1, :].repeat(1, output.shape[1], 1), output], dim=2)
                    return self.output_embed_pi(th.cat([output[:, 1:, :], con_queries.repeat(1, self.query_num, 1)], dim=2)), \
                        self.output_embed_vf(th.cat([output[:, 0, :], th.cat([status_emb, costs_worker], dim=2).view(n_batch, -1)], dim=1))
                elif self.enable_cq:
                    # con_queries = th.zeros((n_batch, (self.con_query_num - 1) * 4), device='cuda')
                    con_queries = th.zeros((n_batch, self.con_query_num * 3), device='cuda')
                    costs_worker_tmp = costs_worker.squeeze(2)
                    for i, count in enumerate(th.bincount(cq_index_row)):
                        col_index = cq_index_col[cq_index_row == i]
                        # if self.query_num > original_query_num:
                        #     col_index_cpu = col_index.cpu().map_(col_index.cpu(), lambda x, _:(x if x < original_query_num else x - inc_num))
                        #     con_queries[i][(self.con_query_num-1)*0:(self.con_query_num-1)*0+count] = col_index_cpu.to(col_index.device)
                        # else:
                        #     con_queries[i][(self.con_query_num-1)*0:(self.con_query_num-1)*0+count] = col_index
                        # con_queries[i][(self.con_query_num-1)*1:(self.con_query_num-1)*1+count] = query_status_raw[i][col_index]
                        # con_queries[i][(self.con_query_num-1)*2:(self.con_query_num-1)*2+count] = observations['time_last'][i][col_index]
                        # con_queries[i][(self.con_query_num-1)*3:(self.con_query_num-1)*3+count] = costs_worker_tmp[i][col_index]
                        con_queries[i][self.con_query_num*0:self.con_query_num*0+count] = query_status_raw[i][col_index]
                        con_queries[i][self.con_query_num*1:self.con_query_num*1+count] = observations['time_last'][i][col_index]
                        con_queries[i][self.con_query_num*2:self.con_query_num*2+count] = costs_worker_tmp[i][col_index]
                    # if self.query_num > original_query_num:
                    #     output_pi = self.output_embed_pi(th.cat([output[:, 1:, :], con_queries.view(n_batch, 1, -1).repeat(1, self.query_num, 1)], dim=2))
                    #     output_pi = th.cat([output_pi[:, :original_query_num, :], output_pi[:, original_query_num-inc_num:original_query_num, :]], dim=1)
                    #     return output_pi, \
                    #         self.output_embed_vf(th.cat([output[:, 0, :], th.cat([status_emb, costs_worker], dim=2).view(n_batch, -1)], dim=1))
                    # else:
                    # print('con_queries', con_queries)
                    output = th.cat([output[:, 0:1, :].repeat(1, output.shape[1], 1), output], dim=2)
                    # print('output', output)

                    # if self.enable_cl:
                    #     output_pi = th.cat([output[:, 1:1+self.query_num, :], con_queries.view(n_batch, 1, -1).repeat(1, self.query_num, 1)], dim=2)
                    #     output_cl = th.zeros((output_pi.shape[0], self.cluster_num, output_pi.shape[2]), dtype=output_pi.dtype, device=output_pi.device)
                    #     idx = self.cluster_result.view(1, -1, 1).expand(output_pi.shape)
                    #     count = th.bincount(self.cluster_result, minlength=self.cluster_num).view(1, -1, 1).float()
                    #     output_cl.scatter_add_(1, idx, output_pi)
                    #     output_cl /= count
                    #     return self.output_embed_pi(output_cl), \
                    #         self.output_embed_vf(th.cat([output[:, 0, :], th.cat([status_emb, costs_worker], dim=2)[:, :self.query_num, :].view(n_batch, -1)], dim=1))
                    # else:
                    if self.feature_align:
                        output = self.output_embed_pi(th.cat([output, con_queries.view(n_batch, 1, -1).repeat(1, output.shape[1], 1)], dim=2))
                        return output[:, 1:, :], output[:, 0, :]
                    else:
                        return self.output_embed_pi(th.cat([output[:, 1:1+self.query_num, :], con_queries.view(n_batch, 1, -1).repeat(1, self.query_num, 1)], dim=2)), \
                            self.output_embed_vf(th.cat([output[:, 0, :], th.cat([status_emb, costs_worker], dim=2)[:, :self.query_num, :].view(n_batch, -1)], dim=1))
                else:
                    output = th.cat([output[:, 0:1, :].repeat(1, output.shape[1], 1), output], dim=2)
                    return self.output_embed_pi(output[:, 1:, :]), self.output_embed_vf(th.cat([output[:, 0, :], th.cat([status_emb, costs_worker], dim=2).view(n_batch, -1)], dim=1))
        elif self.dim_reduction == 'average':
            output, _ = self.embedder(encoded_status_embeddings)
            return output.mean(1)
        elif self.dim_reduction == 'multilayer_perceptron':
            output, _ = self.embedder(encoded_status_embeddings)
            # single_query_embeddings = self.single_query_embed(output)
            global_embedding = self.global_embed(output.view(n_batch, -1))
            return global_embedding


class AttentionExtractorLog2(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256, embedding_path: str = 'nets/embeddings.npy',
                 embedding_type: str = 'fixed', n_heads: int = 4, n_layers: int = 1, normalization: str = 'batch',
                 dim_reduction: str = 'multilayer_perceptron', pre_embed: bool = False, reduced_features_dim: int = 32,
                 train_qf_data: Batch = None, costsw: dict = None, enable_tl: bool = False):
        super().__init__(observation_space, features_dim)
        self.embedding_type, self.dim_reduction, self.pre_embed = embedding_type, dim_reduction, pre_embed
        self.enable_qf, self.enable_cw, self.enable_tl = not train_qf_data is None, not costsw is None, enable_tl
        if self.enable_tl:
            self.query_num, self.status_num = observation_space['query_status'].shape[0], observation_space['query_status'][0].n + 1
        else:
            self.query_num, self.status_num = observation_space.shape[0], observation_space[0].n
        if self.enable_cw:
            self.costs_worker = [[costsw[f'{i} 1'][2] / costsw[f'{i} 1'][3], costsw[f'{i} 3'][2] / costsw[f'{i} 3'][3] \
                                  if f'{i} 3' in costsw.keys() else costsw[f'{i} 1'][2] / costsw[f'{i} 1'][3]] for i in range(1, self.query_num+1)]
        if self.enable_qf:
            self.train_qf_data = train_qf_data
            self.train_qf_data.attn_bias = self.train_qf_data.attn_bias.to('cuda')
            self.train_qf_data.rel_pos = self.train_qf_data.rel_pos.to('cuda')
            self.train_qf_data.x = self.train_qf_data.x.to('cuda')
            self.train_qf_data.heights = self.train_qf_data.heights.to('cuda')
            self.query_former = QueryFormer(emb_size=32, ffn_dim=32, head_size=8, dropout=0.1,
                                            attention_dropout_rate=0.2, n_layers=2, use_sample=True,
                                            use_hist=True, bin_number=50, pred_hid=256)
            self.query_embeddings = None
            node_dim = self.query_former.hidden_dim
        else:
            # Initialize query embeddings
            embeddings = list(np.load(embedding_path))
            self.query_embeddings = th.FloatTensor(np.array(embeddings * math.ceil(self.query_num/len(embeddings)))
                                                   [:self.query_num])[None, :, :].to('cuda')
            node_dim = self.query_embeddings.shape[2]
        # self.init_embed = nn.Linear(node_dim, features_dim)
        # self.init_embed = nn.Sequential(
        #     nn.Linear(node_dim, features_dim),
        #     nn.Tanh(),
        # )
        if self.embedding_type == 'learnable':
            self.query_embeddings = nn.Parameter(self.query_embeddings)
        # Initialize encoders
        if self.pre_embed:
            self.pre_embedder = GraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=features_dim,
                n_layers=n_layers,
                normalization=normalization
            )
        # self.status_encoder = nn.Embedding(observation_space[0].n, embedding_dim=features_dim)
        # self.status_embed = nn.Linear(features_dim + self.status_num, features_dim)
        init_dim = node_dim + self.status_num
        if self.enable_cw: init_dim += 1
        if reduced_features_dim < init_dim // 2:
            self.status_embed = nn.Sequential(
                nn.Linear(init_dim, features_dim // 2),
                nn.Tanh(),
                nn.Linear(features_dim // 2, reduced_features_dim),
                nn.Tanh(),
            )
        else:
            self.status_embed = nn.Sequential(
                nn.Linear(init_dim, reduced_features_dim),
                nn.Tanh(),
            )
        if self.dim_reduction == 'super_node':
            self.super_node_encoder = nn.Embedding(1, embedding_dim=reduced_features_dim)
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=reduced_features_dim,
            n_layers=n_layers,
            normalization=normalization
        )
        if self.dim_reduction == 'multilayer_perceptron':
            # self.single_query_embed = nn.Sequential(
            #     nn.Linear(features_dim, features_dim // 2),
            #     nn.Tanh(),
            #     nn.Linear(features_dim // 2, reduced_features_dim),
            #     nn.Tanh(),
            # )
            self.global_embed = nn.Sequential(
                nn.Linear(self.query_num * reduced_features_dim, self.query_num * reduced_features_dim // 4),
                nn.Tanh(),
                nn.Linear(self.query_num * reduced_features_dim // 4, features_dim),
                nn.Tanh(),
            )
        elif self.dim_reduction == 'super_node':
            self.obs_embed = nn.Sequential(
                nn.Linear(self.query_num * self.status_num, reduced_features_dim),
                nn.Tanh(),
            )
            self.output_embed_pi = nn.Sequential(
                nn.Linear(reduced_features_dim * 2, features_dim),
                nn.Tanh(),
            )
            features_vf_dim = reduced_features_dim * 2 + self.query_num * (self.status_num + self.enable_cw)
            self.output_embed_vf = nn.Sequential(
                nn.Linear(features_vf_dim, features_vf_dim // 2),
                nn.Tanh(),
                nn.Linear(features_vf_dim // 2, features_dim),
                nn.Tanh(),
            )
        self.encoded_embeddings = None

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n_batch = observations['query_status'].shape[0] if self.enable_tl else observations.shape[0]
        # if self.encoded_embeddings is None or n_batch != 1 or observations.max().item() == 0:
        #     if self.pre_embed:
        #         self.encoded_embeddings, _ = self.pre_embedder(self.init_embed(self.query_embeddings))
        #     else:
        #         self.encoded_embeddings = self.init_embed(self.query_embeddings)
        if self.enable_qf and (self.query_embeddings is None or n_batch != 1):
            self.query_embeddings = self.query_former(self.train_qf_data.attn_bias, self.train_qf_data.rel_pos, self.train_qf_data.x,
                                                      self.train_qf_data.heights)
        if self.enable_tl:
            status_emb = th.cat([observations['query_status'].view(n_batch, self.query_num, self.status_num - 1),
                                 observations['time_last'].view(n_batch, self.query_num, 1)], dim=2)
        else:
            status_emb = observations.view(n_batch, self.query_num, self.status_num)
        len_cw = len(self.costs_worker)
        if self.query_num <= len_cw:
            costs_worker = th.tensor([[self.costs_worker[j][1] if status_emb[i][j][2] else self.costs_worker[j][0] for j in range(self.query_num)]
                                      for i in range(n_batch)]).view(n_batch, self.query_num, 1).to('cuda')
        else:
            inc_num = self.query_num - len_cw
            range_j = list(range(len_cw)) + list(range(len_cw))[-inc_num:]
            costs_worker = th.tensor([[self.costs_worker[j][1] if status_emb[i][j][2] else self.costs_worker[j][0] for j in range_j]
                                      for i in range(n_batch)]).view(n_batch, self.query_num, 1).to('cuda')
            self.query_embeddings = th.cat([self.query_embeddings, self.query_embeddings[-inc_num:]], dim=0)
        if self.enable_cw:
            encoded_status_embeddings = self.status_embed(th.cat([self.query_embeddings[:self.query_num].repeat(n_batch, 1, 1), costs_worker, status_emb], dim=2))
        else:
            encoded_status_embeddings = self.status_embed(th.cat([self.query_embeddings.repeat(n_batch, 1, 1), status_emb], dim=2))
        if self.dim_reduction == 'super_node':
            super_node_embedding = self.super_node_encoder.weight.unsqueeze(0).repeat(n_batch, 1, 1)
            total_embeddings = th.cat([super_node_embedding, encoded_status_embeddings], dim=1)
            output, _ = self.embedder(total_embeddings)
            # output = th.cat([output[:, 0:1, :].repeat(1, output.shape[1], 1),
            #                  th.cat([self.obs_embed(status_emb.view(n_batch, -1)).view(n_batch, 1, -1), output[:, 1:, :]], dim=1)], dim=2)
            output = th.cat([output[:, 0:1, :].repeat(1, output.shape[1], 1), output], dim=2)
            # output = self.output_embed(output)
            return self.output_embed_pi(output[:, 1:, :]), self.output_embed_vf(th.cat([output[:, 0, :], th.cat([status_emb, costs_worker], dim=2).view(n_batch, -1)], dim=1))
        elif self.dim_reduction == 'average':
            output, _ = self.embedder(encoded_status_embeddings)
            return output.mean(1)
        elif self.dim_reduction == 'multilayer_perceptron':
            output, _ = self.embedder(encoded_status_embeddings)
            # single_query_embeddings = self.single_query_embed(output)
            global_embedding = self.global_embed(output.view(n_batch, -1))
            return global_embedding


class AttentionExtractorLog1(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256, embedding_path: str = 'nets/embeddings.npy',
                 embedding_type: str = 'fixed', n_heads: int = 4, n_layers: int = 1, normalization: str = 'batch',
                 dim_reduction: str = 'multilayer_perceptron', pre_embed: bool = False, reduced_features_dim: int = 32,
                 train_qf_data: Batch = None, costsw: dict = None, enable_tl: bool = False):
        super().__init__(observation_space, features_dim)
        self.embedding_type, self.dim_reduction, self.pre_embed = embedding_type, dim_reduction, pre_embed
        self.enable_qf, self.enable_cw, self.enable_tl = not train_qf_data is None, not costsw is None, enable_tl
        if self.enable_tl:
            self.query_num, self.status_num = observation_space['query_status'].shape[0], observation_space['query_status'][0].n + 1
        else:
            self.query_num, self.status_num = observation_space.shape[0], observation_space[0].n
        if self.enable_cw:
            self.costs_worker = [[costsw[f'{i} 1'][2] / costsw[f'{i} 1'][3], costsw[f'{i} 3'][2] / costsw[f'{i} 3'][3] \
                                  if f'{i} 3' in costsw.keys() else costsw[f'{i} 1'][2] / costsw[f'{i} 1'][3]] for i in range(1, self.query_num+1)]
        if self.enable_qf:
            self.train_qf_data = train_qf_data
            self.train_qf_data.attn_bias = self.train_qf_data.attn_bias.to('cuda')
            self.train_qf_data.rel_pos = self.train_qf_data.rel_pos.to('cuda')
            self.train_qf_data.x = self.train_qf_data.x.to('cuda')
            self.train_qf_data.heights = self.train_qf_data.heights.to('cuda')
            self.query_former = QueryFormer(emb_size=32, ffn_dim=32, head_size=8, dropout=0.1,
                                            attention_dropout_rate=0.2, n_layers=2, use_sample=True,
                                            use_hist=True, bin_number=50, pred_hid=256)
            self.query_embeddings = None
            node_dim = self.query_former.hidden_dim
        else:
            # Initialize query embeddings
            embeddings = list(np.load(embedding_path))
            self.query_embeddings = th.FloatTensor(np.array(embeddings * math.ceil(self.query_num/len(embeddings)))
                                                   [:self.query_num])[None, :, :].to('cuda')
            node_dim = self.query_embeddings.shape[2]
        # self.init_embed = nn.Linear(node_dim, features_dim)
        # self.init_embed = nn.Sequential(
        #     nn.Linear(node_dim, features_dim),
        #     nn.Tanh(),
        # )
        if self.embedding_type == 'learnable':
            self.query_embeddings = nn.Parameter(self.query_embeddings)
        # Initialize encoders
        if self.pre_embed:
            self.pre_embedder = GraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=features_dim,
                n_layers=n_layers,
                normalization=normalization
            )
        # self.status_encoder = nn.Embedding(observation_space[0].n, embedding_dim=features_dim)
        # self.status_embed = nn.Linear(features_dim + self.status_num, features_dim)
        init_dim = node_dim + self.status_num
        if self.enable_cw: init_dim += 1
        if reduced_features_dim < init_dim // 2:
            self.status_embed = nn.Sequential(
                nn.Linear(init_dim, features_dim // 2),
                nn.Tanh(),
                nn.Linear(features_dim // 2, reduced_features_dim),
                nn.Tanh(),
            )
        else:
            self.status_embed = nn.Sequential(
                nn.Linear(init_dim, reduced_features_dim),
                nn.Tanh(),
            )
        if self.dim_reduction == 'super_node':
            self.super_node_encoder = nn.Embedding(1, embedding_dim=reduced_features_dim)
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=reduced_features_dim,
            n_layers=n_layers,
            normalization=normalization
        )
        if self.dim_reduction == 'multilayer_perceptron':
            # self.single_query_embed = nn.Sequential(
            #     nn.Linear(features_dim, features_dim // 2),
            #     nn.Tanh(),
            #     nn.Linear(features_dim // 2, reduced_features_dim),
            #     nn.Tanh(),
            # )
            self.global_embed = nn.Sequential(
                nn.Linear(self.query_num * reduced_features_dim, self.query_num * reduced_features_dim // 4),
                nn.Tanh(),
                nn.Linear(self.query_num * reduced_features_dim // 4, features_dim),
                nn.Tanh(),
            )
        elif self.dim_reduction == 'super_node':
            self.obs_embed = nn.Sequential(
                nn.Linear(self.query_num * self.status_num, reduced_features_dim),
                nn.Tanh(),
            )
            self.output_embed_pi = nn.Sequential(
                nn.Linear(reduced_features_dim * 2, features_dim),
                nn.Tanh(),
            )
            features_vf_dim = reduced_features_dim * 2 + self.query_num * (self.status_num + self.enable_cw)
            self.output_embed_vf = nn.Sequential(
                nn.Linear(features_vf_dim, features_vf_dim // 2),
                nn.Tanh(),
                nn.Linear(features_vf_dim // 2, features_dim),
                nn.Tanh(),
            )
        self.encoded_embeddings = None

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n_batch = observations['query_status'].shape[0] if self.enable_tl else observations.shape[0]
        # if self.encoded_embeddings is None or n_batch != 1 or observations.max().item() == 0:
        #     if self.pre_embed:
        #         self.encoded_embeddings, _ = self.pre_embedder(self.init_embed(self.query_embeddings))
        #     else:
        #         self.encoded_embeddings = self.init_embed(self.query_embeddings)
        if self.enable_qf and (self.query_embeddings is None or n_batch != 1):
            self.query_embeddings = self.query_former(self.train_qf_data.attn_bias, self.train_qf_data.rel_pos, self.train_qf_data.x,
                                                      self.train_qf_data.heights)
        if self.enable_tl:
            status_emb = th.cat([observations['query_status'].view(n_batch, self.query_num, self.status_num - 1),
                                 observations['time_last'].view(n_batch, self.query_num, 1)], dim=2)
        else:
            status_emb = observations.view(n_batch, self.query_num, self.status_num)
        len_cw = len(self.costs_worker)
        if self.query_num <= len_cw:
            costs_worker = th.tensor([[self.costs_worker[j][1] if status_emb[i][j][2] else self.costs_worker[j][0] for j in range(self.query_num)]
                                      for i in range(n_batch)]).view(n_batch, self.query_num, 1).to('cuda')
        else:
            inc_num = self.query_num - len_cw
            range_j = list(range(len_cw)) + list(range(len_cw))[-inc_num:]
            costs_worker = th.tensor([[self.costs_worker[j][1] if status_emb[i][j][2] else self.costs_worker[j][0] for j in range_j]
                                      for i in range(n_batch)]).view(n_batch, self.query_num, 1).to('cuda')
            self.query_embeddings = th.cat([self.query_embeddings, self.query_embeddings[-inc_num:]], dim=0)
        if self.enable_cw:
            encoded_status_embeddings = self.status_embed(th.cat([self.query_embeddings[:self.query_num].repeat(n_batch, 1, 1), costs_worker, status_emb], dim=2))
        else:
            encoded_status_embeddings = self.status_embed(th.cat([self.query_embeddings.repeat(n_batch, 1, 1), status_emb], dim=2))
        if self.dim_reduction == 'super_node':
            super_node_embedding = self.super_node_encoder.weight.unsqueeze(0).repeat(n_batch, 1, 1)
            total_embeddings = th.cat([super_node_embedding, encoded_status_embeddings], dim=1)
            output, _ = self.embedder(total_embeddings)
            output = th.cat([output[:, 0:1, :].repeat(1, output.shape[1], 1),
                             th.cat([self.obs_embed(status_emb.view(n_batch, -1)).view(n_batch, 1, -1), output[:, 1:, :]], dim=1)], dim=2)
            output = self.output_embed(output)
            return output[:, 1:, :], output[:, 0, :]
        elif self.dim_reduction == 'average':
            output, _ = self.embedder(encoded_status_embeddings)
            return output.mean(1)
        elif self.dim_reduction == 'multilayer_perceptron':
            output, _ = self.embedder(encoded_status_embeddings)
            # single_query_embeddings = self.single_query_embed(output)
            global_embedding = self.global_embed(output.view(n_batch, -1))
            return global_embedding


class AttentionExtractor2(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256,
                 embedding_path: str = 'nets/embeddings.npy', hidden_dim: int = 512,
                 n_heads: int = 8, n_layers: int = 1, ffn_dim: int = 128, dropout: float = 0.1,
                 att_dropout_rate: float = 0.1):
        super().__init__(observation_space, hidden_dim)

        query_num = observation_space.shape[0]
        # Load query embeddings
        embeddings = list(np.load(embedding_path))
        self.query_embeddings = th.FloatTensor(np.array(embeddings * math.ceil(query_num/len(embeddings)))[:query_num])[None, :, :].to('cuda')
        node_dim = self.query_embeddings.shape[2]
        self.init_embed = nn.Linear(node_dim, hidden_dim)
        # Initialize encoders
        self.status_encoder = nn.Embedding(observation_space[0].n, embedding_dim=hidden_dim)
        self.super_node_encoder = nn.Embedding(1, embedding_dim=hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, att_dropout_rate, n_heads) for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        n_batch = observations.shape[0]
        query_embeddings = F.leaky_relu(self.init_embed(self.query_embeddings.repeat(n_batch, 1, 1)))
        super_node_embedding = self.super_node_encoder.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        status_embeddings = self.status_encoder(observations.to(dtype=th.int64))
        output = self.input_dropout(th.cat([super_node_embedding, query_embeddings + status_embeddings], dim=1))
        for enc_layer in self.layers:
            output = enc_layer(output)
        output = self.final_ln(output)
        return output[:, 0, :]