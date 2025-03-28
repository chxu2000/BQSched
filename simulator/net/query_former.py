import torch, torch.nn as nn
from simulator.net.transformer import *
from simulator.net.graph_encoder import GraphAttentionEncoderWithMask


class QueryFormer(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 2, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256
                ):
        
        super(QueryFormer,self).__init__()
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size //8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size //8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        # self.pred = Prediction(hidden_dim, pred_hid)

        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)
        
    # def forward(self, batched_data):
    #     attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x

    #     heights = batched_data.heights     
        
    def forward(self, attn_bias, rel_pos, x, heights):

        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) 
        
        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias


        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        x_view = x.view(-1, 1165)
        node_feature = self.embbed_layer(x_view).view(n_batch,-1, self.hidden_dim)
        
        # -1 is number of dummy
        
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)        
        
        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)
        
        # return self.pred(output[:,0,:]), self.pred2(output[:,0,:]), output[:,0,:]
        return output[:,0,:]


class SharedQueryFormer(nn.Module):
    def __init__(self, hidden_dim=128, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.2, n_layers = 2, \
                 use_sample = True, use_hist = True, bin_number = 50, pred_hid = 256, \
                 query_num = 99, enable_qf = False, train_qf_data=None, con_query_num=20,
                ):
        
        super(SharedQueryFormer, self).__init__()

        self.enable_qf = enable_qf
        
        # self.linear_time = nn.Linear(1, hidden_dim)

        if self.enable_qf:
            self.train_qf_data = train_qf_data
            self.train_qf_data.attn_bias = add_zeros(self.train_qf_data.attn_bias).to('cuda')
            self.train_qf_data.rel_pos = add_zeros(self.train_qf_data.rel_pos).to('cuda')
            self.train_qf_data.x = add_zeros(self.train_qf_data.x).to('cuda')
            self.train_qf_data.heights = add_zeros(self.train_qf_data.heights).to('cuda')
            self.query_former = QueryFormer(emb_size=emb_size, ffn_dim=ffn_dim, head_size=head_size, dropout=dropout,
                                            attention_dropout_rate=attention_dropout_rate, n_layers=n_layers, use_sample=use_sample,
                                            use_hist=use_hist, bin_number=bin_number, pred_hid=pred_hid)
            self.linear_in = nn.Linear(query_num + self.query_former.hidden_dim, hidden_dim-4)
        else:
            self.input_dropout = nn.Dropout(dropout)
            self.linear_in = nn.Linear(query_num, hidden_dim-4)

        self.super_node_encoder = nn.Embedding(1, embedding_dim=hidden_dim)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        #self.final_ln = nn.LayerNorm(hidden_dim*2)
        init_dim = hidden_dim * 2 + con_query_num * 3
        self.final_ln = nn.Sequential(
            nn.Linear(init_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )
        
        # self.super_token = nn.Embedding(1, hidden_dim)
        # self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        # self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        self.pred = Prediction(hidden_dim, pred_hid)
        self.linear_to_time = nn.Linear(hidden_dim, 1)

        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)
        


    def forward(self, qids, embeddings, times, workers, base_cost):

        # time_embedding = self.linear_time(times)
        # time_embedding = times.repeat(1,1, con_queries.shape[2])
        # transfomrer encoder
        if self.enable_qf:
            qids = qids.to(self.train_qf_data.attn_bias.device)
            attn_bias, rel_pos, x, heights = self.train_qf_data.attn_bias[qids], \
                self.train_qf_data.rel_pos[qids], self.train_qf_data.x[qids], self.train_qf_data.heights[qids]
            attn_bias, rel_pos, x, heights = attn_bias.to(times.device), rel_pos.to(times.device), x.to(times.device), heights.to(times.device)
            embeddings_qf = self.query_former(attn_bias.view(-1, attn_bias.shape[-2], attn_bias.shape[-1]), rel_pos.view(-1, rel_pos.shape[-2], rel_pos.shape[-1]),
                                              x.view(-1, x.shape[-2], x.shape[-1]), heights.view(-1, heights.shape[-1]))
            embeddings_qf = embeddings_qf.view(rel_pos.shape[0], rel_pos.shape[1], -1)
            embeddings = self.linear_in(torch.cat([embeddings, embeddings_qf], dim=-1))
        else:
            embeddings = self.input_dropout(embeddings)
            embeddings = self.linear_in(embeddings)

        x = torch.cat([embeddings, times, workers, base_cost, base_cost - times], dim=2)
        #x = torch.cat([con_queries, times, workers,], dim=2)

        n_batch, n_query = x.shape[0], x.shape[1]
        super_node_embedding = self.super_node_encoder.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        output = torch.cat([super_node_embedding, x], dim=1)
        for enc_layer in self.layers:
            output = enc_layer(output)

        # minvalue, _ = torch.min(output, dim=1)
        # minvalue = minvalue.unsqueeze(dim=1)
        # minvalue = minvalue.repeat(1, output.shape[1], 1)

        # output = torch.cat([x - output, minvalue, x], dim=2)

        output = torch.cat([output[:, 1:, :], output[:, 0:1, :].repeat(1, n_query, 1),
                            times.view(n_batch, 1, -1).repeat(1, n_query, 1), workers.view(n_batch, 1, -1).repeat(1, n_query, 1),
                            base_cost.view(n_batch, 1, -1).repeat(1, n_query, 1)], dim=2)
        output = self.final_ln(output)

        # minvalue,_ = torch.min(output,dim=1)
        # minvalue = minvalue.unsqueeze(dim=1)
        # minvalue = torch.repeat(1, output.shape[1],1)

        #output = torch.cat([output, minvalue, x], dim=2)
        #output = torch.cat([output, x], dim=2)

        output_time = self.linear_to_time(output).squeeze()
        output_index = self.pred(output).squeeze()

        #output_time = self.linear_to_time(output).squeeze()
        #output_index = self.pred(output).squeeze()
        # return self.pred(output).squeeze()
        return output_time, output_index

        # return self.pred(output).squeeze()


class ConcurrentQueryFormer(nn.Module):
    def __init__(self, hidden_dim=128, ffn_dim = 32, head_size = 8, dropout = 0.1,
                 attention_dropout_rate = 0.2, n_layers = 2, pred_hid = 256,
                 query_emb_dim = 99, enable_qf = False, train_qf_data=None,
                 info_agg = False, enable_cn = False, args = None, args_cqf = None,
                 args_sqf = None,
                ):
        
        super(ConcurrentQueryFormer,self).__init__()

        self.enable_qf, self.info_agg, self.head_size, self.enable_cn, self.args = \
            enable_qf, info_agg, head_size, enable_cn, args
        self.enable_bn = hasattr(self.args, "enable_bn") and self.args.enable_bn
        
        # self.linear_time = nn.Linear(1, hidden_dim)

        raw_feature_num = 5 if enable_cn else 4
        if self.enable_bn:
            self.batch_norm = nn.BatchNorm1d(num_features=raw_feature_num)

        pre_hidden_dim = hidden_dim - raw_feature_num
        if self.enable_qf:
            self.train_qf_data = train_qf_data
            self.train_qf_data.attn_bias = self.train_qf_data.attn_bias.to("cuda")
            self.train_qf_data.rel_pos = self.train_qf_data.rel_pos.to("cuda")
            self.train_qf_data.x = self.train_qf_data.x.to("cuda")
            self.train_qf_data.heights = self.train_qf_data.heights.to("cuda")
            self.query_former = QueryFormer(emb_size=args_sqf.emb_size, ffn_dim=args_sqf.ffn_dim, head_size=args_sqf.head_size, dropout=args_sqf.dropout,
                                            attention_dropout_rate=args_sqf.attention_dropout_rate, n_layers=args_sqf.n_layers, use_sample=args_sqf.use_sample,
                                            use_hist=args_sqf.use_hist, bin_number=args_sqf.bin_number, pred_hid=args_sqf.pred_hid)
            input_dim = self.query_former.hidden_dim
            # if self.args.enable_cluster_embedding and self.args.query_scale > 1:
            #     input_dim += self.args.query_scale
            self.input_dropout = nn.Dropout(dropout)
            self.linear_in = nn.Linear(input_dim, pre_hidden_dim)
            # self.linear_in = nn.Sequential(
            #     nn.Linear(input_dim, pre_hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(pre_hidden_dim, pre_hidden_dim),
            #     nn.GELU()
            # )
        else:
            self.input_dropout = nn.Dropout(dropout)
            self.linear_in = nn.Linear(query_emb_dim, pre_hidden_dim)

        if self.info_agg:
            self.super_node_encoder_con = nn.Embedding(1, embedding_dim=hidden_dim)

            # # gat
            # self.embedder = GraphAttentionEncoderWithMask(
            #     n_heads=head_size,
            #     embed_dim=hidden_dim,
            #     n_layers=n_layers
            # )

            # ecl
            encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                        for _ in range(n_layers)]
            self.layers = nn.ModuleList(encoders)

            # # torch
            # encoders = [nn.TransformerEncoderLayer(
            #     d_model=hidden_dim,
            #     nhead=head_size,
            #     dim_feedforward=ffn_dim,
            #     dropout=dropout,
            #     activation="gelu",
            #     batch_first=True,
            #     norm_first=True
            # ) for _ in range(n_layers)]
            # self.layers = nn.ModuleList(encoders)

            # self.linear_out = nn.Linear(hidden_dim*4, hidden_dim)
            self.linear_out = nn.Sequential(
                nn.Linear(hidden_dim*4, hidden_dim*2),
                nn.GELU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                # nn.GELU(),
            )
            # self.output_dropout = nn.Dropout(dropout)
        else:
            encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                        for _ in range(n_layers)]
            self.layers = nn.ModuleList(encoders)
            
            # # gat
            # self.embedder = GraphAttentionEncoderWithMask(
            #     n_heads=head_size,
            #     embed_dim=hidden_dim,
            #     n_layers=n_layers
            # )
        
            # self.final_ln = nn.LayerNorm(hidden_dim*2)
            self.linear_out = nn.Linear(hidden_dim*3, hidden_dim)
    
        # self.super_token = nn.Embedding(1, hidden_dim)
        # self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        # self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        self.linear_to_index = Prediction(hidden_dim, pred_hid, mid_layers=args_cqf.mid_layers)
        self.linear_to_time = nn.Linear(hidden_dim, 1)

        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)
        


    def forward(self, qids, embeddings, times, workers, base_cost):

        # time_embedding = self.linear_time(times)
        # time_embedding = times.repeat(1,1, con_queries.shape[2])
        # transfomrer encoder
        if self.enable_qf:
            # qids_mod = (qids - 1) % 99 + 1
            # qids_mod[qids==0] = 0

            embeddings_qf = self.query_former(self.train_qf_data.attn_bias, self.train_qf_data.rel_pos, self.train_qf_data.x, self.train_qf_data.heights)
            embeddings_qf = add_zeros(embeddings_qf)[qids]
            # if self.args.enable_cluster_embedding and self.args.query_scale > 1:
            #     embeddings_qf = torch.cat([embeddings_qf, embeddings[:, :, -self.args.query_scale:]], dim=2)
            embeddings = embeddings_qf
            embeddings = self.input_dropout(embeddings)
            embeddings = self.linear_in(embeddings)
        else:
            embeddings = self.input_dropout(embeddings)
            embeddings = self.linear_in(embeddings)

        if self.enable_cn:
            conn_num = (qids>0).sum(dim=-1).view(-1, 1, 1).repeat(1, times.shape[1], 1)
            raw_features = torch.cat([times, workers, base_cost, base_cost - times, conn_num], dim=2)
            # x = torch.cat([embeddings, times, workers, base_cost, base_cost - times, conn_num], dim=2)
        else:
            raw_features = torch.cat([times, workers, base_cost, base_cost - times], dim=2)
            # x = torch.cat([embeddings, times, workers, base_cost, base_cost - times], dim=2)
        if self.enable_bn:
            raw_features = self.batch_norm(raw_features.view(-1, raw_features.shape[-1])).view(raw_features.shape)
        x = torch.cat([embeddings, raw_features], dim=2)
        #x = torch.cat([con_queries, times, workers,], dim=2)

        if self.info_agg:
            n_batch = x.shape[0]
            super_node_embedding_con = self.super_node_encoder_con.weight.unsqueeze(0).repeat(n_batch, 1, 1)
            x = torch.cat([super_node_embedding_con, x], dim=1)
            output = x

            # indices_con = torch.where(qids > 0)
            # mask = torch.ones((n_batch, output.shape[1], output.shape[1]), dtype=torch.bool, device='cuda')
            # mask[:, 0, 0] = 0       # sn_con2sn_con
            # for batch in range(n_batch):
            #     batch_indices_con = indices_con[1][indices_con[0]==batch] + 1
            #     mask[batch, 0, batch_indices_con] = 0   # sn_con2con
            #     mask[batch, batch_indices_con.unsqueeze(0), batch_indices_con.unsqueeze(1)] = 0     # con2con

            # # gat
            # output = self.embedder(output, mask=mask)
            # output = self.embedder(output)
            # if type(output) != torch.Tensor: output = output[0]

            # ecl & torch
            # mask = mask.unsqueeze(1).repeat(1, self.head_size, 1, 1)
            # mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
            for enc_layer in self.layers:
                output = enc_layer(output)
                # output = enc_layer(output, mask=mask)
        else:
            output = x
            for enc_layer in self.layers:
                output = enc_layer(output)
            
            # # gat
            # output = self.embedder(output)
            # if type(output) != torch.Tensor: output = output[0]

        minvalue, _ = torch.min(output, dim=1)
        minvalue = minvalue.unsqueeze(dim=1)
        minvalue = minvalue.repeat(1, output.shape[1], 1)

        if self.info_agg:
            output = torch.cat([x - output, minvalue, x, output[:, 0, :].unsqueeze(1).repeat(1, output.shape[1], 1)], dim=2)[:, 1:, :]
        else:
            output = torch.cat([x - output, minvalue, x], dim=2)
        output = self.linear_out(output)
        # if self.info_agg:
        #     output = self.output_dropout(output)

        # minvalue,_ = torch.min(output,dim=1)
        # minvalue = minvalue.unsqueeze(dim=1)
        # minvalue = torch.repeat(1, output.shape[1],1)

        #output = torch.cat([output, minvalue, x], dim=2)
        #output = torch.cat([output, x], dim=2)

        output_time = self.linear_to_time(output).squeeze()
        output_index = self.linear_to_index(output).squeeze()

        #output_time = self.linear_to_time(output).squeeze()
        #output_index = self.pred(output).squeeze()
        # return self.pred(output).squeeze()
        # output_index[qids==0] = -np.inf
        output_index = output_index.masked_fill((qids==0).bool(), -np.inf)
        return output_time, output_index

        # return self.pred(output).squeeze()

    def forward1(self, con_queries, times):

        time_embedding = self.linear_time(times)
        # transfomrer encoder
        output = self.input_dropout(con_queries + time_embedding)
        output = self.linear_in(output)

        for enc_layer in self.layers:
            output = enc_layer(output)
        output = self.linear_out(output)

        return self.linear_to_index(output).squeeze()

    def forward2(self, con_queries, times, base_cost):

        # time_embedding = self.linear_time(times)
        # time_embedding = times.repeat(1,1, con_queries.shape[2])
        x = torch.cat([con_queries, times, base_cost, base_cost - times], dim=2)
        # transfomrer encoder
        x = self.input_dropout(x)
        x = self.linear_in(x)

        output = x
        for enc_layer in self.layers:
            output = enc_layer(output)

        minvalue, _ = torch.min(output, dim=1)
        minvalue = minvalue.unsqueeze(dim=1)
        minvalue = minvalue.repeat(1, output.shape[1], 1)

        output = torch.cat([output, x, minvalue], dim=2)
        output = self.linear_out(output)

        # minvalue,_ = torch.min(output,dim=1)
        # minvalue = minvalue.unsqueeze(dim=1)
        # minvalue = torch.repeat(1, output.shape[1],1)

        # output = torch.cat([output, minvalue, x], dim=2)

        output_time = self.linear_to_time(output).squeeze()
        output_index = self.linear_to_index(output).squeeze()
        # return self.pred(output).squeeze()
        return output_time, output_index

        # return self.pred(output).squeeze()


class ConcurrentQueryFormerReg(nn.Module):
    def __init__(self, hidden_dim=128, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.2, n_layers = 2, \
                 use_sample = True, use_hist = True, bin_number = 50, pred_hid = 256, \
                 query_num = 99, enable_qf = False, train_qf_data=None,
                ):
        
        super(ConcurrentQueryFormerReg,self).__init__()

        self.enable_qf = enable_qf
        
        # self.linear_time = nn.Linear(1, hidden_dim)

        if self.enable_qf:
            self.train_qf_data = train_qf_data
            self.train_qf_data.attn_bias = add_zeros(self.train_qf_data.attn_bias)
            self.train_qf_data.rel_pos = add_zeros(self.train_qf_data.rel_pos)
            self.train_qf_data.x = add_zeros(self.train_qf_data.x)
            self.train_qf_data.heights = add_zeros(self.train_qf_data.heights)
            self.query_former = QueryFormer(emb_size=emb_size, ffn_dim=ffn_dim, head_size=head_size, dropout=dropout,
                                            attention_dropout_rate=attention_dropout_rate, n_layers=n_layers, use_sample=use_sample,
                                            use_hist=use_hist, bin_number=bin_number, pred_hid=pred_hid)
            self.linear_in = nn.Linear(query_num + self.query_former.hidden_dim, hidden_dim-4)
        else:
            self.input_dropout = nn.Dropout(dropout)
            self.linear_in = nn.Linear(query_num, hidden_dim-4)

        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        #self.final_ln = nn.LayerNorm(hidden_dim*2)
        self.final_ln = nn.Linear(hidden_dim*3, hidden_dim)
        
        # self.super_token = nn.Embedding(1, hidden_dim)
        # self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        # self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        # self.pred = Prediction(hidden_dim, pred_hid)
        # self.linear_to_time = nn.Linear(20*hidden_dim, 1)
        self.linear_to_time = nn.Linear(hidden_dim, 1)

        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)
        


    def forward(self, qids, embeddings, times, workers, base_cost):

        # time_embedding = self.linear_time(times)
        # time_embedding = times.repeat(1,1, con_queries.shape[2])
        # transfomrer encoder
        if self.enable_qf:
            # qids = qids.to(self.train_qf_data.attn_bias.device)
            attn_bias, rel_pos, x, heights = self.train_qf_data.attn_bias[qids], \
                self.train_qf_data.rel_pos[qids], self.train_qf_data.x[qids], self.train_qf_data.heights[qids]
            # attn_bias, rel_pos, x, heights = attn_bias.to(times.device), rel_pos.to(times.device), x.to(times.device), heights.to(times.device)
            embeddings_qf = self.query_former(attn_bias.view(-1, attn_bias.shape[-2], attn_bias.shape[-1]), rel_pos.view(-1, rel_pos.shape[-2], rel_pos.shape[-1]),
                                              x.view(-1, x.shape[-2], x.shape[-1]), heights.view(-1, heights.shape[-1]))
            embeddings_qf = embeddings_qf.view(rel_pos.shape[0], rel_pos.shape[1], -1)
            embeddings = self.linear_in(torch.cat([embeddings, embeddings_qf], dim=-1))
        else:
            embeddings = self.input_dropout(embeddings)
            embeddings = self.linear_in(embeddings)

        x = torch.cat([embeddings, times, workers, base_cost, base_cost - times], dim=2)
        #x = torch.cat([con_queries, times, workers,], dim=2)

        output = x
        for enc_layer in self.layers:
            output = enc_layer(output)

        minvalue, _ = torch.min(output, dim=1)
        minvalue = minvalue.unsqueeze(dim=1)
        minvalue = minvalue.repeat(1, output.shape[1], 1)

        output = torch.cat([x - output, minvalue, x], dim=2)
        output = self.final_ln(output)

        # minvalue,_ = torch.min(output,dim=1)
        # minvalue = minvalue.unsqueeze(dim=1)
        # minvalue = torch.repeat(1, output.shape[1],1)

        #output = torch.cat([output, minvalue, x], dim=2)
        #output = torch.cat([output, x], dim=2)

        # output_index = self.pred(output).squeeze()

        # output_time = self.linear_to_time(output.view((output.shape[0], -1))).squeeze()
        output_time = self.linear_to_time(output).squeeze()
        #output_index = self.pred(output).squeeze()
        # return self.pred(output).squeeze()
        return output_time

        # return self.pred(output).squeeze()


class MultiQueryFormer(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256, query_num = 8
                ):
        
        super(MultiQueryFormer,self).__init__()
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size //8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size //8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist
        self.query_num = query_num

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        
        self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        self.pred = Prediction(hidden_dim * query_num, pred_hid)

        # if multi-task
        self.pred2 = Prediction(hidden_dim * query_num, pred_hid)
        
    def forward(self, batched_data):
        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x

        heights = batched_data.heights     
        
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) 
        
        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias


        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        x_view = x.view(-1, 1165)
        node_feature = self.embbed_layer(x_view).view(n_batch,-1, self.hidden_dim)
        
        # -1 is number of dummy
        
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)        
        
        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        # aggregate multi-query embedding
        if output.shape[0] % self.query_num != 0:
            raise Exception('shape of batched_data inconsistent with query_num')
        output = output[:,0,:]
        output = torch.cat([torch.cat([output[i+j] for j in range(self.query_num)], dim=0).unsqueeze(0) for i in range(0, output.shape[0], self.query_num)], dim=0)


        
        return self.pred(output), self.pred2(output)
