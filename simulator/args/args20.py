class Args:
    # host = '162.105.86.20'
    host = '162.105.86.21'
    # host = '120.46.160.116'
    # host = '124.70.34.135'
    # host = '1.94.233.201'
    # batch_size = 32
    batch_size = 2048
    cls_loss_src = 'en'
    cuda_visible_devices = "1"
    database = 'tpcds1X'
    datapath_postfix = 'lll'
    # datapath_postfix = 'll'
    # datapath_postfix = 'l'
    # datapath_postfix = 'm'
    # datapath_postfix = 's'
    # datapath_postfix = 'ss'
    # datapath_postfix = ''
    device = 'cuda'
    embedding_path = 'embeddings.npy'
    enable_bn = False
    enable_cluster_embedding = True
    enable_cn = False
    enable_qf = True
    enable_worker = True
    enable_dop = enable_worker and (host == '120.46.160.116' \
                                    or host == '124.70.34.135'\
                                        or host == '1.94.233.201')
    epoch = 150
    info_agg = enable_qf and True
    loss_num = 2
    max_worker = 20
    # max_worker = 26
    # max_worker = 5
    # max_worker = 58
    # max_worker = 8
    # max_worker = 12
    # max_worker = 14
    # model_name = "cqf"
    model_name = "cqfia_ecl_do_none_gelu_no_end"
    query_num = 99 if database.startswith('tpcds') else 22
    query_scale = 1
    reduction = 'mean'
    # reduction_en = 'mean'
    # reduction_rank = 'mean'
    # reduction_reg = 'mean'
    reg_loss_scale = 0.1
    reg_loss_type = 'mse'
    seed = 0
    time_scale = 1
    train_log_freq = 2

class CQFormerArgs:
    attention_dropout_rate = 0.2
    dropout = 0.1
    embed_size = 64
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    pred_hid = 128
    mid_layers = False

class SQFormerArgs:
    attention_dropout_rate = 0.1
    bin_number = 50
    dropout = 0.1
    emb_size = 32
    ffn_dim = 32
    head_size = 8
    n_layers = 4
    pred_hid = 256
    use_hist = True
    use_sample = True

args = Args()
args_cqf = CQFormerArgs()
args_sqf = SQFormerArgs()
