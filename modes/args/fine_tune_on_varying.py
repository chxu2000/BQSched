class Args():
    devices = "0"
    query_num = 99
    init_baseline = 25
    enable_embedding = False
    enable_resource = False
    local_resource = False
    enable_time_last = True
    enable_mask_done = False
    enable_feature_align = False
    enable_soft_masking = False
    enable_worker = True
    max_worker = 2
    action_flatten = True
    train_reward_type = 'relative_time_with_baseline'
    eval_reward_type = 'cost'
    # model
    model_type = 'mppg_cont'
    seed = 0
    hierarchical_mode = 2
    # train
    runs_per_train = 20
    runs_per_eval = 40
    train_eval_episodes = 2
    runs_per_logfile = 2500
    reset_num_timesteps = True
    log_path = 'logs/'
    run_name = 'mtippg_attsncwcqia_rsrtbs_tpcds1Xhtmt_simcqf_wf_cont'
    # run_name = 'aaa'
    pretrain_path = 'outputs/mtippg_attsncwcqia_rsrtbs_tpcds1Xhtmt_simcqf_wf_1/best_model_13'
    hybrid_training = True
    multi_task = True
    task_adaptive = True
    value_adaptive = True
    entropy_adaptive = True
    sub_save_freq = 1
    # test
    test_eval_episodes = 2
    checkpoint_path = 'logs/mtippg_attsncwcqia_rsrtbs_tpcds1Xht_wf_1/outputs/best_model_34'
    # test all
    test_run_name = 'aaa_6'
    # pretrain
    expert_strategy = 'max_cost_first'
    expert_threshold = 18
    pretrain_algorithm = 'bc'
    pretrain_epochs = 3
    pretrain_steps = 4950
    pretrain_offline = False
    pretrain_total_steps = 99
    pretrain_enable_embedding = True
    # PPG
    enable_aux_value = False
    enable_aux_time = True
    aux_value_time = False
    aux_time_ind = True
    aux_vf_distance = False
    emb_agg = False
    aux_learning_rate = 3e-4
    n_policy_iters = 10
    n_aux_epochs = 2
    beta_clone = 6.0
    vf_true_coef = 1.0
    aux_time_learning_rate = 3e-4
    beta_clone_time = 2.0


args = Args()