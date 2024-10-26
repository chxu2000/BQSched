import os, sys, gym, json, numpy as np, random, argparse, importlib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from configparser import ConfigParser
from envs.query_scheduling import QuerySchedulingEnv

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from nets.features_extractor import FlattenExtractor, LinearExtractor, AttentionExtractor, AttentionExtractor2
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.utils import get_latest_run_id

from algorithms.recurrent_maskable.common.policies import RecurrentMaskableActorCriticPolicy, RecurrentMaskableMultiInputActorCriticPolicy
from algorithms.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO

from algorithms.ppg.ppg_mask import MaskablePPG
from algorithms.ppg.policies import MaskableAuxActorCriticPolicy, MaskMultiInputAuxActorCriticPolicy

from nets.policies import MaskableAttentionActorCriticPolicy, MaskableActorCriticPolicyMHA, MaskableActorCriticPolicySN, MaskableMultiInputActorCriticPolicySN, MaskableAuxActorCriticPolicySN, MaskableMultiInputAuxActorCriticPolicySN
from nets.policies import HierarchicalMaskableActorCriticPolicy

from algorithms.pretrain_maskable.ppo_mask_pretrain import PretrainMaskablePPO

from algorithms.a2c_maskable.a2c_mask import MaskableA2C


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #也可以设置成':16:8'
    # torch.use_deterministic_algorithms(True, warn_only=True)


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


sys.path.insert(0, './nets')
sys.path.insert(0, './algorithms')
sys.path.insert(0, './simulator')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train_from_scratch', choices=[
    'train_from_scratch',
    'train_on_clusters',
    'pre_train',
    'fine_tune',
])
pargs = parser.parse_args()

# Get args
args = importlib.import_module(f'modes.args.{pargs.mode}').args
seed_everything(args.seed)
# Get config
conf = ConfigParser()
conf.read(f'modes/config/{pargs.mode}.ini', encoding='utf-8')
query_scale, host, database = int(conf['database']['query_scale']), conf['database']['host'], conf['database']['database']
embedding_path = f"envs/scheduler/cache/{host}/{database}/embeddings.npy"
# Create log files
if not os.path.exists('outs/'):
    os.mkdir('outs/')
# sys.stdout = sys.stderr = open(f'outs/{host[host.rfind(".")+1:]}{".sim" if conf.getboolean("database", "use_simulator") else ""}.out', 'w')
# runtime_log_file = open(f'outs/{host[host.rfind(".")+1:]}.runtime.out', 'w') if not conf.getboolean('database', 'use_simulator') else None
runtime_log_file = open(f'outs/{host[host.rfind(".")+1:]}{".sim" if conf.getboolean("database", "use_simulator") else ""}.runtime.out', 'w')
# Print args and conf
print(json.dumps({attr: args.__getattribute__(attr) for attr in filter(lambda x:not x.startswith('__'), dir(args))}, indent=4), flush=True)
print(json.dumps({section: dict(conf[section]) for section in conf.sections()}, indent=4), flush=True)

# Initialize env
env = QuerySchedulingEnv(reward_type=args.train_reward_type, args=args, conf=conf, embedding_path=embedding_path, runtime_log=runtime_log_file)
eval_env = QuerySchedulingEnv(reward_type=args.eval_reward_type, args=args, conf=conf, embedding_path=embedding_path, runtime_log=runtime_log_file)
# check_env(env=env)

# Wrap to enable masking
env = ActionMasker(env, mask_fn)
eval_env = ActionMasker(eval_env, mask_fn)

# MaskablePPO
policy_kwargs = dict()
if 'att' in args.model_type:
    policy_kwargs['features_extractor_class'] = AttentionExtractor
    # policy_kwargs['preprocess_obs'] = False
    policy_kwargs['features_extractor_kwargs'] = dict(train_qf_data=torch.load(f'envs/scheduler/cache/{host}/{database}/train_qf_data.pt'))
    if 'qfsn' in args.model_type:
        policy_kwargs['features_extractor_kwargs']['costsw'] = json.load(open(f'envs/scheduler/cache/{host}/{database}/costsw.json'))
        policy_kwargs['features_extractor_kwargs']['enable_tl'] = args.enable_time_last
        policy_kwargs['features_extractor_kwargs']['reduced_features_dim'] = 256
        policy_kwargs['features_extractor_kwargs']['dim_reduction'] = 'super_node'
        policy_kwargs['features_extractor_kwargs']['pe_dim'] = 32
        policy_kwargs['features_extractor_kwargs']['share_features'] = True
    elif 'sn' in args.model_type:
        policy_kwargs['emb_agg'] = args.emb_agg
        policy_kwargs['soft_masking'] = args.enable_soft_masking
        policy_kwargs['features_extractor_kwargs']['enable_tl'] = args.enable_time_last
        policy_kwargs['features_extractor_kwargs']['mask_done'] = args.enable_mask_done
        policy_kwargs['features_extractor_kwargs']['feature_align'] = args.enable_feature_align
        policy_kwargs['features_extractor_kwargs']['reduced_features_dim'] = 128
        policy_kwargs['features_extractor_kwargs']['dim_reduction'] = 'super_node'
        policy_kwargs['features_extractor_kwargs']['share_features'] = False
        if 'cw' in args.model_type:
            policy_kwargs['features_extractor_kwargs']['costsw'] = json.load(open(f'envs/scheduler/cache/{host}/{database}/costsw.json'))
        if 'pe' in args.model_type:
            policy_kwargs['features_extractor_kwargs']['pe_dim'] = 32
        if 'cqe' in args.model_type:
            policy_kwargs['features_extractor_kwargs']['reduced_features_dim'] = 64
            policy_kwargs['features_extractor_kwargs']['con_query_num'] = int(conf['scheduler']['max_worker'])
        elif 'cq' in args.model_type:
            policy_kwargs['features_extractor_kwargs']['con_query_num'] = int(conf['scheduler']['max_worker'])
    if conf.getboolean('database', 'overlap_cluster'):
        cluster_result = np.load(conf['database']['cluster_result_path'])['arr_0']
        policy_kwargs['cluster_num'] = int(max(cluster_result) + 1)
        policy_kwargs['features_extractor_kwargs']['cluster_result'] = cluster_result
    # elif 'sncwpecq' in args.model_type:
    #     policy_kwargs['features_extractor_kwargs'] = dict(train_qf_data=torch.load(f'envs/scheduler/cache/{host}/{database}/train_qf_data.pt'),
    #                                                       costsw=json.load(open(f'envs/scheduler/cache/{host}/{database}/costsw.json')),
    #                                                       enable_tl=args.enable_time_last, reduced_features_dim=128, dim_reduction='super_node',
    #                                                       pe_dim=32, share_features=False, con_query_num=int(conf['scheduler']['max_worker']),)
    # elif 'sncwpe' in args.model_type:
    #     policy_kwargs['features_extractor_kwargs'] = dict(train_qf_data=torch.load(f'envs/scheduler/cache/{host}/{database}/train_qf_data.pt'),
    #                                                       costsw=json.load(open(f'envs/scheduler/cache/{host}/{database}/costsw.json')),
    #                                                       enable_tl=args.enable_time_last, reduced_features_dim=128, dim_reduction='super_node',
    #                                                       pe_dim=32, share_features=False)
    # elif 'sncw' in args.model_type:
    #     policy_kwargs['features_extractor_kwargs'] = dict(train_qf_data=torch.load(f'envs/scheduler/cache/{host}/{database}/train_qf_data.pt'),
    #                                                       costsw=json.load(open(f'envs/scheduler/cache/{host}/{database}/costsw.json')),
    #                                                       enable_tl=args.enable_time_last, reduced_features_dim=128, dim_reduction='super_node',
    #                                                       share_features=False)
    # elif 'sn' in args.model_type:
    #     policy_kwargs['features_extractor_kwargs'] = dict(train_qf_data=torch.load(f'envs/scheduler/cache/{host}/{database}/train_qf_data.pt'),
    #                                                       reduced_features_dim=128, dim_reduction='super_node', share_features=False)
    else:
        # policy_kwargs['features_extractor_kwargs'] = dict(embedding_path=embedding_path)
        policy_kwargs['features_extractor_kwargs']['embedding_path'] = embedding_path
if 'hmppo' in args.model_type:
    policy_kwargs['hierarchical_mode'] = args.hierarchical_mode
    policy_kwargs['layer_shapes'] = [query_scale, args.query_num]
    if args.enable_worker and args.action_flatten:
        policy_kwargs['layer_shapes'].append(args.max_worker)
n_steps = args.runs_per_train * (args.query_num * query_scale if env.env.query_cluster else env.env.action_num)
enable_multi_input = args.enable_embedding or args.enable_resource or args.enable_time_last
if args.model_type == 'mppo':
    model = MaskablePPO(policy=MaskableMultiInputActorCriticPolicy if enable_multi_input else MaskableActorCriticPolicy, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'mppo_att' or args.model_type == 'mppo_attqf' or args.model_type == 'mppo_attqfsn':
    model = MaskablePPO(policy=MaskableActorCriticPolicy, policy_kwargs=policy_kwargs, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'mppo_mha':
    model = MaskablePPO(policy=MaskableActorCriticPolicyMHA, policy_kwargs=policy_kwargs, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif 'mppo_attsn' in args.model_type:
    model = MaskablePPO(policy=MaskableMultiInputActorCriticPolicySN if enable_multi_input else MaskableActorCriticPolicySN, policy_kwargs=policy_kwargs, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'mppo_cont':
    model = MaskablePPO.load(args.pretrain_path, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
    model.set_env(env)
elif args.model_type == 'rmppo':
    model = RecurrentMaskablePPO(policy=RecurrentMaskableMultiInputActorCriticPolicy if enable_multi_input else RecurrentMaskableActorCriticPolicy, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'rmppo_att':
    model = RecurrentMaskablePPO(policy=RecurrentMaskableActorCriticPolicy, policy_kwargs=policy_kwargs, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'hmppo':
    model = MaskablePPO(policy=HierarchicalMaskableActorCriticPolicy, policy_kwargs=policy_kwargs, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'mppg':
    model = MaskablePPG(policy=MaskMultiInputAuxActorCriticPolicy if enable_multi_input else MaskableAuxActorCriticPolicy, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'mppg_att' or args.model_type == 'mppg_attqf':
    model = MaskablePPG(policy=MaskMultiInputAuxActorCriticPolicy if enable_multi_input else MaskableAuxActorCriticPolicy, policy_kwargs=policy_kwargs, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'mppg_mha':
    model = MaskablePPG(policy=MaskableActorCriticPolicyMHA, policy_kwargs=policy_kwargs, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif 'mppg_attsn' in args.model_type:
    model = MaskablePPG(policy=MaskableMultiInputAuxActorCriticPolicySN if enable_multi_input else MaskableAuxActorCriticPolicySN, policy_kwargs=policy_kwargs, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path,
                        aux_value_phase=args.enable_aux_value, aux_time_phase=args.enable_aux_time, aux_learning_rate=args.aux_learning_rate, n_policy_iters=args.n_policy_iters, n_aux_epochs=args.n_aux_epochs, beta_clone=args.beta_clone,
                        vf_true_coef=args.vf_true_coef, aux_time_learning_rate=args.aux_time_learning_rate, beta_clone_time=args.beta_clone_time, aux_value_time=args.aux_value_time, aux_time_ind=args.aux_time_ind,
                        aux_vf_distance=args.aux_vf_distance)
elif args.model_type == 'mppg_cont':
    model = MaskablePPG.load(args.pretrain_path, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
    model.set_env(env)
elif args.model_type == 'pmppo':
    model = PretrainMaskablePPO(policy=MaskableMultiInputActorCriticPolicy if enable_multi_input else MaskableActorCriticPolicy, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'ma2c':
    model = MaskableA2C(policy=MaskableMultiInputActorCriticPolicy if enable_multi_input else MaskableActorCriticPolicy, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'ma2c_mha':
    model = MaskableA2C(policy=MaskableActorCriticPolicyMHA, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif 'ma2c_attsn' in args.model_type:
    model = MaskableA2C(policy=MaskableMultiInputActorCriticPolicySN if enable_multi_input else MaskableActorCriticPolicySN, policy_kwargs=policy_kwargs, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
elif args.model_type == 'ma2c_cont':
    model = MaskableA2C.load(args.pretrain_path, env=env, n_steps=n_steps, verbose=1, tensorboard_log=args.log_path)
    model.set_env(env)
else:
    raise Exception('Model type not supported')
print(model.policy)

if args.pretrain_offline:
    model.pretrain(args, conf)

# Train
iter, best_reward, reset_num_timesteps, i = 0, -1e4, args.reset_num_timesteps, get_latest_run_id(args.log_path, args.run_name)+1
# while os.path.exists(f'outputs/{args.run_name}_{i}'):
#     i += 1
if not reset_num_timesteps:
    i -= 1
eval_freq = args.runs_per_eval * (args.query_num * query_scale if env.env.query_cluster else env.env.action_num)
eval_callback = MaskableEvalCallback(eval_env, n_eval_episodes=args.train_eval_episodes, eval_freq=eval_freq, log_path=os.path.join(args.log_path, f'{args.run_name}_{i}/evals'),
                                     best_model_save_path=os.path.join(args.log_path, f'{args.run_name}_{i}/outputs'))
while True:
    # iter += 1
    model.learn(total_timesteps=args.runs_per_logfile*args.query_num*query_scale, tb_log_name=args.run_name, reset_num_timesteps=reset_num_timesteps, callback=eval_callback)
    # model.save(f'outputs/{run_name}_{i}/iter-last')
    # mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
    # print(f'Eval mean reward {mean_reward} (iter {iter})')
    # if mean_reward > best_reward:
    #     print(f'New best eval mean reward {mean_reward} (iter {iter}), saving best model...')
    #     model.save(f'outputs/{run_name}_{i}/iter-{iter}')
    #     best_reward = mean_reward
    reset_num_timesteps = False