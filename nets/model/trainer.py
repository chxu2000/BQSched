import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .dataset import PlanTreeDataset
from .database_util import collator, get_job_table_sample
import os
import time
import torch
from scipy.stats import pearsonr
from tqdm import tqdm
import json

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def print_multi_qerror(pos_preds_unnorm, pos_labels_unnorm, cost_preds_unnorm, cost_labels_unnorm, prints=False):
    qerror = []
    for i in range(len(pos_preds_unnorm)):
        if pos_preds_unnorm[i] > float(pos_labels_unnorm[i]):
            qerror.append(pos_preds_unnorm[i] / float(pos_labels_unnorm[i]))
        else:
            qerror.append(float(pos_labels_unnorm[i]) / float(pos_preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Pos Median: {}".format(e_50))
        print("Pos Mean: {}".format(e_mean))

    pos_res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_mean' : e_mean,
    }
    
    qerror = []
    for i in range(len(cost_preds_unnorm)):
        if cost_preds_unnorm[i] > float(cost_labels_unnorm[i]):
            qerror.append(cost_preds_unnorm[i] / float(cost_labels_unnorm[i]))
        else:
            qerror.append(float(cost_labels_unnorm[i]) / float(cost_preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Cost Median: {}".format(e_50))
        print("Cost Mean: {}".format(e_mean))

    cost_res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_mean' : e_mean,
    }

    return {'pos': pos_res, 'cost': cost_res}

def print_qerror(preds_unnorm, labels_unnorm, prints=False):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Median: {}".format(e_50))
        print("Mean: {}".format(e_mean))

    res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_mean' : e_mean,
    }

    return res

def get_corr(ps, ls): # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))
    
    return corr


def eval_workload(database, workload, methods):

    get_table_sample = methods['get_sample']

    workload_file_name = './data/{}/workloads/'.format(database) + workload
    table_sample = get_table_sample(workload_file_name)
    plan_df = pd.read_csv('./data/{}/{}_plan.csv'.format(database, workload))
    workload_csv = pd.read_csv('./data/{}/workloads/{}.csv'.format(database, workload),sep='#',header=None)
    workload_csv.columns = ['table','join','predicate','cardinality']
    ds = PlanTreeDataset(plan_df, workload_csv, \
        methods['encoding'], methods['hist_file'], methods['cost_norm'], \
        methods['cost_norm'], 'cost', table_sample)

    eval_score = evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'], True, database=database)
    return eval_score, ds


def evaluate(model, ds, bs, norm, device, prints=False, database=''):
    model.eval()
    cost_predss = np.empty(0)

    with torch.no_grad():
        embeddingss = []
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i,min(i+bs, len(ds)) ) ])))

            batch = batch.to(device)

            cost_preds, _, embeddings = model(batch)
            embeddingss.append(embeddings.cpu())
            cost_preds = cost_preds.squeeze()

            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())
    np.save(f'outs/{database}/embeddings', torch.cat(embeddingss).numpy())
    unnorm_cost_predss = norm.unnormalize_labels(cost_predss)
    scores = print_qerror(unnorm_cost_predss, ds.costs, prints)
    corr = get_corr(unnorm_cost_predss, ds.costs)
    if prints:
        print('Corr: ',corr)
    eval_list = []
    for i in range(len(unnorm_cost_predss)):
        eval_list.append([float(unnorm_cost_predss[i]), float(ds.costs[i])])
    with open(f'outs/{database}/eval.json', 'w') as f:
        json.dump(eval_list, f, indent=4)
    return scores, corr

def train(model, train_ds, val_ds, crit, \
    cost_norm, args, optimizer=None, scheduler=None):
    
    to_pred, bs, device, epochs, clip_size = \
        args.to_predict, args.bs, args.device, args.epochs, args.clip_size
    lr = args.lr

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)


    t0 = time.time()

    rng = np.random.default_rng()

    best_prev = 999999

    log_dict = {'epoch': [], 'avg loss': [], 'time': [], 'train scores': [], 'test scores': []}

    for epoch in range(epochs):
        losses = 0
        cost_predss = np.empty(0)

        model.train()

        train_idxs = rng.permutation(len(train_ds))

        cost_labelss = np.array(train_ds.costs)[train_idxs]


        for idxs in chunks(train_idxs, bs):
            optimizer.zero_grad()

            batch, batch_labels = collator(list(zip(*[train_ds[j] for j in idxs])))
            
            l, r = zip(*(batch_labels))

            batch_cost_label = torch.FloatTensor(l).to(device)
            batch = batch.to(device)

            cost_preds, _, embeddings = model(batch)
            cost_preds = cost_preds.squeeze()

            loss = crit(cost_preds, batch_cost_label)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)

            optimizer.step()
            losses += loss.item()
            cost_predss = np.append(cost_predss, cost_preds.detach().cpu().numpy())

        test_scores, corrs = evaluate(model, val_ds, bs, cost_norm, device, False, database=args.database)
        if epoch > 40:

            if test_scores['q_mean'] < best_prev: ## mean mse
                best_model_path = logging(args, epoch, test_scores, filename = 'log.txt', save_model = True, model = model)
                best_prev = test_scores['q_mean']

        if epoch % 20 == 0:
            # if epoch > 40:
            print('Epoch: {}  Avg Loss: {}, Time: {}, Test Scores: {}'.format(epoch,losses/len(train_ds), time.time()-t0, test_scores))
            # else:
            #     print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch,losses/len(train_ds), time.time()-t0))
            train_scores = print_qerror(cost_norm.unnormalize_labels(cost_predss),cost_labelss, True)
            log_dict['epoch'].append(epoch)
            log_dict['avg loss'].append(losses/len(train_ds))
            log_dict['time'].append(time.time()-t0)
            log_dict['train scores'].append(train_scores)
            log_dict['test scores'].append(test_scores)

        scheduler.step()   

    with open(f'outs/{args.database + ("" if args.query_scale == 1 else str(args.query_scale))}/log_dict.json', 'w') as f:
        json.dump(log_dict, f, indent=4)

    return model, best_model_path


def train_multi(model, train_ds, val_ds, crit, \
    pos_norm, cost_norm, args, optimizer=None, scheduler=None):
    
    to_pred, bs, device, epochs, clip_size = \
        args.to_predict, args.bs, args.device, args.epochs, args.clip_size
    lr = args.lr

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)


    t0 = time.time()

    rng = np.random.default_rng()

    best_prev = 999999


    for epoch in range(epochs):
        losses = 0
        pos_predss = np.empty(0)
        cost_predss = np.empty(0)

        model.train()

        train_idxs = rng.permutation(len(train_ds))

        group_lss = list(zip(*[iter(np.array(train_ds.costs)[train_idxs])]*args.query_num))
        pos_labelss = np.array([np.argmin(g) for g in group_lss])
        cost_labelss = np.array([min(g) for g in group_lss])


        for idxs in chunks(train_idxs, bs):
            optimizer.zero_grad()

            batch, batch_labels = collator(list(zip(*[train_ds[j] for j in idxs])))
            
            l, r = zip(*(batch_labels))
            group_l = list(zip(*[iter(l)]*args.query_num))
            pos_labels = pos_norm.normalize_labels([np.argmin(g) for g in group_l])
            cost_labels = np.array([min(g) for g in group_l])

            batch_pos_label = torch.FloatTensor(pos_labels).to(device)
            batch_cost_label = torch.FloatTensor(cost_labels).to(device)
            batch = batch.to(device)

            pos_preds, cost_preds = model(batch)
            pos_preds = pos_preds.squeeze()
            cost_preds = cost_preds.squeeze()

            loss = crit(pos_preds, batch_pos_label) + crit(cost_preds, batch_cost_label)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)

            optimizer.step()
            losses += loss.item()
            pos_predss = np.append(pos_predss, pos_preds.detach().cpu().numpy())
            cost_predss = np.append(cost_predss, cost_preds.detach().cpu().numpy())

        if epoch > 40:
            test_scores, corrs = evaluate(model, val_ds, bs, cost_norm, device, False, database=args.database)

            if test_scores['q_mean'] < best_prev: ## mean mse
                best_model_path = logging(args, epoch, test_scores, filename = 'log.txt', save_model = True, model = model)
                best_prev = test_scores['q_mean']

        if epoch % 20 == 0:
            print('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch,losses/len(train_ds), time.time()-t0))
            train_scores = print_multi_qerror(pos_norm.unnormalize_labels(pos_predss), pos_labelss, 
                                        cost_norm.unnormalize_labels(cost_predss), cost_labelss, True)

        scheduler.step()   

    return model, best_model_path


def logging(args, epoch, qscores, filename = None, save_model = False, model = None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint 


    res = {**res, **qscores}

    filename = args.newpath + filename
    model_checkpoint = args.newpath + model_checkpoint
    
    if filename is not None:
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            df = df.append(res, ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(res, index=[0])
            df.to_csv(filename, index=False)
    if save_model:
        torch.save({
            'model': model.state_dict(),
            'args' : args
        }, model_checkpoint)
    
    return res['model']  