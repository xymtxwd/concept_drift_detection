import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np, sklearn
from sklearn.naive_bayes import GaussianNB
import scipy as sp
from sklearn.ensemble import IsolationForest, RandomForestClassifier
np.random.seed(0)
from sklearn import svm
from sklearn.model_selection import cross_val_score
from scipy.optimize import curve_fit
import math
from scipy import asarray as ar,exp
from scipy.stats import norm
import sys
from useful_functions import Q1,Q2,Q3,Q4,keep_last_consecutive,SineGenerator,Q1u,Q2u,Q3u,Q4u,nn_score,train_clf
from drift_detection_algorithms import DDM,PageHinkley,Ewma,Adwin
import pandas as pd
from copy import deepcopy
import lightgbm as lgb
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datasets.get_dataset import get_dataset
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import models
import utils
import torch.distributions as td
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import os
from optimizers.kfac import KFACOptimizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from utils_KFAC.network_utils import get_network
from utils_KFAC.data_utils import get_dataloader
from spn.algorithms.layerwise.distributions import Normal
from spn.algorithms.layerwise.layers import Sum, Product
import torch
from torch import nn
from spn.algorithms.layerwise.clipper import DistributionClipper

import argparse
parser = argparse.ArgumentParser(description='Online Concept Drift Detection')
parser.add_argument('--model', choices=['ours', 'ddm', 'ph','adwin','ewma'], default='ours', help='type of model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size (default: 64)')
args = parser.parse_args()

np.random.seed(0)
print('Load data')



task = 'm2u'
batch_size = 64
outdim = 50
initial_batches = 50
label_lag = 3
criterion_cel = nn.CrossEntropyLoss()

def nn_score(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num):
    pred_y = []
    correct = 0
    count = 0
    for i in range(len(train_xs)):
        data_s = train_xs[i]
        target_s = train_ys[i]
        data_s, target_s = data_s.cuda(), target_s.cuda(non_blocking=True)
        feature_s = model_f(data_s)
        output = model_c(feature_s)
        pred = output.max(1, keepdim=True)[1]
        for i in range(len(pred)):
            pred_y.append(pred[i].item())
        correct += pred.eq(target_s.view_as(pred)).sum().item()
        count += len(target_s)
    if len(train_xt)!=0:
        for i in range(len(train_xt)):
            data_t = train_xt[i]
            target_t = train_yt[i]
            data_t, target_t = data_t.cuda(), target_t.cuda(non_blocking=True)
            feature_t = model_f(data_t)
            output = model_c(feature_t)
            pred = output.max(1, keepdim=True)[1]
            for i in range(len(pred)):
                pred_y.append(pred[i].item())
            correct += pred.eq(target_t.view_as(pred)).sum().item()
            count += len(target_t)
    return correct*1.0/count

### DDM in batch
if args.model=='ddm':
    criterion_cel = nn.CrossEntropyLoss()
    
    model_f = models.Net_f(task=task, outdim=outdim).cuda()
    model_c = models.Net_c_cway(task=task, outdim=outdim).cuda()
    optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
    optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
    
    drift_num = 0
    source_dataset, target_dataset = get_dataset(task, drift_num)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_xs, train_ys = [], []
    train_xt, train_yt = [], []
    
    dl_source = iter(source_loader)
    dl_target = iter(target_loader)
    
    count = 0
    for i in range(initial_batches):
        data_s, target_s = next(dl_source)
        train_xs.append(data_s)
        train_ys.append(target_s)
    
    drift_num = 0
    source_dataset, target_dataset = get_dataset(task, drift_num)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_xs, train_ys = [], []
    train_xt, train_yt = [], []
    
    dl_source = iter(source_loader)
    dl_target = iter(target_loader)
    
    count = 0
    for i in range(50):
        data_s, target_s = next(dl_source)
        train_xs.append(data_s)
        train_ys.append(target_s)
    
    
    batch_size = 64
    T = len(source_dataset)
    n_batch = int((T-T%batch_size)/batch_size)
    initial_batches = 50
    label_lag = 3
    
    train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
    
    first_undrift_index = initial_batches+1
    dd = DDM()
    warning_index = []
    drift_list = []
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    ret_ind = []
    previous_xs, previous_ys = [], []
    previous_xt, previous_yt = [], []
    no_drift_count = 0
    
    sample_count = 0
    
    for i in range(initial_batches + label_lag, n_batch):
        if (i-(initial_batches + label_lag)+1)%100 == 0:
            drift_num += 1
            _, target_dataset = get_dataset(task, drift_num)
            target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            dl_target = iter(target_loader)
            print('Drift ' + str(drift_num))
            print(np.mean(prequential_acc))
            sample_count = 0
        batch_xs, batch_ys = next(dl_source)
        previous_xs.append(batch_xs)
        previous_ys.append(batch_ys)
        sample_count += 1
        if drift_num != 0:
            try:
                batch_xt, batch_yt = next(dl_target)
            except StopIteration:
                dl_target = iter(target_loader)
                batch_xt, batch_yt = next(dl_target)
            batch_xt, batch_yt = batch_xt[:int(sample_count/2+2)], batch_yt[:int(sample_count/2+2)]
            previous_xt.append(batch_xt)
            previous_yt.append(batch_yt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [batch_xt], [batch_yt], drift_num))
        else:
            batch_xt, batch_yt = [], []
            previous_xt.append(batch_xt)
            previous_yt.append(batch_xt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [], [], drift_num))
        no_drift_count += 1
        if len(prequential_acc)>label_lag-1 and dd.set_input(1 - prequential_acc[-label_lag]):
            #first_training_index = np.min([first_training_index, i-label_lag])
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            #warning_index.append(no_drift_count)
            print('retrain using dataset index '+ str(keep_last_consecutive(warning_index)))
            model_f = models.Net_f(task=task, outdim=outdim).cuda()
            model_c = models.Net_c_cway(task=task, outdim=outdim).cuda()
            optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
            optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
            train_xs, train_ys = [], []
            train_xt, train_yt = [], []
            for j in keep_last_consecutive(warning_index):
                train_xs.append(previous_xs[j])
                train_ys.append(previous_ys[j])
                if len(previous_xt[j])==0:
                    continue
                train_xt.append(previous_xt[j])
                train_yt.append(previous_yt[j])
            train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
            ret_ind.append(keep_last_consecutive(warning_index))
            warning_index=[]
            retraining_time += (time.time() - start_time)
            total_retraining_samples += len(train_xs)
            total_retraining_samples += len(train_xt)
            previous_xs, previous_ys = [], []
            previous_xt, previous_yt = [], []
            no_drift_count = 0
        if dd.is_warning_zone:
            warning_index.append(no_drift_count)
    
    print(np.mean(prequential_acc))

### PH in batch
if args.model=='ph':
    model_f = models.Net_f(task=task).cuda()
    model_c = models.Net_c_cway(task=task).cuda()
    optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
    optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
    
    drift_num = 0
    source_dataset, target_dataset = get_dataset(task, drift_num)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_xs, train_ys = [], []
    train_xt, train_yt = [], []
    
    dl_source = iter(source_loader)
    dl_target = iter(target_loader)
    
    count = 0
    for i in range(50):
        data_s, target_s = next(dl_source)
        train_xs.append(data_s)
        train_ys.append(target_s)
    
    ph = PageHinkley(threshold=0.3,delta=0.005)

    T = len(source_dataset)
    n_batch = int((T-T%batch_size)/batch_size)
    initial_batches = 50
    label_lag = 3

    train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
    
    first_undrift_index = initial_batches+1
    warning_index = []
    drift_list = []
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    ret_ind = []
    previous_xs, previous_ys = [], []
    previous_xt, previous_yt = [], []
    no_drift_count = 0
    
    sample_count = 0
    
    for i in range(initial_batches + label_lag, n_batch):
        if (i-(initial_batches + label_lag)+1)%100 == 0:
            drift_num += 1
            _, target_dataset = get_dataset(task, drift_num)
            target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            dl_target = iter(target_loader)
            print('Drift ' + str(drift_num))
            print(np.mean(prequential_acc))
            sample_count = 0
        batch_xs, batch_ys = next(dl_source)
        previous_xs.append(batch_xs)
        previous_ys.append(batch_ys)
        sample_count += 1
        if drift_num != 0:
            try:
                batch_xt, batch_yt = next(dl_target)
            except StopIteration:
                dl_target = iter(target_loader)
                batch_xt, batch_yt = next(dl_target)
            batch_xt, batch_yt = batch_xt[:int(sample_count/2+2)], batch_yt[:int(sample_count/2+2)]
            previous_xt.append(batch_xt)
            previous_yt.append(batch_yt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [batch_xt], [batch_yt], drift_num))
        else:
            batch_xt, batch_yt = [], []
            previous_xt.append(batch_xt)
            previous_yt.append(batch_xt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [], [], drift_num))
        no_drift_count += 1
        if len(prequential_acc)>label_lag-1:
            ph.add_element(1 - prequential_acc[-label_lag])
        if ph.in_concept_change:
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            #warning_index.append(no_drift_count)
            print('retrain using dataset index '+ str(keep_last_consecutive(warning_index)))
            model_f = models.Net_f(task=task).cuda()
            model_c = models.Net_c_cway(task=task).cuda()
            optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
            optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
            train_xs, train_ys = [], []
            train_xt, train_yt = [], []
            for j in keep_last_consecutive(warning_index):
                train_xs.append(previous_xs[j])
                train_ys.append(previous_ys[j])
                if len(previous_xt[j])==0:
                    continue
                train_xt.append(previous_xt[j])
                train_yt.append(previous_yt[j])
            if len(train_xs)==0:
                train_xs, train_ys = [batch_xs], [batch_ys]
                if len(batch_xt) == 0:
                    train_xt, train_yt = [], []
                else:
                    train_xt, train_yt = [batch_xt], [batch_yt]
            train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
            ret_ind.append(keep_last_consecutive(warning_index))
            warning_index=[]
            retraining_time += (time.time() - start_time)
            total_retraining_samples += len(train_xs)
            total_retraining_samples += len(train_xt)
            previous_xs, previous_ys = [], []
            previous_xt, previous_yt = [], []
            no_drift_count = 0
        if ph.in_warning_zone:
            warning_index.append(no_drift_count)
    
    print(np.mean(prequential_acc))
    
### ADWIN in batch
if args.model=='adwin':
    model_f = models.Net_f(task=task).cuda()
    model_c = models.Net_c_cway(task=task).cuda()
    optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
    optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
    
    drift_num = 0
    source_dataset, target_dataset = get_dataset(task, drift_num)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_xs, train_ys = [], []
    train_xt, train_yt = [], []
    
    dl_source = iter(source_loader)
    dl_target = iter(target_loader)
    
    count = 0
    for i in range(50):
        data_s, target_s = next(dl_source)
        train_xs.append(data_s)
        train_ys.append(target_s)
    
    ad = Adwin(delta=1)
    T = len(source_dataset)
    n_batch = int((T-T%batch_size)/batch_size)
    train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
    
    first_undrift_index = initial_batches+1
    warning_index = []
    drift_list = []
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    ret_ind = []
    previous_xs, previous_ys = [], []
    previous_xt, previous_yt = [], []
    no_drift_count = 0
    
    sample_count = 0
    
    for i in range(initial_batches + label_lag, n_batch):
        if (i-(initial_batches + label_lag)+1)%100 == 0:
            drift_num += 1
            _, target_dataset = get_dataset(task, drift_num)
            target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            dl_target = iter(target_loader)
            print('Drift ' + str(drift_num))
            print(np.mean(prequential_acc))
            sample_count = 0
        batch_xs, batch_ys = next(dl_source)
        previous_xs.append(batch_xs)
        previous_ys.append(batch_ys)
        sample_count += 1
        if drift_num != 0:
            try:
                batch_xt, batch_yt = next(dl_target)
            except StopIteration:
                dl_target = iter(target_loader)
                batch_xt, batch_yt = next(dl_target)
            batch_xt, batch_yt = batch_xt[:int(sample_count/2+2)], batch_yt[:int(sample_count/2+2)]
            previous_xt.append(batch_xt)
            previous_yt.append(batch_yt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [batch_xt], [batch_yt], drift_num))
        else:
            batch_xt, batch_yt = [], []
            previous_xt.append(batch_xt)
            previous_yt.append(batch_xt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [], [], drift_num))
        no_drift_count += 1
        if len(prequential_acc)>label_lag-1 and ad.set_input(1 - prequential_acc[-label_lag]):
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            #warning_index.append(no_drift_count)
            print('retrain using dataset index '+ str(i))
            model_f = models.Net_f(task=task).cuda()
            model_c = models.Net_c_cway(task=task).cuda()
            optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
            optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
            train_xs, train_ys = [], []
            train_xt, train_yt = [], []
            if True:
                train_xs, train_ys = [batch_xs], [batch_ys]
                if len(batch_xt) == 0:
                    train_xt, train_yt = [], []
                else:
                    train_xt, train_yt = [batch_xt], [batch_yt]
            train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
            ret_ind.append(i)
            retraining_time += (time.time() - start_time)
            total_retraining_samples += len(train_xs)
            total_retraining_samples += len(train_xt)
            previous_xs, previous_ys = [], []
            previous_xt, previous_yt = [], []
            no_drift_count = 0
    
    print(np.mean(prequential_acc))


    ### EWMA in batch
if args.model=='ewma':

    model_f = models.Net_f(task=task).cuda()
    model_c = models.Net_c_cway(task=task).cuda()
    optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
    optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
    
    drift_num = 0
    source_dataset, target_dataset = get_dataset(task, drift_num)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_xs, train_ys = [], []
    train_xt, train_yt = [], []
    
    dl_source = iter(source_loader)
    dl_target = iter(target_loader)
    
    count = 0
    for i in range(50):
        data_s, target_s = next(dl_source)
        train_xs.append(data_s)
        train_ys.append(target_s)
    
    ewma = Ewma(alpha=0.1, coefficient=3)
    error_list = []
    T = len(source_dataset)
    n_batch = int((T-T%batch_size)/batch_size)

    train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
    
    first_undrift_index = initial_batches+1
    warning_index = []
    drift_list = []
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    ret_ind = []
    previous_xs, previous_ys = [], []
    previous_xt, previous_yt = [], []
    no_drift_count = 0
    
    sample_count = 0
    
    for i in range(initial_batches + label_lag, n_batch):
        if (i-(initial_batches + label_lag)+1)%100 == 0:
            drift_num += 1
            _, target_dataset = get_dataset(task, drift_num)
            target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            dl_target = iter(target_loader)
            print('Drift ' + str(drift_num))
            print(np.mean(prequential_acc))
            sample_count = 0
        batch_xs, batch_ys = next(dl_source)
        previous_xs.append(batch_xs)
        previous_ys.append(batch_ys)
        sample_count += 1
        if drift_num != 0:
            try:
                batch_xt, batch_yt = next(dl_target)
            except StopIteration:
                dl_target = iter(target_loader)
                batch_xt, batch_yt = next(dl_target)
            batch_xt, batch_yt = batch_xt[:int(sample_count/2+2)], batch_yt[:int(sample_count/2+2)]
            previous_xt.append(batch_xt)
            previous_yt.append(batch_yt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [batch_xt], [batch_yt], drift_num))
        else:
            batch_xt, batch_yt = [], []
            previous_xt.append(batch_xt)
            previous_yt.append(batch_xt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [], [], drift_num))
        no_drift_count += 1
        if len(prequential_acc)>label_lag-1:
            error_list.append(1-prequential_acc[-label_lag])
        if len(prequential_acc)>label_lag-1 and ewma.predict(error_list)==0:
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            #warning_index.append(no_drift_count)
            print('retrain using dataset index '+ str(i))
            model_f = models.Net_f(task=task).cuda()
            model_c = models.Net_c_cway(task=task).cuda()
            optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
            optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
            train_xs, train_ys = [], []
            train_xt, train_yt = [], []
            if True:
                train_xs, train_ys = [batch_xs], [batch_ys]
                if len(batch_xt) == 0:
                    train_xt, train_yt = [], []
                else:
                    train_xt, train_yt = [batch_xt], [batch_yt]
            train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
            ret_ind.append(i)
            retraining_time += (time.time() - start_time)
            total_retraining_samples += len(train_xs)
            total_retraining_samples += len(train_xt)
            previous_xs, previous_ys = [], []
            previous_xt, previous_yt = [], []
            no_drift_count = 0
            error_list = []
    
    print(np.mean(prequential_acc))

### our drift detection model
if args.model=='ours':
    task = 'm2u'
    batch_size = 64
    outdim = 50
    initial_batches = 50
    label_lag = 3
    drift_num = 0
    
    #initialize models
    model_f = models.Net_f(task=task, outdim=outdim).cuda() #encoder
    model_c = models.Net_c_cway(task=task, outdim=outdim).cuda()#classifier
    model_de = models.decoder(task=task, outdim=outdim).cuda() #decoder
    optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
    optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
    optimizer_de = torch.optim.Adam(model_de.parameters(), 0.001)
    
    #load data
    source_dataset, target_dataset = get_dataset(task, drift_num)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_xs, train_ys = [], []
    train_xt, train_yt = [], []
    train_xtogether, train_ytogether = [], []
    previous_xtogether, previous_ytogether = [], []
    
    dl_source = iter(source_loader)
    dl_target = iter(target_loader)
    
    for i in range(initial_batches):
        data_s, target_s = next(dl_source)
        train_xs.append(data_s)
        train_ys.append(target_s)
        train_xtogether.append(data_s)
        train_ytogether.append(target_s)
        previous_xtogether.append(data_s)
        previous_ytogether.append(target_s)
    
    
    T = len(source_dataset)
    n_batch = int((T-T%batch_size)/batch_size)
    
    #train classifier
    train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
    
    first_undrift_index = initial_batches+1
    previous_xs, previous_ys = [], []
    previous_xt, previous_yt = [], []
    no_drift_count = 0
    
    q1_list,q2_list,q3_list,qAE_list,qspn_list,qFS_list = [],[],[],[],[],[]

    kfac_optim = KFACOptimizer(nn.Sequential(model_f,model_c),
                              lr=0.01,
                              momentum=0.9,
                              stat_decay=0.95,
                              damping=1e-3,
                              kl_clip=1e-2,
                              weight_decay=3e-3,
                              TCov=10,
                              TInv=100)
    dd = DDM(3,2)
    warning_index = []
    dd_2 = DDM(3,2)
    warning_index_2 = []
    dd_3 = DDM(3,2)
    warning_index_3 = []
    dd_AE = DDM(3,2)
    warning_index_AE = []
    dd_spn = DDM(3,2)
    warning_index_spn = []
    dd_FS = DDM(3,2)
    warning_index_FS = []
    q1_drift, q2_drift, q3_drift, qAE_drift,qspn_drift,qFS_drift = False, False, False, False, False, False
    first_training_index = sys.maxsize
    drift_1, drift_2, drift_3, drift_AE, drift_spn, drift_FS = [],[],[],[],[],[]
    
    drift_list = []
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    label_lag = 3
    p1_p2_weights = [0.00, 1.00]
    window_size, alpha, beta = 3, 0.1, 0.1
    keep_last = 10
    
    def train_ae(model_f, model_de, train_xs, train_xt, drift_num):
        model_f.eval()
        model_de.train()
        tr_err = 0
        cri = torch.nn.MSELoss()
        if len(train_xt)==0:
            for t in range(30):
                for i in range(len(train_xs)):
                    data_s = train_xs[i]
                    data_s = data_s.cuda()
                    optimizer_de.zero_grad()
                    feature_s = model_f(data_s)
                    output_s = model_de(feature_s)
                    loss = cri(output_s, data_s)
                    loss.backward()
                    optimizer_de.step()
                    optimizer_de.zero_grad()
                    #print(loss.item())
                    tr_err = loss.item()
        if len(train_xt)>0:
            t_count = 0
            for t in range(30):
                for i in range(len(train_xs)):
                    data_s = train_xs[i]
                    data_t = train_xt[t_count]
                    data_s = data_s.cuda()
                    data_t = data_t.cuda()
                    optimizer_de.zero_grad()
                    feature_s = model_f(data_s)
                    output_s = model_de(feature_s)
                    feature_t = model_f(data_t)
                    output_t = model_de(feature_t)
                    loss = cri(output_s, data_s) + cri(output_t, data_t)*drift_num*1.0/10
                    loss.backward()
                    optimizer_de.step()
                    optimizer_de.zero_grad()
                    tr_err = loss.item()
                    t_count += 1
                    if t_count == len(train_xt):
                        t_count = 0
        return tr_err
    AE_tr_err = train_ae(model_f, model_de, train_xs, train_xt, drift_num)
    
    def test_ae(model_f, model_de, test_x):
        model_f.eval()
        model_de.eval()
        cri = torch.nn.MSELoss()
        test_x = test_x.cuda()
        feature = model_f(test_x)
        output = model_de(feature)
        loss = cri(output, test_x)
        return loss.item()
    
    gauss = Normal(multiplicity=5, in_features=50)
    prod1 = Product(in_features=50, cardinality=5)
    sum1 = Sum(in_features=10, in_channels=5, out_channels=1)
    prod2 = Product(in_features=10, cardinality=10)
    spn = nn.Sequential(gauss, prod1, sum1, prod2).cuda()
    clipper = DistributionClipper()
    optimizer_spn = torch.optim.Adam(spn.parameters(), lr=0.001)
    optimizer_spn.zero_grad()
    
    #temp_loss = []
    def train_spn(model_f, spn, train_x):
        model_f.eval()
        spn.train()
        if True:
            for t in range(200):
                for i in range(len(train_x)):
                    data = train_x[i]
                    data = data.cuda()
                    feature = model_f(data)
                    output = spn(feature)
                    loss = -1 * output.mean()
                    loss.backward()
                    optimizer_spn.step()
                    spn.apply(clipper)
                    optimizer_spn.zero_grad()
                    #temp_loss.append(loss.item())
    train_spn(model_f, spn, train_xs)
    
    def test_spn(model_f, spn, test_x):
        model_f.eval()
        spn.eval()
        test_x = test_x.cuda()
        feature = model_f(test_x)
        output = spn(feature)
        loss = -1 * output.mean()
        return loss.item()
    
    
    ### start monitoring
    
    sample_count = 0
    
    for i in range(initial_batches + label_lag, n_batch):
        if (i-(initial_batches + label_lag)+1)%100 == 0:
            drift_num += 1
            _, target_dataset = get_dataset(task, drift_num)
            target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            dl_target = iter(target_loader)
            print('Drift ' + str(drift_num))
            print(np.mean(prequential_acc))
            sample_count = 0
        batch_xs, batch_ys = next(dl_source)
        previous_xs.append(batch_xs)
        previous_ys.append(batch_ys)
        sample_count += 1
        if drift_num != 0:
            try:
                batch_xt, batch_yt = next(dl_target)
            except StopIteration:
                dl_target = iter(target_loader)
                batch_xt, batch_yt = next(dl_target)
            batch_xt, batch_yt = batch_xt[:int(sample_count/2+2)], batch_yt[:int(sample_count/2+2)]
            previous_xt.append(batch_xt)
            previous_yt.append(batch_yt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [batch_xt], [batch_yt], drift_num))
            previous_xtogether.append(torch.cat([batch_xs,batch_xt]))
            previous_ytogether.append(torch.cat([batch_ys,batch_yt]))
        else:
            batch_xt, batch_yt = [], []
            previous_xt.append(batch_xt)
            previous_yt.append(batch_yt)
            prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys], [], [], drift_num))
            previous_xtogether.append(batch_xs)
            previous_ytogether.append(batch_ys)
        no_drift_count += 1    
        if len(previous_xt)!=0:
            data_X = previous_xs + previous_xt
            data_Y = previous_ys + previous_yt
        else:
            data_X = previous_xs
            data_Y = previous_ys
        # each drift detector receives a score for drift detection
        q1_list.append(Q1u(previous_xtogether, previous_ytogether, label_lag, model_f,model_c,window_size=window_size,alpha=alpha, beta=beta, p1_p2_weights=p1_p2_weights))#*0.99+Q2u(previous_xtogether, label_lag, train_xtogether[-50:], model_f)*0.01)
        q2_list.append(Q3u(previous_xtogether, model_f, model_c))
        q3_list.append(Q4u(previous_xtogether, train_xtogether[-50:], model_f))
        qAE_list.append(math.tanh(test_ae(model_f, model_de, previous_xtogether[-1])/AE_tr_err/2))
        qspn_list.append(np.log(test_spn(model_f, spn, previous_xtogether[-1])))

        # compute true fisher
        kfac_optim.zero_grad()
        feat = model_f(previous_xtogether[-1-label_lag].cuda())
        outputs = model_c(feat)
        loss = criterion_cel(F.softmax(outputs), previous_ytogether[-1-label_lag].cuda())

        kfac_optim.acc_stats = True
        with torch.no_grad():
            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1), 1).squeeze().cuda()
        loss_sample = criterion_cel(F.softmax(outputs), sampled_y)
        loss_sample.backward(retain_graph=True)
        #loss_sample.backward()
        kfac_optim.acc_stats = False
        kfac_optim.zero_grad()  # clear the gradient for computing true-fisher.
        loss.backward()
        fisher_score = kfac_optim.step()

        qFS_list.append(fisher_score)

        prequential_acc.append(nn_score(model_f, model_c, [previous_xtogether[-1]], [previous_ytogether[-1]], [], [], 0))
        if q1_drift == False:
            if dd.set_input(q1_list[-1])==True:
                q1_drift=True
                if warning_index!=[]:
                    first_training_index = np.min([first_training_index, keep_last_consecutive(warning_index)[0]])
                else:
                    first_training_index = np.min([first_training_index, i])
                drift_1.append(i)
            elif dd.is_warning_zone:
                warning_index.append(i)
        if q2_drift == False:
            if dd_2.set_input(q2_list[-1])==True:
                q2_drift=True
                if warning_index_2!=[]:
                    first_training_index = np.min([first_training_index, keep_last_consecutive(warning_index_2)[0]]) 
                else:
                    first_training_index = np.min([first_training_index, i]) 
                drift_2.append(i)
            elif dd_2.is_warning_zone:
                warning_index_2.append(i)
        if q3_drift == False:
            if dd_3.set_input(q3_list[-1])==True:
                q3_drift=True
                if warning_index_3!=[]:
                    first_training_index = np.min([first_training_index, keep_last_consecutive(warning_index_3)[0]]) 
                else:
                    first_training_index = np.min([first_training_index, i]) 
                drift_3.append(i)
            elif dd_3.is_warning_zone:
                warning_index_3.append(i)
        if qAE_drift == False:
            if dd_AE.set_input(qAE_list[-1])==True:
                qAE_drift=True
                if warning_index_AE!=[]:
                    first_training_index = np.min([first_training_index, keep_last_consecutive(warning_index_AE)[0]]) 
                else:
                    first_training_index = np.min([first_training_index, i]) 
                drift_AE.append(i)
            elif dd_AE.is_warning_zone:
                warning_index_AE.append(i)
        if qspn_drift == False:
            if dd_spn.set_input(qspn_list[-1])==True:
                qspn_drift=True
                if warning_index_spn!=[]:
                    first_training_index = np.min([first_training_index, keep_last_consecutive(warning_index_spn)[0]]) 
                else:
                    first_training_index = np.min([first_training_index, i]) 
                drift_spn.append(i)
            elif dd_spn.is_warning_zone:
                warning_index_spn.append(i)
        if qFS_drift == False:
            if dd_FS.set_input(qFS_list[-1])==True:
                qFS_drift=True
                if warning_index_FS!=[]:
                    first_training_index = np.min([first_training_index, keep_last_consecutive(warning_index_FS)[0]]) 
                else:
                    first_training_index = np.min([first_training_index, i]) 
                drift_FS.append(i)
            elif dd_FS.is_warning_zone:
                warning_index_FS.append(i)
        # determine if drift has occurred by tailored majority voting
        if q1_drift*3+q2_drift*1+q3_drift*1+qAE_drift*1+qspn_drift*1+qFS_drift*1>=3:
            first_training_index = np.min([first_training_index, i-label_lag])
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            print('retrain starting dataset index '+ str(first_training_index))
            train_xs, train_ys = [], []
            train_xt, train_yt = [], []
            train_xtogether, train_ytogether = [], []
            for j in range(first_training_index, i):
                train_xs.append(previous_xs[j-(initial_batches + label_lag)])
                train_ys.append(previous_ys[j-(initial_batches + label_lag)])
                if len(previous_xt[j-(initial_batches + label_lag)])==0:
                    train_xtogether.append(previous_xs[j-(initial_batches + label_lag)])
                    train_ytogether.append(previous_ys[j-(initial_batches + label_lag)])
                else:
                    train_xtogether.append(torch.cat([previous_xs[j-(initial_batches + label_lag)],previous_xt[j-(initial_batches + label_lag)]]))
                    train_ytogether.append(torch.cat([previous_ys[j-(initial_batches + label_lag)],previous_yt[j-(initial_batches + label_lag)]]))
                    train_xt.append(previous_xt[j-(initial_batches + label_lag)])
                    train_yt.append(previous_yt[j-(initial_batches + label_lag)])
            # retrain the model
            model_f = models.Net_f(task=task, outdim=outdim).cuda()
            model_c = models.Net_c_cway(task=task, outdim=outdim).cuda()
            model_de = models.decoder(task=task, outdim=outdim).cuda()
            optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
            optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
            optimizer_de = torch.optim.Adam(model_de.parameters(), 0.001)
            train_clf(model_f, model_c, train_xtogether, train_ytogether, [], [], drift_num, optimizer_f, optimizer_c)

            AE_tr_err = train_ae(model_f, model_de, train_xs, train_xt, drift_num)
            warning_index = []
            warning_index_2 = []
            warning_index_3 = []
            warning_index_AE = []
            warning_index_spn = []
            warning_index_FS = []
            q1_drift, q2_drift, q3_drift, qAE_drift, qspn_drift, qFS_drift = False, False, False, False, False, False
    
            kfac_optim = KFACOptimizer(nn.Sequential(model_f,model_c),
                                          lr=0.01,
                                          momentum=0.9,
                                          stat_decay=0.95,
                                          damping=1e-3,
                                          kl_clip=1e-2,
                                          weight_decay=3e-3,
                                          TCov=10,
                                          TInv=100)
    
            gauss = Normal(multiplicity=5, in_features=50)
            prod1 = Product(in_features=50, cardinality=5)
            sum1 = Sum(in_features=10, in_channels=5, out_channels=1)
            prod2 = Product(in_features=10, cardinality=10)
            spn = nn.Sequential(gauss, prod1, sum1, prod2).cuda()
            clipper = DistributionClipper()
            optimizer_spn = torch.optim.Adam(spn.parameters(), lr=0.001)
            optimizer_spn.zero_grad()
            train_spn(model_f, spn, train_xs)
    
            retraining_time += (time.time() - start_time)
            first_training_index = sys.maxsize
            print(i)
    print(np.mean(prequential_acc))
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.plot(q1_list)
    plt.savefig('q1_list')
    plt.clf()
    plt.plot(q2_list)
    plt.savefig('q2_list')
    plt.clf()
    plt.plot(q3_list)
    plt.savefig('q3_list')
    plt.clf()
    plt.plot(qAE_list)
    plt.savefig('qAE_list')
    plt.clf()
    plt.plot(qspn_list)
    plt.savefig('qspn_list')
    plt.clf()
    plt.plot(qFS_list)
    plt.savefig('qFS_list')
    plt.clf()
    print(drift_list)
    print(drift_1)
    print(drift_2)
    print(drift_3)
    print(drift_AE)
    print(drift_spn)
    print(drift_FS)
    
