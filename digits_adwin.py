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
from useful_functions import Q1,Q2,Q3,Q4,keep_last_consecutive,SineGenerator,Q1u,Q2u,Q3u,Q4u
from drift_detection_algorithms import DDM,PageHinkley,Ewma,Adwin
import pandas as pd
from copy import deepcopy
import lightgbm as lgb
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from datasets.get_dataset import get_dataset
import models
import torch.distributions as td
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

np.random.seed(0)
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
print('Load data')
    
task = 'm2u'
batch_size = 64
outdim = 50
initial_batches = 50
label_lag = 3


def train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num):
    model_f.train()
    model_c.train()
    if len(train_xt)==0:
        for t in range(20):
            for i in range(len(train_xs)):
                data_s = train_xs[i]
                target_s = train_ys[i]
                data_s, target_s = data_s.cuda(), target_s.cuda(non_blocking=True)
                optimizer_f.zero_grad()
                optimizer_c.zero_grad()
                feature_s = model_f(data_s)
                output_s = model_c(feature_s)
                loss = criterion_cel(F.softmax(output_s), target_s)
                loss.backward()
                optimizer_f.step()
                optimizer_c.step()
                optimizer_f.zero_grad()
                optimizer_c.zero_grad()
                print(loss.item())
    if len(train_xt)>0:
        t_count = 0
        for t in range(20):
            for i in range(len(train_xs)):
                data_s = train_xs[i]
                target_s = train_ys[i]
                data_t = train_xt[t_count]
                target_t = train_yt[t_count]
                data_s, target_s = data_s.cuda(), target_s.cuda(non_blocking=True)
                data_t, target_t = data_t.cuda(), target_t.cuda(non_blocking=True)
                optimizer_f.zero_grad()
                optimizer_c.zero_grad()
                feature_s = model_f(data_s)
                output_s = model_c(feature_s)
                feature_t = model_f(data_t)
                output_t = model_c(feature_t)
                loss = criterion_cel(F.softmax(output_s), target_s) + criterion_cel(F.softmax(output_t), target_t)*drift_num*1.0/10
                loss.backward()
                optimizer_f.step()
                optimizer_c.step()
                optimizer_f.zero_grad()
                optimizer_c.zero_grad()
                t_count += 1
                if t_count == len(train_yt):
                    t_count = 0

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


'''
count = 0
for batch_idx, (data_s, target_s) in enumerate(target_loader):
    train_xt.append(data_s)
    train_yt.append(target_s)
    count += 1
    if count == 50:
        break
'''


batch_size = 64
T = len(source_dataset)
n_batch = int((T-T%batch_size)/batch_size)
initial_batches = 50
label_lag = 3

### ADWIN in batch

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

train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num)

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

for i in range(initial_batches + label_lag, n_batch):
    if (i-(initial_batches + label_lag)+1)%100 == 0:
        drift_num += 1
        _, target_dataset = get_dataset(task, drift_num)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=int(batch_size*1.0*drift_num/10), shuffle=False, num_workers=0)
        dl_target = iter(target_loader)
    batch_xs, batch_ys = next(dl_source)
    previous_xs.append(batch_xs)
    previous_ys.append(batch_ys)
    if drift_num != 0:
        batch_xt, batch_yt = next(dl_target)
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
        train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num)
        ret_ind.append(i)
        retraining_time += (time.time() - start_time)
        total_retraining_samples += len(train_xs)
        total_retraining_samples += len(train_xt)
        previous_xs, previous_ys = [], []
        previous_xt, previous_yt = [], []
        no_drift_count = 0

print(np.mean(prequential_acc))
