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
print('Load data')





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

q1_list,q2_list,q3_list,qAE_list = [],[],[],[]
    
dd = DDM(3,2)
warning_index = []
dd_2 = DDM(3,2)
warning_index_2 = []
dd_3 = DDM(3,2)
warning_index_3 = []
dd_AE = DDM(3,2)
warning_index_AE = []
q1_drift, q2_drift, q3_drift, qAE_drift = False, False, False, False
first_training_index = sys.maxsize
drift_1, drift_2, drift_3, drift_AE = [],[],[],[]

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

### start monitoring

for i in range(initial_batches + label_lag, n_batch):
    # add drift after each 100 batches, by adding one digit from USPS dataset to MNIST dataset
    if (i-(initial_batches + label_lag)+1)%100 == 0:
        drift_num += 1
        _, target_dataset = get_dataset(task, drift_num)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=int(batch_size*1.0*drift_num/10), shuffle=False, num_workers=0)
        dl_target = iter(target_loader)
        print('Drift ' + str(drift_num))
        print(np.mean(prequential_acc))
    # system receives a batch
    batch_xs, batch_ys = next(dl_source)
    previous_xs.append(batch_xs)
    previous_ys.append(batch_ys)
    if drift_num != 0:
        batch_xt, batch_yt = next(dl_target)
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

    # determine if drift has occurred by tailored majority voting
    if q1_drift*3+q2_drift*1+q3_drift*1+qAE_drift*1>=3:
        first_training_index = np.min([first_training_index, i-label_lag])
        start_time = time.time()
        print('CHANGE DETECTED at '+str(i))
        drift_list.append(i)
        print('retrain starting dataset index '+ str(first_training_index))
        model_f = models.Net_f(task=task, outdim=outdim).cuda()
        model_c = models.Net_c_cway(task=task, outdim=outdim).cuda()
        model_de = models.decoder(task=task, outdim=outdim).cuda()
        optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
        optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
        optimizer_de = torch.optim.Adam(model_de.parameters(), 0.001)
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
        train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c)
        # data augmentation - add some previous data whose distribution differs less from the current training data
        if True:
            from sklearn.model_selection import StratifiedKFold
            kfold = StratifiedKFold(n_splits=5, shuffle=True)
            cvscores = []
            one_train_xtogether = torch.cat(train_xtogether).cpu().numpy()
            one_train_ytogether = torch.cat(train_ytogether).cpu().numpy()
            for train, test in kfold.split(one_train_xtogether, one_train_ytogether):
                tmp_model_f = models.Net_f(task=task, outdim=outdim).cuda()
                tmp_model_c = models.Net_c_cway(task=task, outdim=outdim).cuda()
                tmp_optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
                tmp_optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
                tx, ty = [], []
                trx_sub, try_sub = one_train_xtogether[train], one_train_ytogether[train]
                for ind__ in range(int(len(train)/batch_size)):
                    tx.append(torch.tensor(trx_sub[ind__*batch_size:(ind__+1)*batch_size]))
                    ty.append(torch.tensor(try_sub[ind__*batch_size:(ind__+1)*batch_size]))
                    if ind__== int(len(train)/batch_size)-1 and len(train)%batch_size!=0:
                        if len(train)%batch_size==1: continue
                        else:
                            tx.append(torch.tensor(trx_sub[(ind__+1)*batch_size:]))
                            ty.append(torch.tensor(try_sub[(ind__+1)*batch_size:]))
                train_clf(tmp_model_f, tmp_model_c, tx, ty, [], [], drift_num, tmp_optimizer_f, tmp_optimizer_c)
                scores = nn_score(model_f, model_c, [torch.tensor(one_train_xtogether[test])], [torch.tensor(one_train_ytogether[test])], [], [], 0)
                cvscores.append(scores)
            # if accuracy for a batch is larger than the threshold, consider including it
            threshold = np.mean(cvscores) + 1.65*np.std(scores)
            num = 0
            scores = np.zeros(len(previous_ytogether))
            for p in range(0,len(scores)):
                scores[p] = nn_score(model_f, model_c, [previous_xtogether[p]], [previous_ytogether[p]], [], [], 0)
            maxs = scores.argsort()[-int(len(scores)*0.1):][::-2]
            for p in maxs:
                if nn_score(model_f, model_c, [previous_xtogether[p]], [previous_ytogether[p]], [], [], 0)>threshold:
                    train_xtogether.append(previous_xtogether[p])
                    train_ytogether.append(previous_ytogether[p])
                    num+=1
            print(str(num)+' previous batches appended')
        # retrain the model
        model_f = models.Net_f(task=task, outdim=outdim).cuda()
        model_c = models.Net_c_cway(task=task, outdim=outdim).cuda()
        model_de = models.decoder(task=task, outdim=outdim).cuda()
        optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
        optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
        optimizer_de = torch.optim.Adam(model_de.parameters(), 0.001)
        train_clf(model_f, model_c, train_xtogether, train_ytogether, [], [], drift_num, optimizer_f, optimizer_c)
        AE_tr_err = train_ae(model_f, model_de, train_xs, train_xt, drift_num)
        first_undrift_index = i-np.max([i-first_training_index,label_lag])
        warning_index = []
        warning_index_2 = []
        warning_index_3 = []
        warning_index_AE = []
        q1_drift, q2_drift, q3_drift, qAE_drift = False, False, False, False
        retraining_time += (time.time() - start_time)
        first_training_index = sys.maxsize
    print(i)
print(np.mean(prequential_acc))