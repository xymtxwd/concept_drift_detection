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
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context
from spn.algorithms.MPE import mpe
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from datasets.get_dataset import get_dataset
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
import torchvision.datasets as td
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import math
import sklearn.preprocessing as skp
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model




np.random.seed(0)
print('Load data')
    


from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context
from spn.algorithms.MPE import mpe
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.algorithms.LearningWrappers import learn_parametric
from spn.algorithms.Inference import log_likelihood

    
class spn_dist_fitter:
    def __init__(self, n_estimators=20):
        self.spnfitter = None
    def fit(self, train_X):
        param_type = [Gaussian for _ in range(train_X.shape[1])]
        self.spnfitter = learn_parametric(train_X, 
                                      Context(parametric_types=param_type).add_domains(train_X), 
                                      min_instances_slice=20)
    def score(self, test_X):
        return np.mean(log_likelihood(self.spnfitter, test_X))

def train_clf(model_f, model_c, train_xs, train_ys):
    model_f.train()
    model_c.train()
    if True:
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
                #print(loss.item())

def nn_score(model_f, model_c, train_xs, train_ys):
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
    return correct*1.0/count

criterion_cel = nn.CrossEntropyLoss()

### change to your own model for your data
outdim = 50
model_f = models.Net_f(outdim=outdim).cuda()
model_c = models.Net_c_cway(outdim=outdim).cuda()
model_de = models.decoder(outdim=outdim).cuda()
optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
optimizer_de = torch.optim.Adam(model_de.parameters(), 0.001)

### change .txt file name, this .txt file should contains one image folder address per line. Please also prepare the image folders in the correct format 
### Please refer to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision
image_batches = []
with open('image_folder_addresses.txt') as f:
    for line in f:
        temp_dataset = td.ImageFolder(root=line)
        temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=len(temp_dataset), shuffle=False, num_workers=0)
        temp_dl = iter(temp_loader)
        data_s, target_s = next(temp_dl)
        image_batches.append([data_s, target_s])

### use how many batches to train initially
train_batch_num = 50


use_kfac = True

if use_kfac:
    kfac_optim = KFACOptimizer(nn.Sequential(model_f,model_c),
                              lr=0.01,
                              momentum=0.9,
                              stat_decay=0.95,
                              damping=1e-3,
                              kl_clip=1e-2,
                              weight_decay=3e-3,
                              TCov=10,
                              TInv=100)


train_xs, train_ys = [], []
train_xtogether, train_ytogether = [], []
previous_xtogether, previous_ytogether = [], []

for i in range(train_batch_num):
    data_s, target_s = image_batches.pop(0)
    train_xs.append(data_s)
    train_ys.append(target_s)
    train_xtogether.append(data_s)
    train_ytogether.append(target_s)
    previous_xtogether.append(data_s)
    previous_ytogether.append(target_s)

train_clf(model_f, model_c, train_xs, train_ys)

previous_xs, previous_ys = [], []

q1_list,q2_list,q3_list,qAE_list,qspn_list,qFS_list = [],[],[],[],[],[]
    
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
q1_drift, q2_drift, q3_drift, qAE_drift,qspn_drift, qFS_drift = False, False, False, False, False, False
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

def train_ae(model_f, model_de, train_xs):
    model_f.eval()
    model_de.train()
    tr_err = 0
    cri = torch.nn.MSELoss()
    if True:
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
    return tr_err
AE_tr_err = train_ae(model_f, model_de, train_xs)

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


for i in range(len(image_batches)):
    batch_xs, batch_ys = image_batches[i]
    previous_xs.append(batch_xs)
    previous_ys.append(batch_ys)
    if True:
        prequential_acc.append(nn_score(model_f, model_c, [batch_xs], [batch_ys]))
        previous_xtogether.append(batch_xs)
        previous_ytogether.append(batch_ys)
        data_X = previous_xs
        data_Y = previous_ys
    q1_list.append(Q1u(previous_xtogether, previous_ytogether, label_lag, model_f,model_c,window_size=window_size,alpha=alpha, beta=beta, p1_p2_weights=p1_p2_weights))#*0.99+Q2u(previous_xtogether, label_lag, train_xtogether[-50:], model_f)*0.01)
    q2_list.append(Q3u(previous_xtogether, model_f, model_c))
    q3_list.append(Q4u(previous_xtogether, train_xtogether[-50:], model_f))
    qAE_list.append(math.tanh(test_ae(model_f, model_de, previous_xtogether[-1])/AE_tr_err/2))
    #qAE_list.append(math.tanh(test_ae(model_f, model_de, previous_xtogether[-1])))
    qspn_list.append(np.log(test_spn(model_f, spn, previous_xtogether[-1])))

    if use_kfac:# and kfac_optim.steps % kfac_optim.TCov == 0:
        # compute true fisher
        kfac_optim.zero_grad()
        feat = model_f(previous_xtogether[-1-label_lag].cuda())
        outputs = model_c(feat)
        loss = criterion_cel(F.softmax(outputs), previous_ytogether[-1-label_lag].cuda())

        kfac_optim.acc_stats = True
        with torch.no_grad():
            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                          1).squeeze().cuda()
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

    if q1_drift*3+q2_drift*1+q3_drift*1+qAE_drift*1+qspn_drift*1+qFS_drift*1>=3:
        first_training_index = np.min([first_training_index, i-label_lag])
        start_time = time.time()
        print('CHANGE DETECTED at '+str(i))
        drift_list.append(i)
        print('retrain starting dataset index '+ str(first_training_index))
        train_xs, train_ys = [], []
        train_xtogether, train_ytogether = [], []
        for j in range(first_training_index, i):
            train_xs.append(previous_xs[j-label_lag])
            train_ys.append(previous_ys[j-label_lag])
            train_xtogether.append(previous_xs[j-label_lag])
            train_ytogether.append(previous_ys[j-label_lag])
        model_f = models.Net_f(outdim=outdim).cuda()
        model_c = models.Net_c_cway(outdim=outdim).cuda()
        model_de = models.decoder(outdim=outdim).cuda()
        optimizer_f = torch.optim.Adam(model_f.parameters(), 0.001)
        optimizer_c = torch.optim.Adam(model_c.parameters(), 0.001)
        optimizer_de = torch.optim.Adam(model_de.parameters(), 0.001)
        train_clf(model_f, model_c, train_xtogether, train_ytogether)
        AE_tr_err = train_ae(model_f, model_de, train_xs)
        warning_index = []
        warning_index_2 = []
        warning_index_3 = []
        warning_index_AE = []
        warning_index_spn = []
        warning_index_FS = []
        q1_drift, q2_drift, q3_drift, qAE_drift, qspn_drift, qFS_drift = False, False, False, False, False, False

        if use_kfac:
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
