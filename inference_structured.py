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
from useful_functions import Q1,Q2,Q3,Q4,keep_last_consecutive,SineGenerator
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


np.random.seed(0)

print('Load data')

### change .txt file name, this .txt file should contain one file address per line. The last column of the file is the label 
batches = []
batch_X = []
batch_Y = []
with open('file_addresses.txt') as f:
    for line in f:
        l = line.replace('\n','')
        data = np.loadtxt(l)
        X = data[:,:-1]
        Y = data[:,-1].astype(int)
        batches.append([X, Y])
        batch_X.append(X)
        batch_Y.append(Y)
    


n_batch = len(batches)

### use how many batches to train initially
train_batch_num = 10
label_lag = 3

    
criterion_cel = nn.CrossEntropyLoss()

class ffn_classifier:
    def __init__(self, n_estimators=20):
        self.model_f = models.Classifier_u2m(50, X.shape[1]).cuda()
        self.model_c = models.Classifier_u2m(2, 50).cuda()
        self.optimizer_f = torch.optim.Adam(self.model_f.parameters(), 0.001)
        self.optimizer_c = torch.optim.Adam(self.model_c.parameters(), 0.001)
    def fit(self, train_X, train_Y):
        self.model_f.train()
        self.model_c.train()
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        if True:
            for t in range(20):
                for i in range(int(len(train_X)/32)):
                    data_s = train_X[i*32:(i+1)*32]
                    target_s = train_Y[i*32:(i+1)*32]
                    data_s, target_s = torch.from_numpy(data_s).float().cuda(), torch.from_numpy(target_s).cuda(non_blocking=True)
                    self.optimizer_f.zero_grad()
                    self.optimizer_c.zero_grad()
                    feature_s = self.model_f(data_s)
                    output_s = self.model_c(feature_s)
                    loss = criterion_cel(F.softmax(output_s), target_s)
                    loss.backward()
                    self.optimizer_f.step()
                    self.optimizer_c.step()
                    self.optimizer_f.zero_grad()
                    self.optimizer_c.zero_grad()
    def score(self, test_X, test_Y):
        pred_y = []
        correct = 0
        count = 0
        for i in range(int(len(test_X)/32)):
            data_s = test_X[i*32:(i+1)*32]
            target_s = test_Y[i*32:(i+1)*32]
            data_s, target_s = torch.from_numpy(data_s).float().cuda(), torch.from_numpy(target_s).cuda(non_blocking=True)
            feature_s = self.model_f(data_s)
            output = self.model_c(feature_s)
            pred = output.max(1, keepdim=True)[1]
            for i in range(len(pred)):
                pred_y.append(pred[i].item())
            correct += pred.eq(target_s.view_as(pred)).sum().item()
            count += len(target_s)
        return correct*1.0/count
    def predict(self, test_X):
        if True:
            data_s = test_X
            data_s = torch.from_numpy(data_s).float().cuda()
            feature_s = self.model_f(data_s)
            output = self.model_c(feature_s)
        return output[:,0].cpu().detach().numpy()
    def predict_proba(self, test_X, n_class = 2):
        prob = np.zeros((len(test_X), n_class))
        temp = self.predict(test_X)
        for i in range(0, len(test_X)):
            prob[i,0]=temp[i]
            prob[i,1]=1-temp[i]
        return prob


    

classification_method = ffn_classifier
    
if True:
    
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
    
    scaler = skp.StandardScaler()
    
    
    
    train_X = np.concatenate(batch_X[:train_batch_num])
    train_Y = np.concatenate(batch_Y[:train_batch_num])
    
    #normalize features
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    
    first_undrift_index = train_batch_num+1
    
    clf2 = classification_method(n_estimators=50)
    clf2.fit(train_X,train_Y)
    
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
    q1_drift, q2_drift, q3_drift, qAE_drift, qspn_drift, qFS_drift = False, False, False, False, False, False
    first_training_index = sys.maxsize
    drift_1, drift_2, drift_3, drift_AE, drift_spn, drift_FS = [],[],[],[],[],[]
    
    drift_list = []
    prev_train_X = train_X
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    
    p1_p2_weights = [0.01, 0.99]
    window_size, alpha, beta = 3, 0.1, 0.1
    keep_last = 10
    
    if True:
        feature_size = train_X.shape[1]
        input_img = Input(shape=(feature_size,))
        #encoded = Dense(6, activation='relu', activity_regularizer=regularizers.l2(10e-5))(input_img)
        encoded = Dense(3, activation='relu')(input_img)
        #encoded = Dense(6, activation='relu', activity_regularizer=regularizers.l2(10e-5))(encoded)
        decoded = Dense(feature_size)(encoded)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(train_X, train_X,
                        epochs=20,
                        batch_size=16,
                        shuffle=True)
    
        AE_tr_err = autoencoder.evaluate(train_X, train_X)
    
    kfac_optim = KFACOptimizer(nn.Sequential(clf2.model_f,clf2.model_c),
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
    
    #temp_loss = []
    def train_spn(model_f, spn, train_x):
        model_f.eval()
        spn.train()
        if True:
            for t in range(200):
                for i in range(int(len(train_x)/32)):
                    data = train_x[i*32:(i+1)*32]
                    feature = model_f(torch.from_numpy(data).float().cuda())
                    output = spn(feature)
                    loss = -1 * output.mean()
                    loss.backward()
                    optimizer_spn.step()
                    spn.apply(clipper)
                    optimizer_spn.zero_grad()
                    #temp_loss.append(loss.item())
    train_spn(clf2.model_f, spn, train_X)
    
    def test_spn(model_f, spn, test_x):
        model_f.eval()
        spn.eval()
        test_x = torch.from_numpy(test_x).float().cuda()
        feature = model_f(test_x)
        output = spn(feature)
        loss = -1 * output.mean()
        return loss.item()

    
    for i in range(train_batch_num + label_lag, n_batch):
        #i inclusive
        data_X = batch_X[first_undrift_index:i+1]
        data_Y = batch_Y[first_undrift_index:i+1]
        #normalize features
        data_X = [scaler.transform(datax) for datax in data_X]
    
        q1_list.append(Q1(data_X, data_Y, label_lag, train_X, train_Y, clf2,window_size=window_size,alpha=alpha, beta=beta, p1_p2_weights=p1_p2_weights))
        q2_list.append(Q3(data_X, data_Y, label_lag, train_X, train_Y, clf2))
        q3_list.append(Q4(data_X, data_Y, label_lag, train_X, train_Y, clf2))
        qAE_list.append(math.tanh(autoencoder.evaluate(scaler.transform(batch_X[i]), scaler.transform(batch_X[i]))/AE_tr_err/2))
        qspn_list.append(np.log(test_spn(clf2.model_f, spn, batch_X[i])))

        kfac_optim.zero_grad()
            
        feat = clf2.model_f(torch.from_numpy(scaler.transform(batch_X[i])).float().cuda())
        outputs = clf2.model_c(feat)
        loss = criterion_cel(F.softmax(outputs), torch.from_numpy(batch_Y[i]).cuda())

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

        prequential_acc.append(clf2.score(scaler.transform(batch_X[i]),batch_Y[i]))
    
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
                if warning_index_3!=[]:
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
                warning_index_AE.append(i)            
        if q1_drift*3+q2_drift*1+q3_drift*1+qAE_drift*1+qspn_drift*1+qFS_drift*1>=3:
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            print('retrain starting dataset index '+ str(first_training_index))
            clf2=classification_method(n_estimators=50)
            previous_train_X = train_X
            if q1_drift==True:
                first_training_index = keep_last_consecutive(warning_index)[0]
            train_X, train_Y = np.concatenate([batch_X[j] for j in np.arange(first_training_index,i)[-keep_last:]]), np.concatenate([batch_Y[j] for j in np.arange(first_training_index,i)[-keep_last:]])
    
                
            #normalize features
            scaler = skp.StandardScaler()
            scaler.fit(train_X)
            train_X = scaler.transform(train_X)

            clf2.fit(train_X, train_Y)
            first_undrift_index = i-np.max([i-first_training_index,label_lag])
            warning_index = []
            warning_index_2 = []
            warning_index_3 = []
            warning_index_AE = []
            warning_index_spn = []
            warning_index_FS = []
            q1_drift, q2_drift, q3_drift, qAE_drift, qspn_drift, qFS_drift = False, False, False, False, False, False

            if True:
                feature_size = train_X.shape[1]
                input_img = Input(shape=(feature_size,))
                #encoded = Dense(6, activation='relu', activity_regularizer=regularizers.l2(10e-5))(input_img)
                encoded = Dense(3, activation='relu')(input_img)
                #encoded = Dense(6, activation='relu', activity_regularizer=regularizers.l2(10e-5))(encoded)
                decoded = Dense(feature_size)(encoded)
                autoencoder = Model(input_img, decoded)
                autoencoder.compile(optimizer='adam', loss='mse')
                autoencoder.fit(train_X, train_X,
                                epochs=20,
                                batch_size=16,
                                shuffle=True)
                AE_tr_err = autoencoder.evaluate(train_X, train_X)
                
            gauss = Normal(multiplicity=5, in_features=50)
            prod1 = Product(in_features=50, cardinality=5)
            sum1 = Sum(in_features=10, in_channels=5, out_channels=1)
            prod2 = Product(in_features=10, cardinality=10)
            spn = nn.Sequential(gauss, prod1, sum1, prod2).cuda()
            clipper = DistributionClipper()
            optimizer_spn = torch.optim.Adam(spn.parameters(), lr=0.001)
            optimizer_spn.zero_grad()
            train_spn(clf2.model_f, spn, train_X)

            kfac_optim = KFACOptimizer(nn.Sequential(clf2.model_f,clf2.model_c),
                                      lr=0.01,
                                      momentum=0.9,
                                      stat_decay=0.95,
                                      damping=1e-3,
                                      kl_clip=1e-2,
                                      weight_decay=3e-3,
                                      TCov=10,
                                      TInv=100)            
            
            retraining_time += (time.time() - start_time)
            total_retraining_samples += train_X.shape[0]
            first_training_index = sys.maxsize

print('Mean accuracy:')
print(np.mean(prequential_acc))
print('Total retraining time:')
print(retraining_time)
