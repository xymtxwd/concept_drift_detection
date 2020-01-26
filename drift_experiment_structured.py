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
parser = argparse.ArgumentParser(description='Online Concept Drift Detection')
parser.add_argument('--model', choices=['ours', 'ourslinear', 'ddm', 'ph','adwin','ewma'], default='ours',
                    help='type of model')
parser.add_argument('--classifier', choices=['lgbm', 'lstm', 'xgb','rf'], default='lgbm',
                    help='type of classifier')
parser.add_argument('--dataset', choices=['elec', 'sea', 'rbf','hyperplane','weather','weather2'], default='sea',
                    help='type of dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--lstm-ae', default=1, type=int, 
                    help='use lstm as autoencoder (default: True)')
args = parser.parse_args()

np.random.seed(0)

print('Load data')

dataset = args.dataset
if dataset == 'elec':
    df = pd.read_csv("data/elecNormNew.csv")
    df['class'] = df['class'].map({'UP': 0, 'DOWN': 1})
    L = 8
    labels = df.columns.values.tolist()[L:]
    data = df.values
    T = len(data)
    Y = data[:, L:]
    X = data[:, 0:L]
if dataset == 'sea':
    dfX = pd.read_csv("data/sea/SEA_training_data.csv", header=None)
    dfY = pd.read_csv("data/sea/SEA_training_class.csv", header=None)
    X = dfX.values
    Y = np.squeeze(dfY.values)
    T = len(X)
if dataset == 'hyperplane':
    X = np.loadtxt("data/hyperplane/rotatingHyperplane.txt")
    dfY = pd.read_csv("data/hyperplane/rotatingHyperplane.labels", header=None)
    Y = np.squeeze(dfY.values)
    T = len(X)
if dataset == 'sine2':
    stream = SineGenerator(classification_function = 2, random_state = 112, balance_classes = False, has_noise = True)
    stream.prepare_for_use()
    X, sin_y = stream.next_sample(100000)
    Y = stream.gen_drift(sin_y.astype(int), 5)
    T = len(X)
if dataset == 'rbf':
    X = np.loadtxt("data/rbf/interchangingRBF.data")
    Y = np.loadtxt("data/rbf/interchangingRBF.labels")
    T = len(X)
if dataset == 'weather':
    X = pd.read_csv("data/weather/NEweather_data.csv", header=None).values
    Y = np.squeeze(pd.read_csv("data/weather/NEweather_class.csv", header=None).values)
    T = len(X)
if dataset == 'poker':
    X = np.loadtxt("data/poker/poker.data")
    Y = np.loadtxt("data/poker/poker.labels")
    X = X[:15000]
    Y = Y[:15000]
    T = len(X)
if dataset == 'stock':
    data= np.loadtxt(open('C:/Users/Yiming/Desktop/LidaFiles/returnsForDiego.csv','rb'),delimiter=',')
    data = data[:,5]
    time_step = 8
    L = len(data)
    bs = int((L-time_step)/time_step)
    bs = min(bs,50000)
    X = np.zeros((bs,time_step))
    Y = np.zeros(bs)
    for i in range(0,bs):
        X[i,:] = data[i*time_step:(i+1)*time_step]
        # 0 down, 1 up
        Y[i] = 1 if data[(i+1)*time_step]>data[(i+1)*time_step-1] else 0
    T = len(X)

if dataset == 'weather2':
    tmp = pd.read_csv('C:/Users/Yiming/Desktop/backup_919/drift_detection-master/NEW-DATA-1.T15.txt', header=None)
    tmp2 = pd.read_csv('C:/Users/Yiming/Desktop/backup_919/drift_detection-master/NEW-DATA-2.T15.txt', header=None)
    tmp = pd.concat([tmp,tmp2])
    tmp = tmp.values

    X = np.zeros((4139, 21))
    for i,item in enumerate(tmp):
        if not item[0].startswith('#'):
            temp = item[0].split(' ')[2:-1]
            for j in range(21):
                X[i,j] = np.float(temp[j])
    
    X = np.delete(X,[0, 2765],0)
    Y = X[:, -2]
    X = np.delete(X, 19, 1)
    labels = []
    for i in range(1, len(Y)-1):
        if Y[i]>Y[i-1]:
            labels.append(1)
        else:
            labels.append(0)
    X = X[1:-1,:]
    Y = np.array(labels)
    T = len(X)

    
seq_data = False
use_lstm_as_ae = args.lstm_ae
if seq_data:
    pre = 5
    fea_num = X.shape[1]
    #new_X = X[:T-T%pre].reshape(((T-T%pre)/pre, fea_num*pre))
    #new_X = np.zeros(fea_num*pre)
    new_X = np.zeros((T-pre,fea_num*pre))
    new_Y = np.zeros(T-pre)
    for i in range(0,T-pre):
        new_X[i,:] = X[i:i+pre,:].reshape((fea_num*pre))
        new_Y[i] = Y[i+pre-1]
    X = new_X
    Y = new_Y
    T = len(X)


class lstm_classifier:
    def __init__(self, n_estimators=20):
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(5, fea_num), return_sequences=False))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    def fit(self, train_X, train_Y):
        return self.model.fit(train_X.reshape((len(train_X),5,fea_num)), train_Y, epochs=30, batch_size=16)
    def score(self, test_X, test_Y):
        return self.model.evaluate(test_X.reshape((len(test_X),5,fea_num)),test_Y)[1]
    def predict(self, test_X):
        return np.round(self.model.predict(test_X.reshape((len(test_X),5,fea_num))))
    def predict_proba(self, test_X, n_class = 2):
        prob = np.zeros((len(test_X), n_class))
        temp = self.model.predict(test_X.reshape((len(test_X),5,fea_num)))
        for i in range(0, len(test_X)):
            prob[i,0]=temp[i]
            prob[i,1]=1-temp[i]
        return prob
    
if args.classifier=='lstm':
    classification_method = lstm_classifier
if args.classifier=='lgbm':
    classification_method = lgb.LGBMClassifier
if args.classifier=='rf':
    classification_method = RandomForestClassifier
if args.classifier=='xgb':
    classification_method = xgb.XGBClassifier
    
batch_size = args.batch_size
n_batch = int((T-T%batch_size)/batch_size)
batch_Y = np.split(Y[:T-T%batch_size],n_batch)
batch_X = np.split(X[:T-T%batch_size],n_batch)
initial_batches = 50
label_lag = 3

### DDM in batch
if args.model=='ddm':
    train_X = X[:initial_batches*batch_size]
    train_Y = Y[:initial_batches*batch_size]
    first_undrift_index = initial_batches+1
    
    clf2 = classification_method(n_estimators=20)
    clf2.fit(train_X,train_Y)
    
    dd = DDM()
    warning_index = []
    drift_list = []
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    
    ret_ind = []
    
    for i in range(initial_batches + label_lag+50, n_batch):
        
        prequential_acc.append(clf2.score(batch_X[i],batch_Y[i]))
        if dd.set_input(1-clf2.score(batch_X[i-3],batch_Y[i-3])):
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            warning_index.append(i)
            print('retrain using dataset index '+ str(keep_last_consecutive(warning_index)))
            clf2=classification_method(n_estimators=20)
            ret_ind.append(keep_last_consecutive(warning_index))
            train_X, train_Y = np.concatenate([batch_X[j] for j in keep_last_consecutive(warning_index)]), np.concatenate([batch_Y[j] for j in keep_last_consecutive(warning_index)])
            clf2.fit(train_X, train_Y)
            warning_index=[]
            retraining_time += (time.time() - start_time)
            total_retraining_samples += train_X.shape[0]
        if dd.is_warning_zone:
            warning_index.append(i)
    




### PH in batch
if args.model=='ph':
    
    train_X = X[:initial_batches*batch_size]
    train_Y = Y[:initial_batches*batch_size]
    first_undrift_index = initial_batches+1
    
    clf2 = classification_method(n_estimators=20)
    clf2.fit(train_X,train_Y)
    
    ph = PageHinkley(threshold=0.3,delta=0.005)
    warning_index = []
    drift_list = []
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    for i in range(initial_batches + label_lag+50, n_batch):
        ph.add_element(1-clf2.score(batch_X[i-3],batch_Y[i-3]))
        prequential_acc.append(clf2.score(batch_X[i],batch_Y[i]))
        if ph.in_concept_change:
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            warning_index.append(i)
            print('retrain using dataset index '+ str(keep_last_consecutive(warning_index)))
            clf2=classification_method(n_estimators=20)
            train_X, train_Y = np.concatenate([batch_X[j] for j in keep_last_consecutive(warning_index)]), np.concatenate([batch_Y[j] for j in keep_last_consecutive(warning_index)])
            clf2.fit(train_X, train_Y)
            warning_index=[]
            retraining_time += (time.time() - start_time)
            total_retraining_samples += train_X.shape[0]
        if ph.in_warning_zone:
            warning_index.append(i)

### ADWIN in batch
if args.model=='adwin':
    
    train_X = X[:initial_batches*batch_size]
    train_Y = Y[:initial_batches*batch_size]
    first_undrift_index = initial_batches+1
    
    clf2 = classification_method(n_estimators=20)
    clf2.fit(train_X,train_Y)
    
    ad = Adwin(delta=1)
    drift_list = []
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    for i in range(initial_batches + label_lag+50, n_batch):
        
        prequential_acc.append(clf2.score(batch_X[i],batch_Y[i]))
        if ad.set_input(1-clf2.score(batch_X[i-3],batch_Y[i-3])):
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            print('retrain using dataset index '+ str(i))
            clf2=classification_method(n_estimators=20)
            train_X, train_Y = batch_X[i],batch_Y[i]
            clf2.fit(train_X, train_Y)
            retraining_time += (time.time() - start_time)
            total_retraining_samples += train_X.shape[0]

### EWMA
if args.model=='ewma':
    
    train_X = X[:initial_batches*batch_size]
    train_Y = Y[:initial_batches*batch_size]
    first_undrift_index = initial_batches+1
    
    clf2 = classification_method(n_estimators=20)
    clf2.fit(train_X,train_Y)
    
    ewma = Ewma(alpha=0.1, coefficient=3)
    drift_list = []
    prequential_acc = []
    error_list = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    for i in range(initial_batches + label_lag+50, n_batch):
        error_list.append(1-clf2.score(batch_X[i-label_lag],batch_Y[i-label_lag]))
        prequential_acc.append(clf2.score(batch_X[i],batch_Y[i]))
        if len(error_list)>1 and ewma.predict(error_list)==0:
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            print('retrain using dataset index '+ str(i))
            clf2=classification_method(n_estimators=20)
            train_X, train_Y = batch_X[i],batch_Y[i]
            clf2.fit(train_X, train_Y)
            error_list = []
            retraining_time += (time.time() - start_time)
            total_retraining_samples += train_X.shape[0]


### retrain every time step
if args.model=='retrain_everystep':
    train_X = X[:initial_batches*batch_size]
    train_Y = Y[:initial_batches*batch_size]
    first_undrift_index = initial_batches+1
    
    clf2 = classification_method(n_estimators=20)
    clf2.fit(train_X,train_Y)
    
    prequential_acc = []
    retraining_time = 0
    
    for i in range(initial_batches + label_lag+50, n_batch):
        
        prequential_acc.append(clf2.score(batch_X[i],batch_Y[i]))
        if True:
            start_time = time.time()
            clf2=classification_method(n_estimators=20)
            train_X, train_Y = np.concatenate([batch_X[j] for j in np.arange(i-20-3, i-3)]), np.concatenate([batch_Y[j] for j in np.arange(i-20-3, i-3)])
            clf2.fit(train_X, train_Y)
            retraining_time += (time.time() - start_time)

### our model with linear weights for scores
if args.model=='ourslinear':

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras import regularizers
    import math
    
    batch_size = 64
    n_batch = int((T-T%batch_size)/batch_size)
    batch_Y = np.split(Y[:T-T%batch_size],n_batch)
    batch_X = np.split(X[:T-T%batch_size],n_batch)
    initial_batches = 50
    label_lag = 3
    
    
    train_X = X[:initial_batches*batch_size]
    train_Y = Y[:initial_batches*batch_size]
    first_undrift_index = initial_batches+1
    
    clf2 = classification_method(n_estimators=20)
    clf2.fit(train_X,train_Y)
    
    q1_list,q2_list,q3_list,q4_list,q_list = [],[],[],[],[]
        
    dd = DDM(3,2)
    warning_index = []
    
    target_drift_list = []
    for i in range(1,67):
        target_drift_list.append(math.ceil(i*3000/batch_size))
        
    def avg_wait_time(target_drift_list,drift_list):
        count = 0
        for i,d in enumerate(target_drift_list):
            count += (drift_list[i] - target_drift_list[i])
            print(drift_list[i] - target_drift_list[i])
        return count/len(target_drift_list)
            
    
        
    drift_list = []
    #model_list = []
    prev_train_X = train_X
    prequential_acc = []
    lmd = 0.50
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    Q4(train_X[:int(len(train_X)/2)], None, label_lag, train_X[int(len(train_X)/2):], None, clf2, for_type_calc=True)
    
    if Q4(train_X[:int(len(train_X)/2)], None, label_lag, train_X[int(len(train_X)/2):], None, clf2, for_type_calc=True)>lmd:
        weights = [0.10, 0.35, 0.35, 0.20]
        p1_p2_weights = [0.8,0.2]
        window_size, alpha, beta = 2, 0.1, 0.1
        keep_last  = 10
        print('ENTER COV SHIFT MODE')
        lmd -= 0.1
        #dd = DDM(2.0,1.6)
    else:
        weights = [0.935, 0.002, 0.002, 0.001, 0.06]
        p1_p2_weights = [0.1, 0.9]
        window_size, alpha, beta = 5, 0.05, 0.05
        keep_last = 10
        print('ENTER CONCEPT SHIFT MODE')
        lmd += 0.5
        
    shift_type_flag = False
    
    feature_size = train_X.shape[1]
    input_img = Input(shape=(feature_size,))
    #encoded = Dense(6, activation='relu', activity_regularizer=regularizers.l2(10e-5))(input_img)
    encoded = Dense(3, activation='relu')(input_img)
    #encoded = Dense(6, activation='relu', activity_regularizer=regularizers.l2(10e-5))(encoded)
    decoded = Dense(feature_size)(encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(train_X, train_X,
                    epochs=30,
                    batch_size=16,
                    shuffle=True)
    AE_tr_err = autoencoder.evaluate(train_X, train_X)
    for i in range(initial_batches + label_lag+50, n_batch):
        #i inclusive
        if shift_type_flag:
            print(Q4(previous_train_X, data_Y, label_lag, train_X, train_Y, clf2, for_type_calc=True))
            if Q4(previous_train_X, data_Y, label_lag, train_X, train_Y, clf2, for_type_calc=True)>lmd:
                weights = [0.10, 0.35, 0.35, 0.20]
                p1_p2_weights = [0.8,0.2]
                window_size, alpha, beta = 2, 0.1, 0.1
                keep_last  = 10
                print('ENTER COV SHIFT MODE')
                shift_type_flag = False
            else:
                weights = [0.935, 0.002, 0.002, 0.001, 0.06]
                p1_p2_weights = [0.1, 0.9]
                window_size, alpha, beta = 5, 0.05, 0.05
                keep_last = 10
                print('ENTER CONCEPT SHIFT MODE')
                shift_type_flag = False
                
        data_X = batch_X[first_undrift_index:i+1]
        data_Y = batch_Y[first_undrift_index:i+1]
        
        q1_list.append(Q1(data_X, data_Y, label_lag, train_X, train_Y, clf2,window_size=window_size,alpha=alpha, beta=beta, p1_p2_weights=p1_p2_weights))
        q2_list.append(Q2(data_X, data_Y, label_lag, train_X, train_Y, clf2))
        q3_list.append(Q3(data_X, data_Y, label_lag, train_X, train_Y, clf2))
        q4_list.append(Q4(data_X, data_Y, label_lag, train_X, train_Y, clf2))
        q_list.append(q1_list[-1]*weights[0] + q2_list[-1]*weights[1] + q3_list[-1]*weights[2] + q4_list[-1]*weights[3] + math.log(autoencoder.evaluate(batch_X[i], batch_X[i])/AE_tr_err+1)*weights[4])
        prequential_acc.append(clf2.score(batch_X[i],batch_Y[i]))
    
                    
                        
        if dd.set_input(q_list[-1]): 
            start_time = time.time()
    
            print('CHANGE DETECTED at '+str(i) + ' with q = ' + str(q_list[-1]))
            drift_list.append(i)
            warning_index.append(i)
            print('retrain using dataset index '+ str(keep_last_consecutive(warning_index)))
            clf2=classification_method(n_estimators=50)
            previous_train_X = train_X
            keep_last = label_lag if len(keep_last_consecutive(warning_index))>100 else keep_last
            train_X, train_Y = np.concatenate([batch_X[j] for j in keep_last_consecutive(warning_index)[-keep_last:]]), np.concatenate([batch_Y[j] for j in keep_last_consecutive(warning_index)[-keep_last:]])
    
            #more training sets
            temp_clf = classification_method(n_estimators=20)
            temp_clf.fit(train_X, train_Y)
            scores = cross_val_score(clf2, train_X, np.squeeze(train_Y), cv=5)
            threshold = scores.mean()+2.33*scores.std()
            
            num = 0
            scores = np.zeros(keep_last_consecutive(warning_index)[0])
            for p in range(0,keep_last_consecutive(warning_index)[0]):
                scores[p] = temp_clf.score(batch_X[p],batch_Y[p])
    
            maxs = scores.argsort()[-int(len(scores)*0.1):][::-2]
            
            for p in maxs:
                if temp_clf.score(batch_X[p],batch_Y[p])>threshold and p>int(len(scores)/2):
                    train_X = np.concatenate([train_X,batch_X[p]])
                    train_Y = np.concatenate([train_Y,batch_Y[p]])
                    num+=1
            
            print(str(num)+' previous batches appended')
    
    
            clf2.fit(train_X, train_Y)
            first_undrift_index = i-np.max([len(keep_last_consecutive(warning_index)),label_lag])
            warning_index=[]
            shift_type_flag = True
            #if num!= 0: print(0/0)
            #model_list.append(deepcopy(clf2))
            
            input_img = Input(shape=(feature_size,))
            #encoded = Dense(6, activation='relu', activity_regularizer=regularizers.l2(10e-5))(input_img)
            encoded = Dense(3, activation='relu')(input_img)
            #encoded = Dense(6, activation='relu', activity_regularizer=regularizers.l2(10e-5))(encoded)
            decoded = Dense(feature_size)(encoded)
            autoencoder = Model(input_img, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(train_X, train_X,
                            epochs=30,
                            batch_size=16,
                            shuffle=True)
            AE_tr_err = autoencoder.evaluate(train_X, train_X)
            retraining_time += (time.time() - start_time)
            total_retraining_samples += train_X.shape[0]
        if dd.is_warning_zone:
            warning_index.append(i)


### AE - ensemble normalized
if args.model=='ours':
    
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
    
    batch_size = 64
    n_batch = int((T-T%batch_size)/batch_size)
    batch_Y = np.split(Y[:T-T%batch_size],n_batch)
    batch_X = np.split(X[:T-T%batch_size],n_batch)
    initial_batches = 50
    label_lag = 3
    
    
    train_X = X[:initial_batches*batch_size]
    train_Y = Y[:initial_batches*batch_size]
    
    #normalize features
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    
    first_undrift_index = initial_batches+1
    
    clf2 = classification_method(n_estimators=20)
    clf2.fit(train_X,train_Y)
    
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
    prev_train_X = train_X
    prequential_acc = []
    retraining_time = 0
    total_retraining_samples = 0
    total_added_samples = 0
    
    p1_p2_weights = [0.01, 0.99]
    window_size, alpha, beta = 3, 0.1, 0.1
    keep_last = 10
    
    if use_lstm_as_ae:
        # define input sequence
        # reshape input into [samples, timesteps, features]
        n_in = 8
        sequence = train_X.reshape((-1, n_in, X.shape[1]))
        # define model
        autoencoder = Sequential()
        autoencoder.add(LSTM(100, activation='relu', input_shape=(n_in,X.shape[1])))
        autoencoder.add(RepeatVector(n_in))
        autoencoder.add(LSTM(100, activation='relu', return_sequences=True))
        autoencoder.add(TimeDistributed(Dense(X.shape[1])))
        autoencoder.compile(optimizer='adam', loss='mse')
        # fit model
        autoencoder.fit(sequence, sequence, epochs=50, verbose=1)
        AE_tr_err = autoencoder.evaluate(sequence, sequence)
    
    else:
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
    
    for i in range(initial_batches + label_lag+50, n_batch):
        #i inclusive
        data_X = batch_X[first_undrift_index:i+1]
        data_Y = batch_Y[first_undrift_index:i+1]
        #normalize features
        data_X = [scaler.transform(datax) for datax in data_X]
    
        q1_list.append(Q1(data_X, data_Y, label_lag, train_X, train_Y, clf2,window_size=window_size,alpha=alpha, beta=beta, p1_p2_weights=p1_p2_weights)*0.99+Q2(data_X, data_Y, label_lag, train_X, train_Y, clf2)*0.01)
        q2_list.append(Q3(data_X, data_Y, label_lag, train_X, train_Y, clf2))
        q3_list.append(Q4(data_X, data_Y, label_lag, train_X, train_Y, clf2))
        if use_lstm_as_ae:
            qAE_list.append(math.tanh(autoencoder.evaluate(scaler.transform(batch_X[i]).reshape((-1,n_in,X.shape[1])), scaler.transform(batch_X[i]).reshape((-1,n_in,X.shape[1])))/AE_tr_err/2))
        else:
            qAE_list.append(math.tanh(autoencoder.evaluate(scaler.transform(batch_X[i]), scaler.transform(batch_X[i]))/AE_tr_err/2))
    
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
    
            
        if q1_drift*3+q2_drift*1+q3_drift*1+qAE_drift*2>=3:
            start_time = time.time()
            print('CHANGE DETECTED at '+str(i))
            drift_list.append(i)
            print('retrain starting dataset index '+ str(first_training_index))
            clf2=classification_method(n_estimators=50)
            previous_train_X = train_X
            #keep_last = label_lag if len(keep_last_consecutive(warning_index))>100 else keep_last
            #keep_last = 100 if i-first_training_index>100 else sys.maxsize
            if q1_drift==True:
                first_training_index = keep_last_consecutive(warning_index)[0]
            train_X, train_Y = np.concatenate([batch_X[j] for j in np.arange(first_training_index,i)[-keep_last:]]), np.concatenate([batch_Y[j] for j in np.arange(first_training_index,i)[-keep_last:]])
    
                
            #normalize features
            scaler = skp.StandardScaler()
            scaler.fit(train_X)
            train_X = scaler.transform(train_X)
    
            if args.classifier!='lstm':
                #more training sets
                temp_clf = classification_method(n_estimators=20)
                temp_clf.fit(train_X, train_Y)
                scores = cross_val_score(clf2, train_X, np.squeeze(train_Y), cv=5)
                threshold = scores.mean()+2.33*scores.std()
                
                num = 0
                scores = np.zeros(first_training_index)
                for p in range(0,first_training_index):
                    scores[p] = temp_clf.score(scaler.transform(batch_X[p]),batch_Y[p])
        
                maxs = scores.argsort()[-int(len(scores)*0.1):][::-2]
                
                for p in maxs:
                    if temp_clf.score(scaler.transform(batch_X[p]),batch_Y[p])>threshold and p>int(len(scores)/2):
                        train_X = np.concatenate([train_X,scaler.transform(batch_X[p])])
                        train_Y = np.concatenate([train_Y,batch_Y[p]])
                        num+=1
                
                print(str(num)+' previous batches appended')
            else:
                #more training sets
                from sklearn.model_selection import StratifiedKFold
                kfold = StratifiedKFold(n_splits=5, shuffle=True)
                cvscores = []
                temp_clf = classification_method(n_estimators=50)
                temp_clf.fit(train_X, train_Y)
                for train, test in kfold.split(train_X, train_Y):
                    	temp_model = classification_method(n_estimators=50)
                    	temp_model.fit(train_X[train], train_Y[train])
                    	scores = temp_model.score(train_X[test], train_Y[test])
                    	cvscores.append(scores)
                threshold = np.mean(cvscores) + 2.33*np.std(scores)
                
                num = 0
                scores = np.zeros(first_training_index)
                for p in range(0,first_training_index):
                    scores[p] = temp_clf.score(scaler.transform(batch_X[p]),batch_Y[p])
        
                maxs = scores.argsort()[-int(len(scores)*0.1):][::-2]
                
                for p in maxs:
                    if temp_clf.score(scaler.transform(batch_X[p]),batch_Y[p])>threshold and p>int(len(scores)/2):
                        train_X = np.concatenate([train_X,scaler.transform(batch_X[p])])
                        train_Y = np.concatenate([train_Y,batch_Y[p]])
                        num+=1
                
                print(str(num)+' previous batches appended')
    
            clf2.fit(train_X, train_Y)
            first_undrift_index = i-np.max([i-first_training_index,label_lag])
            
            warning_index = []
            warning_index_2 = []
            warning_index_3 = []
            warning_index_AE = []
            q1_drift, q2_drift, q3_drift, qAE_drift = False, False, False, False
        
            if use_lstm_as_ae:
                # define input sequence
                # reshape input into [samples, timesteps, features]
                n_in = 8
                sequence = train_X.reshape((-1, n_in, X.shape[1]))
                # define model
                autoencoder = Sequential()
                autoencoder.add(LSTM(100, activation='relu', input_shape=(n_in,X.shape[1])))
                autoencoder.add(RepeatVector(n_in))
                autoencoder.add(LSTM(100, activation='relu', return_sequences=True))
                autoencoder.add(TimeDistributed(Dense(X.shape[1])))
                autoencoder.compile(optimizer='adam', loss='mse')
                # fit model
                autoencoder.fit(sequence, sequence, epochs=50, verbose=1)
                AE_tr_err = autoencoder.evaluate(sequence, sequence)
            
            else:
                
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
    
            
            retraining_time += (time.time() - start_time)
            total_retraining_samples += train_X.shape[0]
            first_training_index = sys.maxsize

print('Mean accuracy:')
print(np.mean(prequential_acc))
print('Total retraining time:')
print(retraining_time)
