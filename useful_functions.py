import numpy as np
import scipy as sp
from sklearn.ensemble import IsolationForest
np.random.seed(0)
from sklearn import svm
import math
from scipy.stats import norm
import math
import torch.nn as nn
import torch.nn.functional as F


class SineGenerator:
    """ SineGenerator
    This generator is an implementation of the dara stream with abrupt
    concept drift, as described in Gama, Joao, et al [1]_.
    It generates up to 4 relevant numerical attributes, that vary from 0 to 1,
    where only 2 of them are relevant to the classification task and the other
    2 are added by request of the user. A classification function is chosen
    among four possible ones:
    0. SINE1. Abrupt concept drift, noise-free examples. It has two relevant
       attributes. Each attributes has values uniformly distributed in [0; 1].
       In the first context all points below the curve :math:`y = sin(x)` are
       classified as positive.
    1. Reversed SINIE1. The reversed classification of SINE1.
    2. SINE2. The same two relevant attributes. The classification function
       is :math:`y < 0.5 + 0.3 sin(3 \pi  x)`.
    3. Reversed SINIE1. The reversed classification of SINE2.
    The abrupt drift is generated by changing the classification function,
    thus changing the threshold.
    Two important features are the possibility to balance classes, which
    means the class distribution will tend to a uniform one, and the possibility
    to add noise, which will, add two non relevant attributes.
    Parameters
    ----------
    classification_function: int (Default: 0)
        Which of the four classification functions to use for the generation.
        From 0 to 3.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    balance_classes: bool (Default: False)
        Whether to balance classes or not. If balanced, the class
        distribution will converge to a uniform distribution.
    has_noise: bool (Default: False)
        Adds 2 non relevant features to the stream.
    References
    ----------
    .. [1] Gama, Joao, et al.'s 'Learning with drift
       detection.' Advances in artificial intelligence–SBIA 2004. Springer Berlin
       Heidelberg, 2004. 286-295."
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.sine_generator import SineGenerator
    >>> # Setting up the stream
    >>> stream = SineGenerator(classification_function = 2, random_state = 112, balance_classes = False,
    ... has_noise = True)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[0.37505713, 0.64030462, 0.95001658, 0.0756772 ]]), array([1.]))
    >>> stream.next_sample(10)
    (array([[0.77692966, 0.83274576, 0.05480574, 0.81767738],
       [0.88535146, 0.72234651, 0.00255603, 0.98119928],
       [0.34341985, 0.09475989, 0.39464259, 0.00494492],
       [0.73670683, 0.95580687, 0.82060937, 0.344983  ],
       [0.37854446, 0.78476361, 0.08623151, 0.54607394],
       [0.16222602, 0.29006973, 0.04500817, 0.33218776],
       [0.73653322, 0.83921149, 0.70936161, 0.18840112],
       [0.98566856, 0.38800331, 0.50315448, 0.76353033],
       [0.68373245, 0.72195738, 0.21415209, 0.76309258],
       [0.07521616, 0.6108907 , 0.42563042, 0.23435109]]), array([1., 0., 1., 0., 1., 1., 1., 0., 0., 1.]))
    >>> stream.n_remaining_samples()
    -1
    >>> stream.has_more_samples()
    True
    """
    _NUM_BASE_ATTRIBUTES = 2
    _TOTAL_ATTRIBUTES_INCLUDING_NOISE = 4

    def __init__(self, classification_function=0, random_state=None, balance_classes=False, has_noise=False):
        super().__init__()

        # Classification functions to use
        self._classification_functions = [self.classification_function_zero, self.classification_function_one,
                                         self.classification_function_two, self.classification_function_three]
        self.classification_function_idx = classification_function
        self._original_random_state = random_state
        self.has_noise = has_noise
        self.balance_classes = balance_classes
        self.n_num_features = self._NUM_BASE_ATTRIBUTES
        self.n_classes = 2
        self.n_targets = 1
        self.random_state = None
        self.next_class_should_be_zero = False
        self.name = "Sine Generator"

        self.__configure()

    def __configure(self):
        if self.has_noise:
            self.n_num_features = self._TOTAL_ATTRIBUTES_INCLUDING_NOISE
        self.n_features = self.n_num_features
        self.target_names = ["target_0"]
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_features)]
        self.target_values = [i for i in range(self.n_classes)]

    @property
    def classification_function_idx(self):
        """ Retrieve the index of the current classification function.
        Returns
        -------
        int
            index of the classification function [0,1,2,3]
        """
        return self._classification_function_idx

    @classification_function_idx.setter
    def classification_function_idx(self, classification_function_idx):
        """ Set the index of the current classification function.
        Parameters
        ----------
        classification_function_idx: int (0,1,2,3)
        """
        if classification_function_idx in range(4):
            self._classification_function_idx = classification_function_idx
        else:
            raise ValueError("classification_function_idx takes only these "
                             "values: 0, 1, 2, 3, and {} was "
                             "passed".format(classification_function_idx))

    @property
    def balance_classes(self):
        """ Retrieve the value of the option: Balance classes
        Returns
        -------
        Boolean
            True is the classes are balanced
        """
        return self._balance_classes

    @balance_classes.setter
    def balance_classes(self, balance_classes):
        """ Set the value of the option: Balance classes.
        Parameters
        ----------
        balance_classes: Boolean
        """
        if isinstance(balance_classes, bool):
            self._balance_classes = balance_classes
        else:
            raise ValueError("balance_classes should be boolean,"
                             " and {} was passed".format(balance_classes))

    @property
    def has_noise(self):
        """ Retrieve the value of the option: add noise.
        Returns
        -------
        Boolean
            True is the noise is added
        """
        return self._has_noise

    @has_noise.setter
    def has_noise(self, has_noise):
        """ Set the value of the option: add noise.
        Parameters
        ----------
        has_noise: Boolean
        """
        if isinstance(has_noise, bool):
            self._has_noise = has_noise
        else:
            raise ValueError("has_noise should be boolean, {} was passed".format(has_noise))

    def prepare_for_use(self):
        """
        Should be called before generating the samples.
        """

        self.random_state = np.random.RandomState(seed=1024)
        self.next_class_should_be_zero = False
        self.sample_idx = 0

    def next_sample(self, batch_size=1):
        """ next_sample
        The sample generation works as follows: The two attributes are
        generated with the random generator, initialized with the seed passed
        by the user. Then, the classification function decides whether to
        classify the instance as class 0 or class 1. The next step is to
        verify if the classes should be balanced, and if so, balance the
        classes. The last step is to add noise, if the has_noise is True.
        The generated sample will have 2 relevant features, and an additional
        two noise features if option chosen, and 1 label (it has one
        classification task).
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for
            the batch_size samples that were requested.
        """

        data = np.zeros([batch_size, self.n_features + 1])

        for j in range(batch_size):
            self.sample_idx += 1
            att1 = att2 = 0.0
            group = 0
            desired_class_found = False
            while not desired_class_found:
                att1 = self.random_state.rand()
                att2 = self.random_state.rand()
                group = self._classification_functions[self.classification_function_idx](att1, att2)

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self.next_class_should_be_zero and (group == 0)) or \
                            ((not self.next_class_should_be_zero) and (group == 1)):
                        desired_class_found = True
                        self.next_class_should_be_zero = not self.next_class_should_be_zero

            data[j, 0] = att1
            data[j, 1] = att2

            if self.has_noise:
                for i in range(self._NUM_BASE_ATTRIBUTES, self._TOTAL_ATTRIBUTES_INCLUDING_NOISE):
                    data[j, i] = self.random_state.rand()
                data[j, 4] = group
            else:
                data[j, 2] = group

        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = data[:, self.n_features:].flatten()

        return self.current_sample_x, self.current_sample_y

    def generate_drift(self):
        """
        Generate drift by switching the classification function randomly.
        """
        new_function = self.random_state.randint(4)
        while new_function == self.classification_function_idx:
            new_function = self.random_state.randint(4)
        self.classification_function_idx = new_function

    @staticmethod
    def classification_function_zero(att1, att2):
        """ classification_function_zero
        Decides the sample class label based on SINE1 function.
        Parameters
        ----------
        att1: float
            First numeric attribute.
        att2: float
            Second numeric attribute.
        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.
        """
        return 0 if (att1 >= np.sin(att2)) else 1

    @staticmethod
    def classification_function_one(att1, att2):
        """ classification_function_one
        Decides the sample class label based on reversed SINE1 function.
        Parameters
        ----------
        att1: float
            First numeric attribute.
        att2: float
            Second numeric attribute.
        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.
        """
        return 0 if (att1 < np.sin(att2)) else 1

    @staticmethod
    def classification_function_two(att1, att2):
        """ classification_function_two
        Decides the sample class label based on SINE2 function.
        Parameters
        ----------
        att1: float
            First numeric attribute.
        att2: float
            Second numeric attribute.
        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.
        """
        return 0 if (att1 >= 0.5 + 0.3 * np.sin(3 * np.pi * att2)) else 1

    @staticmethod
    def classification_function_three(att1, att2):
        """ classification_function_three
        Decides the sample class label based on reversed SINE2 function.
        Parameters
        ----------
        att1: float
            First numeric attribute.
        att2: float
            Second numeric attribute.
        Returns
        -------
        int
            Returns the sample class label, either 0 or 1.
        """
        return 0 if (att1 < 0.5 + 0.3 * np.sin(3 * np.pi * att2)) else 1

    def get_info(self):
        return 'SineGenerator: classification_function: ' + str(self.classification_function_idx) + \
               ' - random_state: ' + str(self._original_random_state) + \
               ' - balance_classes: ' + str(self.balance_classes) + \
               ' - has_noise: ' + str(self.has_noise)
    def gen_drift(self, labels, n_drifts):
        bs = int(len(labels)/n_drifts)
        drift = 1
        for i in range(0,n_drifts):
            if drift==1:
                labels[bs*i:bs*(i+1)] = np.ones(bs).astype(int) - labels[bs*i:bs*(i+1)]
            drift = 1 - drift
        return labels




def train_clf(model_f, model_c, train_xs, train_ys, train_xt, train_yt, drift_num, optimizer_f, optimizer_c):
    model_f.train()
    model_c.train()
    criterion_cel = nn.CrossEntropyLoss()

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
                #print(loss.item())
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

def Q1(data_X, data_Y, label_lag, train_X, train_Y, clf, alpha=0.5, beta=0.5, window_size = 10, p1_p2_weights=[0.3,0.7]):
    
    current_Y_available = False if label_lag>0 else True
    if current_Y_available: 
        return 1 - clf.score(data_X[-1],data_Y[-1])
    if len(data_X)==0: return 0.0#print('no inputs!')
    
    if len(data_X)<window_size+label_lag:
        window_size=len(data_X)-label_lag
    if window_size<=0: return 0.0#print('len(data_X)<label_lag!')
    
    p1_weights = np.zeros(window_size)
    p1_weights[-1] = beta
    for i in range(1,window_size):
        p1_weights[window_size-i-1] = beta*p1_weights[window_size-i]
    p1_weights = p1_weights/np.sum(p1_weights)
    
    kd_list = []
    for j in range(0, window_size):
        #print('j='+str(j))
        #print('window_size='+str(window_size))
        #print('len_data_X='+str(len(data_X)))
        temp_batch = data_X[j-window_size-label_lag]
        kd_list.append(sp.spatial.cKDTree(temp_batch, leafsize=100))
    count = 0
    for i, sample in enumerate(data_X[-1]):
        temp_distance = np.zeros(len(kd_list))
        temp_NN_index = []
        for j, kd in enumerate(kd_list):
            d, NN_index = kd.query(sample)
            temp_distance[j] = d*p1_weights[j]
            temp_NN_index.append(NN_index)
        min_id = np.argmin(temp_distance)
        Xnn,Ynn = data_X[min_id-window_size-label_lag][temp_NN_index[min_id]],data_Y[min_id-window_size-label_lag][temp_NN_index[min_id]]

        if clf.predict(sample.reshape(1, -1)) != clf.predict(Xnn.reshape(1, -1)):
            count += 1
        elif clf.predict(sample.reshape(1, -1)) != Ynn:
            count += 0.5
    p1 = count*1.0/len(data_X[-1])

    p2_weights = np.zeros(window_size)
    p2_weights[-1] = alpha
    for i in range(1,window_size):
        p2_weights[window_size-i-1] = alpha*p2_weights[window_size-i]
    p2_weights = p2_weights/np.sum(p2_weights)
    p2_temp = 0
    for i in range(0,window_size):
        p2_temp += p2_weights[i]*clf.score(data_X[i-window_size-label_lag],data_Y[i-window_size-label_lag])
    p2 = 1 - p2_temp#clf.score(data_X[-1-label_lag],data_Y[-1-label_lag])
    return p1_p2_weights[0]*p1+p1_p2_weights[1]*p2

def Q1u(data_X, data_Y, label_lag, model_f, clf, alpha=0.5, beta=0.5, window_size = 10, p1_p2_weights=[0.3,0.7]):
    current_Y_available = False if label_lag>0 else True
    if current_Y_available: 
        return 1 - nn_score(model_f, clf, [data_X[-1]], [data_Y[-1]], [], [], 0)
    if len(data_X)==0: return 0.0#print('no inputs!')
    if len(data_X)<window_size+label_lag:
        window_size=len(data_X)-label_lag
    if window_size<=0: return 0.0#print('len(data_X)<label_lag!')
    if False:
        p1_weights = np.zeros(window_size)
        p1_weights[-1] = beta
        for i in range(1,window_size):
            p1_weights[window_size-i-1] = beta*p1_weights[window_size-i]
        p1_weights = p1_weights/np.sum(p1_weights)
        kd_list = []
        for j in range(0, window_size):
            temp_batch = model_f(data_X[j-window_size-label_lag].cuda()).cpu().detach().numpy()
            kd_list.append(sp.spatial.cKDTree(temp_batch, leafsize=100))
        count = 0
        last_batch_embedded = model_f(data_X[-1].cuda()).cpu().detach().numpy()
        for i, sample in enumerate(last_batch_embedded):
            temp_distance = np.zeros(len(kd_list))
            temp_NN_index = []
            for j, kd in enumerate(kd_list):
                d, NN_index = kd.query(sample)
                temp_distance[j] = d*p1_weights[j]
                temp_NN_index.append(NN_index)
            min_id = np.argmin(temp_distance)
            Xnn,Ynn = data_X[min_id-window_size-label_lag][temp_NN_index[min_id]],data_Y[min_id-window_size-label_lag][temp_NN_index[min_id]]
            if clf.predict(sample.reshape(1, -1)) != clf.predict(Xnn.reshape(1, -1)):
                count += 1
            elif clf.predict(sample.reshape(1, -1)) != Ynn:
                count += 0.5
        p1 = count*1.0/len(data_X[-1])
    p2_weights = np.zeros(window_size)
    p2_weights[-1] = alpha
    for i in range(1,window_size):
        p2_weights[window_size-i-1] = alpha*p2_weights[window_size-i]
    p2_weights = p2_weights/np.sum(p2_weights)
    p2_temp = 0
    for i in range(0,window_size):
        p2_temp += p2_weights[i]*nn_score(model_f, clf, [data_X[i-window_size-label_lag]], [data_Y[i-window_size-label_lag]], [], [], 0)
    p2 = 1 - p2_temp#clf.score(data_X[-1-label_lag],data_Y[-1-label_lag])
    return p1_p2_weights[1]*p2#p1_p2_weights[0]*p1+

def Q2(data_X, data_Y, label_lag, train_X, train_Y, clf, detection_method = 'SVM'):
    if detection_method == 'SVM':
        AD = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    if detection_method == 'IF':
        AD = IsolationForest()
    AD.fit(train_X)
    y_pred = AD.predict(data_X[-1])
    return y_pred[y_pred == -1].size*1.0/len(data_X[-1])
    #sklearn_score_anomalies = IF.decision_function(train_X)
    #original_paper_score = [-1*s - .5 for s in sklearn_score_anomalies]
import torch
def Q2u(data_X, label_lag, train_X, model_f, detection_method = 'SVM'):
    if detection_method == 'SVM':
        AD = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    if detection_method == 'IF':
        AD = IsolationForest()
    last_batch_embedded = model_f(data_X[-1].cuda()).cpu().detach().numpy()
    AD.fit(model_f(torch.cat(train_X).cuda()).cpu().detach().numpy())
    y_pred = AD.predict(last_batch_embedded)
    return y_pred[y_pred == -1].size*1.0/len(last_batch_embedded)
    #sklearn_score_anomalies = IF.decision_function(train_X)
    #original_paper_score = [-1*s - .5 for s in sklearn_score_anomalies]

def overlap(f,s):
    m1,std1 = norm.fit(f)
    m2,std2 = norm.fit(s)
    if (std2, m2) < (std1, m1):   # sort to assure commutativity
        m1,m2 = m2,m1
        std1,std2 = std2,std1
    X_var, Y_var = std1**2.0, std2**2.0
    if std1*std2 == 0: return 0
    dv = Y_var - X_var
    dm = np.abs(m2 - m1)
    if not dv:
        return 1.0 - math.erf(dm / (2.0 * std1 * np.sqrt(2.0)))
    a = m1 * Y_var - m2 * X_var
    b = std1 * std2 * np.sqrt(dm**2.0 + dv * np.log(Y_var / X_var))
    x1 = (a + b) / dv
    x2 = (a - b) / dv
    return 1.0 - (np.abs(norm.cdf(x1,m2,std2) - norm.cdf(x1,m1,std1)) + np.abs(norm.cdf(x2,m2,std2) - norm.cdf(x2,m1,std1)))

def Q3u(data_X, model_f, clf, bins=10):
    certainty = clf(model_f(data_X[-1].cuda())).cpu().detach().numpy()
    f, s = np.max(certainty, axis=1), np.partition(certainty, -2, axis=1)[:,-2]
    return overlap(f, s)

def Q3(data_X, data_Y, label_lag, train_X, train_Y, clf, bins=10):
    certainty = clf.predict_proba(data_X[-1])
    f, s = np.max(certainty, axis=1), np.partition(certainty, -2, axis=1)[:,-2]
    return overlap(f, s)
    #h_f, _ = np.histogram(f, bins=bins, density=True, range=(0,1))
    #h_s, _ = np.histogram(s, bins=bins, density=True, range=(0,1))
    #h_f, h_s = h_f*1.0/bins, h_s*1.0/bins
    #return np.sum(np.minimum(h_f*1.0/bins,h_s*1.0/bins))

def Q4(data_X, data_Y, label_lag, train_X, train_Y, clf, bins=20, for_type_calc = False):
    curr = data_X[-1] if for_type_calc == False else data_X
    n_features = train_X.shape[1]
    score = np.zeros(n_features)
    for i in range(0, n_features):
        max_, min_ = np.max(np.concatenate([train_X,curr])[:,i]), np.min(np.concatenate([train_X,curr])[:,i])
        num_train, _ = np.histogram(train_X[:,i], range=(min_,max_), bins=bins)
        num_curr, _ = np.histogram(curr[:,i], range=(min_,max_), bins=bins)
        for j in range(0,bins):
            score[i] += (np.sqrt(num_train[j]*1.0/len(train_X)) - np.sqrt(num_curr[j]*1.0/len(curr)))**2
        score[i] = np.sqrt(score[i])
    return np.mean(score)
    
def Q4u(data_X, train_X, model_f, bins=20, for_type_calc = False):
    curr = model_f(data_X[-1].cuda()).cpu().detach().numpy() if for_type_calc == False else data_X
    train_X = model_f(torch.cat(train_X).cuda()).cpu().detach().numpy()
    n_features = train_X.shape[1]
    score = np.zeros(n_features)
    for i in range(0, n_features):
        max_, min_ = np.max(np.concatenate([train_X,curr])[:,i]), np.min(np.concatenate([train_X,curr])[:,i])
        num_train, _ = np.histogram(train_X[:,i], range=(min_,max_), bins=bins)
        num_curr, _ = np.histogram(curr[:,i], range=(min_,max_), bins=bins)
        for j in range(0,bins):
            score[i] += (np.sqrt(num_train[j]*1.0/len(train_X)) - np.sqrt(num_curr[j]*1.0/len(curr)))**2
        score[i] = np.sqrt(score[i])
    return np.mean(score)

def keep_last_consecutive(l):
    if len(l)==1: 
        return l
    for i in range(1,len(l)):
        if l[-i]!=l[-i-1]+1:
            return l[-i:]
    return l
