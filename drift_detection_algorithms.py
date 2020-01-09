import sys
import math
import numpy as np
class DDM:
    """
    The drift detection method (DDM) controls the number of errors
    produced by the learning model during prediction. It compares
    the statistics of two windows: the first contains all the data,
    and the second contains only the data from the beginning until
    the number of errors increases.
    Their method doesn't store these windows in memory.
    It keeps only statistics and a window of recent errors data.".

    References
    ---------
    Gama, J., Medas, P., Castillo, G., Rodrigues, P.:
    "Learning with drift detection". In: Bazzan, A.L.C., Labidi,
    S. (eds.) SBIA 2004. LNCS (LNAI), vol. 3171, pp. 286–295. Springer, Heidelberg (2004)
    """

    def __init__(self, a=3, b=2, min_samples=30):
        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        self.m_psmin = sys.float_info.max
        self.m_pmin = sys.float_info.max
        self.m_smin = sys.float_info.max
        self.change_detected = False
        self.is_initialized = True
        self.estimation = 0.0
        self.is_warning_zone = False
        self.a = a
        self.b = b
        self.past_input = []
        self.min_samples = min_samples
    def set_input(self, prediction):
        """
        The number of errors in a sample of n examples is modelled by a binomial distribution.
        For each point t in the sequence that is being sampled, the error rate is the probability
        of mis-classifying p(t), with standard deviation s(t).
        DDM checks two conditions:
        1) p(t) + s(t) > p(min) + 2 * s(min) for the warning level
        2) p(t) + s(t) > p(min) + 3 * s(min) for the drift level

        Parameters
        ----------
        prediction : new element, it monitors the error rate

        Returns
        -------
        change_detected : boolean
                    True if a change was detected.
        """
        if self.change_detected is True or self.is_initialized is False:
            self.reset()
            self.is_initialized = True
        self.past_input.append(prediction)
        self.m_p += (prediction - self.m_p) / float(self.m_n)
        self.m_s = np.std(self.past_input)/np.sqrt(self.m_n)#math.sqrt(self.m_p * (1 - self.m_p) / float(self.m_n))

        self.m_n += 1
        self.estimation = self.m_p
        self.change_detected = False

        if self.m_n < self.min_samples:
            return False

        if self.m_p + self.m_s <= self.m_psmin:
            self.m_pmin = self.m_p;
            self.m_smin = self.m_s;
            self.m_psmin = self.m_p + self.m_s;

        if self.m_p + self.m_s > self.m_pmin + self.a * self.m_smin:
            self.change_detected = True
        elif self.m_p + self.m_s > self.m_pmin + self.b * self.m_smin:
            self.is_warning_zone = True
        else:
            self.is_warning_zone = False

        return self.change_detected

    def reset(self):
        """reset the DDM drift detector"""
        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        self.m_psmin = sys.float_info.max
        self.m_pmin = sys.float_info.max
        self.m_smin = sys.float_info.max
        self.past_input = []

class PageHinkley:
    """ Page-Hinkley method for concept drift detection
    Notes
    -----
    This change detection method works by computing the observed 
    values and their mean up to the current moment. Page-Hinkley
    won't output warning zone warnings, only change detections. 
    The method works by means of the Page-Hinkley test [1]_. In general
    lines it will detect a concept drift if the observed mean at 
    some instant is greater then a threshold value lambda.
    References
    ----------
    .. [1] E. S. Page. 1954. Continuous Inspection Schemes.
       Biometrika 41, 1/2 (1954), 100–115.
    
    Parameters
    ----------
    min_num_instances: int (default=30)
        The minimum number of instances before detecting change.
    delta: float (default=0.005)
        The delta factor for the Page Hinkley test.
    threshold: int (default=50)
        The change detection threshold (lambda).
    alpha: float (default=1 - 0.0001)
        The forgetting factor, used to weight the observed value 
        and the mean.
    """
    
    def __init__(self, min_num_instances=30, delta=0.005, threshold=50, alpha=1 - 0.0001):
        self.min_instances = min_num_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = None
        self.sample_count = None
        self.sum = None
        self.reset()

    def reset(self):
        """ reset
        Resets the change detector parameters.
        """
        self.in_concept_change = False
        self.in_warning_zone = False
        self.estimation = 0.0
        self.delay = 0.0
        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def add_element(self, x):
        """ Add a new element to the statistics
        
        Parameters
        ----------
        x: numeric value
            The observed value, from which we want to detect the
            concept change.
        
        Notes
        -----
        After calling this method, to verify if change was detected, one 
        should call the super method detected_change, which returns True 
        if concept drift was detected and False otherwise.
        
        """
        if self.in_concept_change:
            self.reset()

        self.x_mean = self.x_mean + (x - self.x_mean) / float(self.sample_count)
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)

        self.sample_count += 1

        self.estimation = self.x_mean
        self.in_concept_change = False
        self.in_warning_zone = False

        self.delay = 0

        if self.sample_count < self.min_instances:
            return None

        if self.sum > self.threshold:
            self.in_concept_change = True

    def get_info(self):
        """ Collect information about the concept drift detector.
        Returns
        -------
        string
            Configuration for the concept drift detector.
        """
        description = type(self).__name__ + ': '
        description += 'min_num_instances: {} - '.format(self.min_instances)
        description += 'delta: {} - '.format(self.delta)
        description += 'threshold (lambda): {} - '.format(self.threshold)
        description += 'delta: {} - '.format(self.delta)
        description += 'alpha: {} - '.format(self.alpha)
        return description
    
class Ewma(object):
    """
    In statistical quality control, the EWMA chart (or exponentially weighted moving average chart)
    is a type of control chart used to monitor either variables or attributes-type data using the monitored business
    or industrial process's entire history of output. While other control charts treat rational subgroups of samples
    individually, the EWMA chart tracks the exponentially-weighted moving average of all prior sample means.
    WIKIPEDIA: https://en.wikipedia.org/wiki/EWMA_chart
    """

    def __init__(self, alpha=0.3, coefficient=3):
        """
        :param alpha: Discount rate of ewma, usually in (0.2, 0.3).
        :param coefficient: Coefficient is the width of the control limits, usually in (2.7, 3.0).
        """
        self.alpha = alpha
        self.coefficient = coefficient

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.
        :param X: the time series to detect of
        :param type X: pandas.Series
        :return: 1 denotes normal, 0 denotes abnormal
        """
        s = [X[0]]
        for i in range(1, len(X)):
            temp = self.alpha * X[i] + (1 - self.alpha) * s[-1]
            s.append(temp)
        s_avg = np.mean(s)
        sigma = np.sqrt(np.var(X))
        ucl = s_avg + self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
        lcl = s_avg - self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
        if s[-1] > ucl or s[-1] < lcl:
            return 0
        return 1
    
class AdwinListNode(object):
    """Implementation of a node of adwin list"""

    def __init__(self, max_number_of_buckets):
        """Init a node with a given parameter number_of_buckets

        Parameters
        ----------
        max_number_of_buckets : In each row, the max number of buckets
        """
        self.max_number_of_buckets = max_number_of_buckets
        self.size = 0
        self.next = None
        self.prev = None
        self.sum = []
        self.variance = []
        for i in range(self.max_number_of_buckets + 1):
            self.sum.append(0.0)
            self.variance.append(0.0)

    def insert_bucket(self, value, variance):
        """Insert a bucket at the end

        Parameters
        ----------
        value: the totally size of the new one
        variance : the variance of the new one
        """
        self.sum[self.size] = value
        self.variance[self.size] = variance
        self.size += 1

    def drop_bucket(self, n=1):
        """Drop the older portion of the bucket

        Parameters
        ----------
        n :number data of drop bucket
        """
        for k in range(n, self.max_number_of_buckets + 1):
            self.sum[k - n] = self.sum[k]
            self.variance[k - n] = self.variance[k]
        for k in range(1, n + 1):
            self.sum[self.max_number_of_buckets - k + 1] = 0.0
            self.variance[self.max_number_of_buckets - k + 1] = 0.0
        self.size -= n

class AdwinList(object):
    def __init__(self, max_number_bucket):
        """Init a adwin list with a given parameter max_number_buckets

        Parameters
        ----------
        max_number_bucket : max number of elements in the bucket
        """
        self.head = None
        self.tail = None
        self.count = 0
        self.max_number_bucket = max_number_bucket
        self.add_to_head()

    def add_to_tail(self):
        """add a node at the tail of adwin list, used in the initialization of an AdwinList"""
        temp = AdwinListNode(self.max_number_bucket)
        if self.tail is not None:
            temp.prev = self.tail
            self.tail.next = temp
        self.tail = temp
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def add_to_head(self):
        """Add a node to the head of an AdwinList"""
        temp = AdwinListNode(self.max_number_bucket)
        if self.head is not None:
            temp.next = self.head
            self.head.prev = temp
        self.head = temp
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def remove_from_head(self):
        """Remove the head node of an AdwinList"""
        temp = self.head
        self.head = self.head.next
        if self.head is not None:
            self.head.prev = None
        else:
            self.tail = None
        self.count -= 1

    def remove_from_tail(self):
        """Remove the tail node of an AdwinList"""
        temp = self.tail
        self.tail = self.tail.prev
        if self.tail is None:
            self.head = None
        else:
            self.tail.next = None
        self.count -= 1

class Adwin(object):
    """The Adwin algorithm is a change detector and estimator.
    It keeps a sliding (variable-length) window with the most
    recently read example,with the property that the window
    has the maximal length statistically consistent with the
    hypothesis that "there has been no change in the average
    value inside the window".

    References
    ----------
    A. Bifet, R. Gavalda. (2007). "Learning from Time-Changing
    Data with Adaptive Windowing". Proceedings of the 2007 SIAM
    International Conference on Data Mining 443-448.
    http://www.lsi.upc.edu/~abifet/Timevarying.pdf

    A. Bifet, J. Read, B.Pfahringer.G. Holmes, I. Zliobaite.
    (2013). "CD-MOA: Change Detection Framework for Massive Online
    Analysis". Springer Berlin Heidelberg 8207(9): 443-448.
    https://sites.google.com/site/zliobaitefiles/cdMOA-CR.pdf?attredirects=0
    """

    def __init__(self, delta=0.01):
        """Init the buckets

        Parameters
        ----------
        delta : float
            confidence value.
        """

        self.mint_clock = 1.0
        self.min_window_length = 16
        self.delta = delta
        self.max_number_of_buckets = 5
        self.bucket_list = AdwinList(self.max_number_of_buckets)
        self.mint_time = 0.0
        self.min_clock = self.mint_clock
        self.mdbl_error = 0.0
        self.mdbl_width = 0.0
        self.last_bucket_row = 0
        self.sum = 0.0
        self.width = 0.0
        self.variance = 0.0
        self.bucket_number = 0

    def get_estimation(self):
        """Get the estimation value"""
        if self.width > 0:
            return self.sum / float(self.width)
        else:
            return 0

    def set_input(self, value):
        """Add new element and reduce the window

        Parameters
        ----------
        value : new element

        Returns
        -------
        boolean: the return value of the method check_drift(), true if a drift was detected.
        """
        self.insert_element(value)
        self.compress_buckets()
        return self.check_drift()

    def length(self):
        """Get the length of window"""
        return self.width

    def insert_element(self, value):
        """insert new bucket"""
        self.width += 1
        self.bucket_list.head.insert_bucket(float(value), 0.0)
        self.bucket_number += 1
        if self.width > 1:
            self.variance += (self.width - 1) * (value - self.sum / (self.width - 1)) \
                             * (value - self.sum / (self.width - 1)) / self.width
        self.sum += value

    def compress_buckets(self):
        """
        Merge buckets.
        Find the number of buckets in a row, if the row is full, then merge the two buckets.
        """
        i = 0
        cont = 0
        cursor = self.bucket_list.head
        next_node = None
        while True:
            k = cursor.size
            if k == self.max_number_of_buckets + 1:
                next_node = cursor.next
                if next_node is None:
                    self.bucket_list.add_to_tail()
                    next_node = cursor.next
                    self.last_bucket_row += 1
                n1 = self.bucket_size(i)
                n2 = self.bucket_size(i)
                u1 = cursor.sum[0] / n1
                u2 = cursor.sum[1] / n2
                internal_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
                next_node.insert_bucket(cursor.sum[0] + cursor.sum[1],
                                        cursor.variance[0] + cursor.variance[1] + internal_variance)
                self.bucket_number -= 1
                cursor.drop_bucket(2)
                if next_node.size <= self.max_number_of_buckets:
                    break
                else:
                    break
            cursor = cursor.next
            i += 1
            if cursor is None:
                break

    def check_drift(self):
        """
        Reduce the window, detecting if there is a drift.

        Returns
        -------
        change : boolean value
        Result of whether the window has changed.
        """

        change = False
        exit = False
        cursor = None
        self.mint_time += 1
        if self.mint_time % self.min_clock == 0 and self.width > self.min_window_length:
            reduce_width = True
            while reduce_width:
                reduce_width = False
                exit = False
                n0 = 0.0
                n1 = float(self.width)
                u0 = 0.0
                u1 = float(self.sum)
                cursor = self.bucket_list.tail
                i = self.last_bucket_row
                while True:
                    for k in range(cursor.size):
                        if i == 0 and k == cursor.size - 1:
                            exit = True
                            break
                        n0 += self.bucket_size(i)
                        n1 -= self.bucket_size(i)
                        u0 += cursor.sum[k]
                        u1 -= cursor.sum[k]
                        min_length_of_subwindow = 5
                        if n0 >= min_length_of_subwindow and n1 >= min_length_of_subwindow and self.cut_expression(n0,
                                                                                                                   n1,
                                                                                                                   u0,
                                                                                                                   u1):
                            reduce_width = True
                            change = True
                            if self.width > 0:
                                self.delete_element()
                                exit = True
                                break
                    cursor = cursor.prev
                    i -= 1
                    if exit or cursor is None:
                        break
        return change

    def delete_element(self):
        """delete the bucket at the tail of window"""
        node = self.bucket_list.tail
        n1 = self.bucket_size(self.last_bucket_row)
        self.width -= n1
        self.sum -= node.sum[0]
        u1 = node.sum[0] / n1
        incVariance = float(
            node.variance[0] + n1 * self.width * (u1 - self.sum / self.width) * (u1 - self.sum / self.width)) / (
                          float(n1 + self.width))
        self.variance -= incVariance
        node.drop_bucket()
        self.bucket_number -= 1
        if node.size == 0:
            self.bucket_list.remove_from_tail()
            self.last_bucket_row -= 1

    def cut_expression(self, n0_, n1_, u0, u1):
        """Expression calculation"""
        n0 = float(n0_)
        n1 = float(n1_)
        n = float(self.width)
        diff = float(u0 / n0) - float(u1 / n1)
        v = self.variance / self.width
        dd = math.log(2.0 * math.log(n) / self.delta)
        min_length_of_subwindow = 5
        m = (float(1 / (n0 - min_length_of_subwindow + 1))) + (float(1 / (n1 - min_length_of_subwindow + 1)))
        eps = math.sqrt(2 * m * v * dd) + float(2 / 3 * dd * m)
        if math.fabs(diff) > eps:
            return True
        else:
            return False

    def bucket_size(self, Row):
        return int(math.pow(2, Row))