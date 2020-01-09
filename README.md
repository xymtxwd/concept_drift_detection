# Concept drift detection

A minimal working version for our concept drift detection method. More functions & benchmarks will be added. 

# Requirements
```
pip3 install sklearn
pip3 install matplotlib
pip3 install torch
pip3 install torchvision
pip3 install numpy
pip3 install pandas
pip3 install lightgbm
pip3 install xgboost
pip3 install keras
pip3 install tensorflow-gpu
pip3 install mpmath
pip3 install statsmodels
```

# Scripts and usages

datasets: specify digits datasets for drift detection experiment, i.e. MNIST, USPS

useful_functions.py: functions to calculate scores (i.e., q1, q2, etc.) for monitoring drift and other useful functions

drift_detection_method.py: drift detectors, e.g. DDM, ADWIN, etc.

models.py: classification models in PyTorch

# Experiments

In the deep learning setting, we utilize the conventional MNIST and USPS datasets in domain adaptation to validate our framework. The two digits datasets contain 10 same classes, i.e. digits 0-9, but their distributions differ. In our experiments, we use MNIST as the 'base' (or 'source') dataset, i.e. any incoming batch will contain MNIST 0-9. USPS is used as the 'drift' (or 'target') dataset, i.e. after each 100 batches, a digit will be added to the incoming data distribution. For example, at batch # 80, all samples are MNIST digits. At batch # 180, however, USPS digit 0 samples will be available. Our evaluation metric is the average accuracy in this data flow.

digits_ddm.py: main file to run for drift detection experiment using DDM benchmark

drift_experiment.py: main file to run for drift detection experiment using our method
