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

digits_ddm.py: main file to run for drift detection experiment using DDM benchmark

drift_experiment.py: main file to run for drift detection experiment using our method
