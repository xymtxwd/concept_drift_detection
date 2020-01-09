# Concept drift detection

A minimal working version for our concept drift detection method. More will be added.

# Scripts and usages

datasets: specify digits datasets for drift detection experiment, i.e. MNIST, USPS

useful_functions.py: functions to calculate scores (i.e., q1, q2, etc.) for monitoring drift and other useful functions

drift_detection_method.py: drift detectors, e.g. DDM, ADWIN, etc.

models.py: classification models in PyTorch

# Experiments

digits_ddm.py: main file to run for drift detection experiment using DDM benchmark

drift_experiment.py: main file to run for drift detection experiment using our method
