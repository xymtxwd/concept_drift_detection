# Concept drift detection

A minimal working version for concept drift detection.

#Scripts and usages

```
datasets
```
- specify digits datasets for drift detection experiment, i.e. MNIST, USPS

```
useful_functions.py
```
- functions to calculate scores (i.e., q1, q2, etc.) for monitoring drift and other useful functions


```
drift_detection_method.py
```
- drift detectors, e.g. DDM, ADWIN, etc.

```
models.py
```
- classification models in PyTorch

```
drift_experiment.py
```
- main file to run for drift detection experiment
