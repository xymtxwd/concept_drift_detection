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
pip3 install networkx
pip3 install pillow==6.1
```

# Structured data experiments

In the conventional structured data setting, we have some datasets which contain drifts. If you would like to run experiments, please download data from https://drive.google.com/open?id=1X22viKZER9PlwoqmxjkDe5izDBWFN4AR, then run `drift_experiment_structured.py`.

```
python drift_experiment_structured.py --model ours --classifier lgbm --dataset elec
```

arguments: 
- model: ['ours', 'ourslinear', 'ddm', 'ph','adwin','ewma'], default='ours'
- classifier: ['lgbm', 'lstm', 'xgb','rf','ffn'], default='ffn'
- dataset: ['elec', 'sea', 'rbf','hyperplane','weather','weather2'], default='sea'
- batch_size, default=64
- lstm_ae, default=1, (use lstm as autoencoder)

Our evaluation metric is the average accuracy in this data flow.

# Unstructured data experiments (images)

In the deep learning setting, we utilize the conventional MNIST and USPS datasets in domain adaptation to validate our framework. The two digits datasets contain 10 same classes, i.e. digits 0-9, but their distributions differ. In our experiments, we use MNIST as the 'base' (or 'source') dataset, i.e. any incoming batch will contain MNIST 0-9. USPS is used as the 'drift' (or 'target') dataset, i.e. after each 100 batches, a digit will be gradually added to the incoming data distribution. For example, at batch # 80, all samples are MNIST digits. At batch # 180, however, USPS digit 0 samples will be available. 


```
python drift_experiment_gradual_and_sudden.py --model ours
```

arguments: 
- model: ['ours', 'ddm', 'ph','adwin','ewma'], default='ours'

Our evaluation metric is the average accuracy in this data flow.

# Inference

To test on your own image dataset, you need `inference_image.py`, and prepare data in batches. Each batch should be a folder in a certain format, please refer to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision. Then put the names of the folder in a .txt file, which should contain one image folder address per line. Change name in line 153. 
In line 143 (image encoder) 144(classifier) 145(image decoder), you need to change the model to fit the size of your own data. This also needs to be changed in line 407-409. For inference for structured data, please use `inference_structured.py`.

# Scripts and usages

`datasets`: specify digits datasets for drift detection experiment, i.e. MNIST, USPS

`useful_functions.py`: functions to calculate scores (i.e., q1, q2, etc.) for monitoring drift and other useful functions

`drift_detection_method.py`: drift detectors, e.g. DDM, ADWIN, etc.

`models.py`: classification models in PyTorch

`drift_experiment_structured.py`: main file to run for structured drift detection experiment

`drift_experiment_gradual_and_sudden.py`: main file to run for unstructured drift detection experiment

`inference_image.py`: main file to run for inference


# References

We thank Alejandro Molina for the great SPN implementation: https://github.com/SPFlow/SPFlow
