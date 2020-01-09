from torchvision import transforms
import torchvision
import torch
from .mnist import *
from .svhn import *
from .usps import *
import numpy as np

def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0
    def _cutout(image):
        image = np.asarray(image).copy()
        if np.random.random() > p:
            return image
        h, w = image.shape[:2]
        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset
        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image
    return _cutout

def affine_func():
    def aff(image):
        return torchvision.transforms.functional.affine(image, 0, [0.1,0.1], 1, 0)
    return aff


def get_dataset(task, drift_num, n_samples=5000):
    if task == 's2m':
        train_dataset = SVHN('../data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        
        test_dataset = MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))

######################

    elif task == 'u2m':
        train_dataset = USPS('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28,padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1576,), ( 0.2327,))
                ]))
        
        test_dataset = MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    #transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(( 0.1309, ), ( 0.2890,))
                ]))

#######################

    else:
        train_dataset = MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

        test_dataset = USPS('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    
    return relabel_dataset(train_dataset, test_dataset, task, drift_num, n_samples)


def relabel_dataset(train_dataset, test_dataset, task, drift_num, n_samples):
    image_path = []
    image_label = []
    if task == 's2m':
        for i in range(len(train_dataset.data)):
            if True:
                image_path.append(train_dataset.data[i])
                image_label.append(train_dataset.labels[i])
        train_dataset.data = image_path
        train_dataset.labels = image_label
    if task == 'u2m':
        for i in range(len(train_dataset.train_data)):
            if True:
                image_path.append(train_dataset.train_data[i])
                image_label.append(train_dataset.train_labels[i])
        train_dataset.train_data = image_path
        train_dataset.train_labels = image_label
    if task == 'm2u':
        for i in range(len(train_dataset.train_data)):
            if True:
                image_path.append(train_dataset.train_data[i])
                image_label.append(train_dataset.train_labels[i])
        train_dataset.data = image_path
        train_dataset.targets = image_label

    test_image_path = []
    test_image_label = []
    if task == 's2m':
        for i in range(len(test_dataset.train_data)):
            if int(test_dataset.train_labels[i]) < 4:
                test_image_path.append(test_dataset.train_data[i])
                test_image_label.append(test_dataset.train_labels[i])
            elif int(test_dataset.train_labels[i]) > 6:
                test_image_path.append(test_dataset.train_data[i])
                test_image_label.append(4)
        test_dataset.data = test_image_path
        test_dataset.targets = test_image_label

    if task == 'u2m':
        for i in range(len(test_dataset.train_data)):
            if int(test_dataset.train_labels[i]) < drift_num:
                test_image_path.append(test_dataset.train_data[i])
                test_image_label.append(test_dataset.train_labels[i])
        test_dataset.data = test_image_path
        test_dataset.targets = test_image_label

    if task == 'm2u':
        for i in range(len(test_dataset.train_data)):
            if int(test_dataset.train_labels[i]) < drift_num:
                test_image_path.append(test_dataset.train_data[i])
                test_image_label.append(test_dataset.train_labels[i])
        test_dataset.train_data = test_image_path
        test_dataset.train_labels = test_image_label


    return train_dataset, test_dataset