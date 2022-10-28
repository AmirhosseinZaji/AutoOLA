import os
import numpy as np
import properties
import matplotlib.pyplot as plt

curdir = os.curdir
train_arrays_dir = os.path.join(os.curdir, 'results', 'arrays', 'train')
valid_arrays_dir = os.path.join(os.curdir, 'results', 'arrays', 'valid')

# f1
train_f1 = np.load(os.path.join(train_arrays_dir, 'f1.npy'))
train_f1 = train_f1.reshape(-1, len(properties.distance_threshold))

valid_f1 = np.load(os.path.join(valid_arrays_dir, 'f1.npy'))
valid_f1 = valid_f1.reshape(-1, len(properties.distance_threshold))

print('helo')
