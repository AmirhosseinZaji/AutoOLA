import numpy as np
import os
import paths
import functions
import sys
sys.path.append(os.path.abspath('DL'))

ga_out_position = np.load(os.path.join(paths.results_path, 'ga_out_position.npy'))

functions.generate_augmented_images(ga_out_position, 150, paths.roi_images, False, 100)

curdir = os.getcwd()
os.chdir('DL')
import train
cost = train.train()
os.chdir(curdir)

print('end')