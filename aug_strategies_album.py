import time
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import paths
import os
from functions import generate_image_mask
from functions import add_padding


ground_name = 'ground_2e0b83da-c175-408d-a02e-d2f064d5050f_9.png'
spike_name = 'spike_2e0b83da-c175-408d-a02e-d2f064d5050f_47.png'
leaf_name = 'leaf_312fe9e6-5437-4cac-b38b-d9e717a9a4ea_25.png'
stem_name = 'stem_312fe9e6-5437-4cac-b38b-d9e717a9a4ea_6.png'

def visualize(image):
    plt.figure()
    plt.axis('off')
    plt.imshow(image)

def read_figure(name):
    image = cv2.imread(os.path.join(paths.roi_images, name), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    rgb, trans = generate_image_mask (image)
    # image = image[:, :, 0:3]
    rgb = cv2.cvtColor (rgb, cv2.COLOR_BGR2RGB)
    rgb, trans = add_padding (rgb, trans, None, False)
    return rgb

image = read_figure(spike_name)
visualize(image)

# Add number of spikes, stems, leaves, grounds
# transform = A.Blur(blur_limit=[3, 7], p=1)
# transform = A.MedianBlur(blur_limit=[3,7], p=1)
# transform = A.ISONoise(color_shift=(0.01, 1), intensity=(0.01, 0.2), p=1)
# transform = A.CLAHE(clip_limit=0, tile_grid_size=(8, 8), p=1)
# transform = A.Downscale(scale_min=0.99, scale_max=0.99, p=1)
# transform = A.Emboss(alpha=(1, 1), strength=(0.2, 0.7), p=1)
# transform = A.RGBShift(r_shift_limit=19.5, g_shift_limit=0, b_shift_limit=0, p=1)
# transform = A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1)
# transform = A.MultiplicativeNoise(multiplier=(0.7, 1.3), per_channel=True, elementwise=False, p=1)
# transform = A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=1)
# transform = A.RandomToneCurve(scale=0, p=1)
# transform = A.CoarseDropout(max_holes=16, max_height=1, max_width=1, fill_value=0, p=1)

# transform = A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), interpolation=1, border_mode=1, p=1)
# transform = A.ElasticTransform(alpha=100, sigma=10, alpha_affine=5, interpolation=1, border_mode=4, value=None, mask_value=None, approximate=False, same_dxdy=False, p=1)
# transform = A.Perspective(scale=(0, 0.2), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=True, interpolation=1, p=1)
# transform = A.PiecewiseAffine(scale=(0.01, 0.04), nb_rows=4, nb_cols=4, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, keypoints_threshold=0.01, p=1)
# transform = A.Flip(p=1)
# transform = A.Affine(scale=(0.5, 1.2), translate_percent=None, translate_px=None, rotate=(0, 0), shear=(0, 0), interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, p=1)
# transform = A.Affine(scale=(1, 1), translate_percent=None, translate_px=None, rotate=(0, 0), shear=(-30, 30), interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, p=1)
# transform = A.Affine(scale=(1, 1), translate_percent=None, translate_px=None, rotate=(-90, 90), shear=(0, 0), interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, p=1)

image_ = transform(image=image)['image']
visualize(image_)

print('end')