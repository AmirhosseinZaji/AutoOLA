import cv2
import matplotlib.pyplot as plt
import numpy as np
import paths
import os
import z_functions
from imgaug import augmenters as iaa

ground_name = 'ground_2e0b83da-c175-408d-a02e-d2f064d5050f_9.png'
spike_name = 'spike_2e0b83da-c175-408d-a02e-d2f064d5050f_47.png'
leaf_name = 'leaf_312fe9e6-5437-4cac-b38b-d9e717a9a4ea_25.png'
stem_name = 'stem_312fe9e6-5437-4cac-b38b-d9e717a9a4ea_6.png'

os.chdir(properties.roi_images)

def read_figure(name):
    image = cv2.imread(name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    image, temp = z_functions.generate_image_mask(image)
    return image


def normalize(X):
    return (X / 255.0).copy()


def denormalize(X):
    X_dn = (X * 255).copy()
    return X_dn


def augmentation(aug_type, magnitude, name):

    X = read_figure(name)

    plt.figure()
    plt.imshow(X)

    if False:
        pass

    elif aug_type == "crop":
        crop = iaa.Crop(px=(0, int(magnitude * 32)))
        X_aug = crop(image=X)

    elif aug_type == "gaussian_blur":
        gaussian_blur = iaa.GaussianBlur(sigma=(0, magnitude * 5.0))
        X_aug = gaussian_blur(image=X)

    elif aug_type == "rotate":
        rotate = iaa.Affine(rotate=(-180 * magnitude, 180 * magnitude))
        X_aug = rotate(image=X)

    elif aug_type == "shear":
        shear = iaa.Affine(shear=(-90 * magnitude, 90 * magnitude))
        X_aug = shear(image=X)

    elif aug_type == "translate_x":
        translate_x = iaa.Affine(translate_percent={"x": (-magnitude, magnitude), "y": (0, 0)})
        X_aug = translate_x(image=X)

    elif aug_type == "translate_y":
        translate_y = iaa.Affine(translate_percent={"x": (0, 0), "y": (-magnitude, magnitude)})
        X_aug = translate_y(image=X)

    elif aug_type == "horizontal_flip":
        horizontal_flip = iaa.Fliplr(magnitude)
        X_aug = horizontal_flip(image=X)

    elif aug_type == "vertical_flip":
        vertical_flip = iaa.Flipud(magnitude)
        X_aug = vertical_flip(image=X)

    elif aug_type == "sharpen":
        sharpen = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.50, 5 * magnitude))
        X_aug = sharpen(image=X)

    elif aug_type == "emboss":
        emboss = iaa.Emboss(alpha=(0, 1.0), strength=(0.0, 20.0 * magnitude))
        X_aug = emboss(image=X)

    elif aug_type == "additive_gaussian_noise":
        additive_gaussian_noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, magnitude * 255), per_channel=0.5)
        X_aug = additive_gaussian_noise(image=X)

    elif aug_type == "dropout":
        dropout = iaa.Dropout((0.01, max(0.011, magnitude)), per_channel=0.5)  # Dropout first argument should be smaller than second one
        X_aug = dropout(image=X)

    elif aug_type == "coarse_dropout":
        coarse_dropout = iaa.CoarseDropout((0.03, 0.15), size_percent=(0.30, np.log10(magnitude * 3)), per_channel=0.2)
        X_aug = coarse_dropout(image=X)

    elif aug_type == "gamma_contrast":
        X_norm = normalize(X)
        gamma_contrast = iaa.GammaContrast(magnitude * 1.75)  # needs 0-1 values
        X_aug_norm = gamma_contrast(image=X_norm)
        X_aug = denormalize(X_aug_norm)

    elif aug_type == "brighten":
        brighten = iaa.Add((int(-40 * magnitude), int(40 * magnitude)), per_channel=0.5)  # brighten
        X_aug = brighten(image=X)

    elif aug_type == "invert":
        invert = iaa.Invert(1.0)  # magnitude not used
        X_aug = invert(image=X)

    elif aug_type == "fog":
        fog = iaa.Fog()  # magnitude not used
        X_aug = fog(image=X)

    elif aug_type == "clouds":
        clouds = iaa.Clouds()  # magnitude not used
        X_aug = clouds(image=X)

    elif aug_type == "histogram_equalize":
        histogram_equalize = iaa.AllChannelsHistogramEqualization()  # magnitude not used
        X_aug = histogram_equalize(image=X)

    elif aug_type == "super_pixels":  # deprecated
        X_norm = normalize(X)
        X_norm2 = (X_norm * 2) - 1
        super_pixels = iaa.Superpixels(p_replace=(0, magnitude), n_segments=(100, 100))
        X_aug_norm2 = super_pixels(image=X_norm2)
        X_aug_norm = (X_aug_norm2 + 1) / 2
        X_aug = denormalize(X_aug_norm)

    elif aug_type == "perspective_transform":
        X_norm = normalize(X)
        perspective_transform = iaa.PerspectiveTransform(scale=(0.01, max(0.02, magnitude)))  # first scale param must be larger
        X_aug_norm = perspective_transform(image=X_norm)
        np.clip(X_aug_norm, 0.0, 1.0, out=X_aug_norm)
        X_aug = denormalize(X_aug_norm)

    elif aug_type == "elastic_transform":  # deprecated
        X_norm = normalize(X)
        X_norm2 = (X_norm * 2) - 1
        elastic_transform = iaa.ElasticTransformation(alpha=(0.0, max(0.5, magnitude * 300)), sigma=5.0)
        X_aug_norm2 = elastic_transform(image=X_norm2)
        X_aug_norm = (X_aug_norm2 + 1) / 2
        X_aug = denormalize(X_aug_norm)

    elif aug_type == "add_to_hue_and_saturation":
        add_to_hue_and_saturation = iaa.AddToHueAndSaturation((int(-45 * magnitude), int(45 * magnitude)))
        X_aug = add_to_hue_and_saturation(image=X)

    elif aug_type == "coarse_salt_pepper":
        coarse_salt_pepper = iaa.CoarseSaltAndPepper(p=0.2, size_percent=magnitude)
        X_aug = coarse_salt_pepper(image=X)

    elif aug_type == "grayscale":
        grayscale = iaa.Grayscale(alpha=(0.0, magnitude))
        X_aug = grayscale(image=X)

    elif aug_type == "piecewise_affine":
        piecewise_affine = iaa.PiecewiseAffine(scale=(0.0, min(0.05, magnitude)))
        X_aug = piecewise_affine(image=X)

    plt.figure()
    plt.imshow(X_aug)
    plt.show()

    return X_aug

roi_name = spike_name
aug_type = "piecewise_affine"
for i in range(10):
    augmentation(aug_type, 1, roi_name)




print('end')

