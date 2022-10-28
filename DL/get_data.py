import os
import shutil
import numpy as np
from scipy.io import loadmat
import properties
import cv2
########################################################################
def create_dataset(dst_image,
                   dst_actual_label,
                   actual_labels,
                   src):

    for i, actual_label in enumerate(actual_labels):

        img_path = os.path.join(src, f"wheat2017_{str(i + 1).zfill(4)}.jpg")
        image = np.array(cv2.imread(img_path))
        actual_label = generate_actual_label(actual_label[0][0][0], image.shape[0:2])

        np.save(os.path.join(dst_image, str(i + 1)) + ".npy", image)
        np.save(os.path.join(dst_actual_label, str(i + 1)) + ".npy", actual_label)
########################################################################
def generate_actual_label(label_info, image_shape):
    # create an empty density map
    actual_label = np.zeros(image_shape, dtype=np.uint8)

    for x, y, *_ in label_info:
        if y < image_shape[0] and x < image_shape[1]:
            actual_label[int(y)][int(x)] = properties.point_scale

    return actual_label
########################################################################
def generate_Pound1_data():
    # create the necessary folders
    dst = os.path.join("..", "running_dataset")

    dst_train = os.path.join(dst, "training")
    dst_valid = os.path.join(dst, "validation")

    dst_train_image = os.path.join(dst_train, "images")
    dst_valid_image = os.path.join(dst_valid, "images")

    dst_train_actual_label = os.path.join(dst_train, "actual_labels")
    dst_valid_actual_label = os.path.join(dst_valid, "actual_labels")

    if os.path.exists(dst):
        shutil.rmtree(dst)

    os.makedirs(dst)
    os.makedirs(dst_train_image)
    os.makedirs(dst_train_actual_label)
    os.makedirs(dst_valid_image)
    os.makedirs(dst_valid_actual_label)

    src_train = os.path.join("..", "mat_output_dataset", "training", "frames")
    src_valid = os.path.join("..", "mat_output_dataset", "validation", "frames")

    # load labels information from provided MATLAB file
    actual_labels_train = loadmat(os.path.join(src_train, "..", "Pound1_gt_train.mat"))['train_frame'][0]
    actual_labels_valid = loadmat(os.path.join(src_valid, "..", "Pound1_gt_valid.mat"))['valid_frame'][0]

    # create datasets for CNN
    create_dataset(dst_train_image,
                   dst_train_actual_label,
                   actual_labels_train,
                   src_train)

    create_dataset(dst_valid_image,
                   dst_valid_actual_label,
                   actual_labels_valid,
                   src_valid)
########################################################################

if __name__ == '__main__':
    generate_Pound1_data()
