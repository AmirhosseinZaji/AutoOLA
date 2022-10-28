import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import properties
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
########################################################################
class ImageFolder(Dataset):

    def __init__(self,
                 mode,
                 randomHueSaturationValue_prob,
                 randomShiftScaleRotate_prob,
                 randomHorizontalFlip_prob,
                 randomVerticalFlip_prob,
                 randomRotate90_prob):

        self.mode = mode
        self.images, self.actual_labels = read_datasets(self.mode)
        self.randomHueSaturationValue_prob = randomHueSaturationValue_prob
        self.randomShiftScaleRotate_prob = randomShiftScaleRotate_prob
        self.randomHorizontalFlip_prob = randomHorizontalFlip_prob
        self.randomVerticalFlip_prob = randomVerticalFlip_prob
        self.randomRotate90_prob = randomRotate90_prob

    def __getitem__(self, index):
        image, label, actual_label= loader(self.images[index],
                                           self.actual_labels[index],
                                           self.randomHueSaturationValue_prob,
                                           self.randomShiftScaleRotate_prob,
                                           self.randomHorizontalFlip_prob,
                                           self.randomVerticalFlip_prob,
                                           self.randomRotate90_prob)

        image = torch.Tensor(image)
        label = torch.Tensor(label)
        actual_label = torch.Tensor(actual_label)

        return image, label, actual_label

    def __len__(self):
        return len(self.images)
########################################################################
def read_datasets(mode):
    images = []
    actual_labels = []

    if mode == "train":
        image_root = os.path.join(properties.valid_images_path)
        actual_label_root = os.path.join(properties.valid_actual_labels_path)

        for image_name in os.listdir(image_root):
            image_path = os.path.join(image_root, image_name)
            actual_label_path = os.path.join(actual_label_root, image_name)
            images.append(image_path)
            actual_labels.append(actual_label_path)

    elif mode == "valid":
        image_root = os.path.join (properties.test_images_path)
        actual_label_root = os.path.join(properties.test_actual_labels_path)

        for image_name in os.listdir (image_root):
            image_path = os.path.join (image_root, image_name)
            actual_label_path = os.path.join (actual_label_root, image_name)
            images.append (image_path)
            actual_labels.append (actual_label_path)

    return images, actual_labels
########################################################################
def loader(image_path,
           actual_label_path,
           randomHueSaturationValue_prob,
           randomShiftScaleRotate_prob,
           randomHorizontalFlip_prob,
           randomVerticalFlip_prob,
           randomRotate90_prob):

    image = np.load(image_path)
    actual_label = np.load(actual_label_path)

    if properties.resize:
        image, actual_label = resizing(image, actual_label)

    if properties.create_label_before_augmentation:

        if properties.gaussian:
            label = gaussian_GT_generator(actual_label)
        else:
            label = constant_GT_generator(actual_label)

        image, label, actual_label = random_crop_augmentation(image,
                                                              label,
                                                              actual_label,
                                                              randomHueSaturationValue_prob,
                                                              randomShiftScaleRotate_prob,
                                                              randomHorizontalFlip_prob,
                                                              randomVerticalFlip_prob,
                                                              randomRotate90_prob)

    else:
        label = None
        image, actual_label = random_crop_augmentation(image,
                                                       label,
                                                       actual_label,
                                                       randomHueSaturationValue_prob,
                                                       randomShiftScaleRotate_prob,
                                                       randomHorizontalFlip_prob,
                                                       randomVerticalFlip_prob,
                                                       randomRotate90_prob)
        if properties.gaussian:
            label = gaussian_GT_generator(actual_label)
        else:
            label = constant_GT_generator(actual_label)

    # print(actual_label.sum())
    # print(label.sum())
    # import matplotlib as mpl
    # mpl.rcParams['figure.dpi'] = 300
    # plt.subplot(311)
    # plt.imshow(image)
    # plt.subplot(312)
    # plt.imshow(label)
    # plt.subplot(313)
    # plt.imshow(actual_label)
    # plt.show()

    label = np.expand_dims(label, axis=2)
    label = np.array(label, np.float32).transpose(2, 0, 1)

    actual_label = np.expand_dims(actual_label, axis=2)
    actual_label = np.array(actual_label, np.float32).transpose(2, 0, 1)

    image = np.array(image, np.float32).transpose(2, 0, 1) / 255.0

    return image, label, actual_label
########################################################################
def cropping(image, label, actual_label):
    image_size = image.shape[0:2]
    crop_size = properties.crop_size

    # gt_features = nms(label)
    gt_features = get_gt_features(actual_label)

    random_feature = gt_features[np.random.randint(len(gt_features))]

    needed_space = crop_size / 2
    up = int(random_feature[0] + needed_space)  # should be less than image_size[0]
    down = int(random_feature[0] - needed_space)  # should be positive
    right = int(random_feature[1] + needed_space)  # should be less tahn image_size[1]
    left = int(random_feature[1] - needed_space)  # should be positive

    if up > image_size[0]:
        up = image_size[0]
        down = image_size[0] - crop_size
    elif down < 0:
        up = crop_size
        down = 0

    if right > image_size[1]:
        right = image_size[1]
        left = image_size[1] - crop_size
    elif left < 0:
        right = crop_size
        left = 0

    image = image[down:up, left:right, :]
    actual_label = actual_label[down:up, left:right]
    if label is not None:
        label = label[down:up, left:right]

    if label is None:
        return image, actual_label
    else:
        return image, label, actual_label
########################################################################
def randomHueSaturationValue(image,
                             hue_shift_limit,
                             sat_shift_limit,
                             val_shift_limit,
                             prob):
    if np.random.random() < prob:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image
########################################################################
def randomShiftScaleRotate(image,
                           label,
                           actual_label,
                           shift_limit,
                           scale_limit,
                           rotate_limit,
                           aspect_limit,
                           prob):

    if np.random.random() < prob:

        temp_image = image
        temp_label = label
        temp_actual_label = actual_label

        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        image = cv2.warpPerspective(image,
                                    mat,
                                    (width, height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0,))

        real_number_of_features = len(np.where(actual_label > 0)[0])

        actual_label = cv2.warpPerspective(actual_label,
                                           mat,
                                           (width, height),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0,))

        if actual_label.sum() < properties.point_scale:
            image = temp_image
            label = temp_label
            actual_label = temp_actual_label

        else:
            actual_label = actual_label_correction(actual_label, real_number_of_features)

            if label is not None:
                label = cv2.warpPerspective(label,
                                            mat,
                                            (width, height),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0,))

    if label is None:
        return image, actual_label
    else:
        return image, label, actual_label
########################################################################
def randomHorizontalFlip(image,
                         label,
                         actual_label,
                         prob):
    if np.random.random() < prob:
        image = cv2.flip(image, 1)
        actual_label = cv2.flip(actual_label, 1)
        if label is not None:
            label = cv2.flip(label, 1)

    if label is None:
        return image, actual_label
    else:
        return image, label, actual_label
########################################################################
def randomVerticalFlip(image,
                       label,
                       actual_label,
                       prob):
    if np.random.random() < prob:
        image = cv2.flip(image, 0)
        actual_label = cv2.flip(actual_label, 0)
        if label is not None:
            label = cv2.flip(label, 0)

    if label is None:
        return image, actual_label
    else:
        return image, label, actual_label
########################################################################
def randomRotate90(image,
                   label,
                   actual_label,
                   prob):
    if np.random.random() < prob:
        image=np.rot90(image)
        actual_label=np.rot90(actual_label)
        if label is not None:
            label=np.rot90(label)

    if label is None:
        return image, actual_label
    else:
        return image, label, actual_label
########################################################################
def get_gt_features(actual_label):
    temp = np.where(actual_label == properties.point_scale)
    temp_a = temp[0]
    temp_b = temp[1]
    temp_a = np.expand_dims(temp_a, axis=1)
    temp_b = np.expand_dims(temp_b, axis=1)
    return np.append(temp_a, temp_b, axis=1)
########################################################################
def resizing(image, actual_label):

    image = cv2.resize(image, properties.resize_to)

    scale_1 = actual_label.shape[0] / properties.resize_to[0]
    scale_2 = actual_label.shape[1] / properties.resize_to[1]
    non_zero = np.nonzero(actual_label)

    new_pos = [[], []]
    new_pos[0] = np.floor(non_zero[0] / scale_1)
    new_pos[1] = np.floor(non_zero[1] / scale_2)
    new_pos = np.array(new_pos)
    new_pos = new_pos.astype(int)

    actual_label = np.zeros(image.shape[0:2])
    actual_label[new_pos[0], new_pos[1]] = properties.point_scale

    return image, actual_label
########################################################################
def random_crop_augmentation(image,
                             label,
                             actual_label,
                             randomHueSaturationValue_prob,
                             randomShiftScaleRotate_prob,
                             randomHorizontalFlip_prob,
                             randomVerticalFlip_prob,
                             randomRotate90_prob):
    if properties.crop:
        if label is None:
            image, actual_label = cropping(image, label, actual_label)
        else:
            image, label, actual_label = cropping(image, label, actual_label)

    if properties.HueSaturationValue:
        image = randomHueSaturationValue(image,
                                         properties.hue_shift_limit,
                                         properties.sat_shift_limit,
                                         properties.val_shift_limit,
                                         randomHueSaturationValue_prob)
    if properties.ShiftScaleRotate:
        if label is None:
            image, actual_label = randomShiftScaleRotate(image,
                                                         label,
                                                         actual_label,
                                                         properties.shift_limit,
                                                         properties.scale_limit,
                                                         properties.rotate_limit,
                                                         properties.aspect_limit,
                                                         randomShiftScaleRotate_prob)
        else:
            image, label, actual_label = randomShiftScaleRotate(image,
                                                                label,
                                                                actual_label,
                                                                properties.shift_limit,
                                                                properties.scale_limit,
                                                                properties.rotate_limit,
                                                                properties.aspect_limit,
                                                                randomShiftScaleRotate_prob)

    if properties.HorizontalFlip:
        if label is None:
            image, actual_label = randomHorizontalFlip(image,
                                                      label,
                                                      actual_label,
                                                      randomHorizontalFlip_prob)
        else:
            image, label, actual_label = randomHorizontalFlip(image,
                                                              label,
                                                              actual_label,
                                                              randomHorizontalFlip_prob)

    if properties.VerticalFlip:
        if label is None:
            image, actual_label = randomVerticalFlip(image,
                                                     label,
                                                     actual_label,
                                                     randomVerticalFlip_prob)
        else:
            image, label, actual_label = randomVerticalFlip(image,
                                                            label,
                                                            actual_label,
                                                            randomVerticalFlip_prob)

    if properties.Rotate90:
        if label is None:
            image, actual_label = randomRotate90(image,
                                                label,
                                                actual_label,
                                                randomRotate90_prob)
        else:
            image, label, actual_label = randomRotate90(image,
                                                        label,
                                                        actual_label,
                                                        randomRotate90_prob)

    if label is None:
        return image, actual_label
    else:
        return image, label, actual_label
########################################################################
def gaussian_GT_generator(actual_label):
    label = gaussian_filter(actual_label,
                            sigma=(properties.sigma, properties.sigma),
                            order=0)
    return label
########################################################################
def constant_GT_generator(actual_label):
    label = gaussian_filter(actual_label,
                            sigma=(properties.sigma, properties.sigma),
                            order=0)
    label[np.nonzero(label)] = actual_label.sum() / len(np.nonzero(label)[0])
    return label
########################################################################
def actual_label_correction(actual_label, real_number_of_features):
    for i in range(real_number_of_features):
        points = np.where(actual_label > 0)

        if i >= len(points[0]):
            break

        shape = actual_label.shape

        xmin = points[0][i] - 3 if points[0][i] - 3 >= 0 else 0
        xmax = points[0][i] + 4 if points[0][i] + 4 <= shape[0] else shape[0]
        ymin = points[1][i] - 3 if points[1][i] - 3 >= 0 else 0
        ymax = points[1][i] + 4 if points[1][i] + 4 <= shape[1] else shape[1]
        box = actual_label[xmin:xmax, ymin:ymax]
        box_max_position = np.where(box == np.amax(box))
        actual_label[xmin:xmax, ymin:ymax] = 0

        if len(box_max_position[0]) > 1:
            a = box_max_position[0][len(box_max_position[0])//2]
            b = box_max_position[1][len(box_max_position[1])//2]
            actual_label[xmin + a, ymin + b] = properties.point_scale
        else:
            actual_label[xmin + box_max_position[0], ymin + box_max_position[1]] = properties.point_scale

    return actual_label
########################################################################



