# model properties
input_channels = 3

# probabilities
randomHueSaturationValue_prob = 0.5
randomShiftScaleRotate_prob = 0.5
randomHorizontalFlip_prob = 0.5
randomVerticalFlip_prob = 0.5
randomRotate90_prob = 0.5

# augmentation properties
hue_shift_limit = (-30, 30)
sat_shift_limit = (-30, 30)
val_shift_limit = (-30, 30)

shift_limit=(-0.03, 0.03)
scale_limit=(-0.03, 0.03)
aspect_limit=(-0.03, 0.03)
rotate_limit=(-15, 15)

# dataset properties

# for optimization
# resize_to = (256, 256) # 384, 512, 640
# crop_size = 128  # 256, 512, 768

# for real run
resize_to = (512, 512) # 384, 512, 640
crop_size = 256  # 256, 512, 768

point_scale = 100
# todo: the problem is that it should be adjusted everytime according to point_scale and gaussian properties
point_threshold = 1
# todo: this is not efficient because in every dataset we need the length of wheats. In addition, scale the image problem
distance_threshold = [30, 25, 20, 15, 10, 5, 3, 1, 0] # original is 25
sigma = 1

# boolean
resize = True
crop = True
create_label_before_augmentation = False
gaussian = True

HueSaturationValue = True
ShiftScaleRotate = True
HorizontalFlip = True
VerticalFlip = True
Rotate90 = True

plot = False
plot_in_looper = True

# modeling parameters
# for Optimization
filters_child = [3, 4, 8, 16, 32, 64]
# for unets and resnet32 and resnet 18
filters_small = [3, 64, 128, 256, 512, 1024]
# for resnext and resnet50
filters_large = [3, 256, 512, 1024, 2048, 4096]
learning_rate = 2.5e-4
epochs = 2000
batch_size = 15

# testing
train_regression_epochs = 100
test_epochs = 100


# # # # # # # # # # # # # # # datasets paths # # # # # # # # # # # # # # #
# train
# Nregaug_Yola_Nopt
# train_images_path = '../../../../results/Nregaug_Yola_Nopt/train_dataset/images'
# train_actual_labels_path = '../../../../results/Nregaug_Yola_Nopt/train_dataset/actual_labels'
# Yregaug_Yola_Yopt
# train_images_path = '../../../../results/Yregaug_Yola_Yopt/train_dataset/images'
# train_actual_labels_path = '../../../../results/Yregaug_Yola_Yopt/train_dataset/actual_labels'
# During optimization
# train_images_path = '../../../../models/datasets/OLA_EA/outputs/augmented_images/images'
# train_actual_labels_path = '../../../../models/datasets/OLA_EA/outputs/augmented_images/actual_labels'

# valid
valid_images_path = '../../../../results/valid_test_datasets/valid_dataset/images'
valid_actual_labels_path = '../../../../results/valid_test_datasets/valid_dataset/actual_labels'

# test
test_images_path = '../../../../results/valid_test_datasets/test_dataset/images'
test_actual_labels_path = '../../../../results/valid_test_datasets/test_dataset/actual_labels'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# # # # # # # # # # # # # # # test results paths # # # # # # # # # # # # # #
# Nregaug_Yola_Nopt
# test_results_path = '../../../../results/Nregaug_Yola_Nopt/test_results'
# Yregaug_Yola_Yopt
test_results_path = '../../../../results/Yregaug_Yola_Yopt/test_results'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# # # # # # # # # # # # # checkpoints path # # # # # # # # # # # # # # # # #
# Nregaug_Yola_Nopt
# checkpoints_path = '../../../../results/Nregaug_Yola_Nopt/checkpoints'
# Yregaug_Yola_Yopt
checkpoints_path = '../../../../results/Yregaug_Yola_Yopt/checkpoints'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

