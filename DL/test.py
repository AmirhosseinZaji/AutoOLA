from test_data import ImageFolder
import torch
import numpy as np
from matplotlib import pyplot
from test_looper import Looper
import properties
import sys
sys.path.append('models')
import time
import os
import shutil
from pathlib import Path
from scipy.io import savemat
os.environ['TORCH_HOME'] = os.path.join("..", os.path.curdir)

from models.my_unet import UNet
# from models.my_unet_small import UNet

def test():

    """
    Here train means the validation dataset and valid means the testing dataset.

    We have 105 samples that were not used in the training procedure.
    Here, we used the first 40 of them for validation and the rest of them for testing.

    Validation is used to adjust the regression problem. This adjustment can be an average of results
    of regression based and localization based results, or just a simple shift of the results.

    Then, the testing results are generated.
    """

    folder_path = properties.checkpoints_path
    checkpoint_name = '1999_complete'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = {}     # training and validation
    dataloader = {}  # training and validation

    for mode in ['train', 'valid']:

        # dataset[mode] = ImageFolder(mode,
        #                             properties.randomHueSaturationValue_prob if mode == 'train' else 0,
        #                             properties.randomShiftScaleRotate_prob if mode == 'train' else 0,
        #                             properties.randomHorizontalFlip_prob if mode == 'train' else 0,
        #                             properties.randomVerticalFlip_prob if mode == 'train' else 0,
        #                             properties.randomRotate90_prob if mode == 'train' else 0)

        dataset[mode] = ImageFolder(mode, 0, 0, 0, 0, 0)

        dataloader[mode] = torch.utils.data.DataLoader(dataset[mode],
                                                       batch_size=properties.batch_size)



    # initialize a model based on chosen network_architecture
    network = UNet().to(device)

    network = torch.nn.DataParallel(network)
    if torch.cuda.is_available():
        checkpoint_loaded = torch.load(os.path.join(folder_path, checkpoint_name+'.pth'))
    else:
        checkpoint_loaded = torch.load(os.path.join(folder_path, checkpoint_name+'.pth'), map_location=torch.device('cpu'))

    network.load_state_dict(checkpoint_loaded['state_dict'])

    # print the number of parameters in the model
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of trainable parameters: {params}")

    # initialize the optimizer
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=properties.learning_rate,
                                 weight_decay=1e-5)

    loss = torch.nn.MSELoss()

    '''The following folders and arrays are created in order to save the models and its statistics'''
    # making the folders
    results_dist = os.path.join(properties.test_results_path, 'test_checkpoint_'+checkpoint_name)
    if os.path.exists(results_dist):
        shutil.rmtree(results_dist)
    Path(results_dist).mkdir(parents=True, exist_ok=True)


    for mode in ['train', 'valid']:
        # initialize the arrays
        tp = np.array([])
        fp = np.array([])
        fn = np.array([])
        precision = np.array([])
        recall = np.array([])
        f1 = np.array([])
        rmse = np.array([])
        mae = np.array([])
        r2 = np.array([])
        mape = np.array([])
        count_true = np.array([])
        count_regression = np.array([])
        count_localization = np.array([])
        count_true_total = np.array([])
        count_regression_total = np.array([])
        count_localization_total = np.array([])

        if mode == 'train':
            temp_path = os.path.join(results_dist, 'validation_arrays')
        else:
            temp_path = os.path.join(results_dist, 'testing_arrays')

        Path(temp_path).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(temp_path, 'tp'), tp)
        np.save(os.path.join(temp_path, 'fp'), fp)
        np.save(os.path.join(temp_path, 'fn'), fn)
        np.save(os.path.join(temp_path, 'precision'), precision)
        np.save(os.path.join(temp_path, 'recall'), recall)
        np.save(os.path.join(temp_path, 'f1'), f1)
        np.save(os.path.join(temp_path, 'rmse'), rmse)
        np.save(os.path.join(temp_path, 'mae'), mae)
        np.save(os.path.join(temp_path, 'r2'), r2)
        np.save(os.path.join(temp_path, 'mape'), mape)
        np.save(os.path.join(temp_path, 'count_true'), count_true)
        np.save(os.path.join(temp_path, 'count_regression'), count_regression)
        np.save(os.path.join(temp_path, 'count_localization'), count_localization)
        np.save(os.path.join(temp_path, 'count_true_total'), count_true_total)
        np.save(os.path.join(temp_path, 'count_regression_total'), count_regression_total)
        np.save(os.path.join(temp_path, 'count_localization_total'), count_localization_total)


        if mode == 'train':
            testing=False
        else:
            testing=True

        # create testing Looper to handle a single epoch
        test_looper = Looper(network,
                             device,
                             loss,
                             optimizer,
                             dataloader[mode],
                             len(dataset[mode]),
                             results_dist,
                             testing,
                             validation=True)
        if mode == 'train':
            Range = range(properties.train_regression_epochs)
        else:
            Range = range(properties.test_epochs)

        for epoch in Range:
            print(f"Epoch {epoch}\n")
            # run test epoch
            with torch.no_grad():
                result_mape_valid, result_f1_valid = test_looper.run(epoch)
            print("\n", "-"*80, "\n", sep='')

        # save the arrays in matlab format
        current_dir = Path.cwd()
        os.chdir(temp_path)
        count_localization = np.load('count_localization.npy')
        count_regression = np.load('count_regression.npy')
        count_true = np.load('count_true.npy')
        count_localization_total = np.load('count_localization_total.npy')
        count_regression_total = np.load('count_regression_total.npy')
        count_true_total = np.load('count_true_total.npy')
        f1 = np.load('f1.npy')
        precision = np.load('precision.npy')
        recall = np.load('recall.npy')
        fn = np.load('fn.npy')
        fp = np.load('fp.npy')
        tp = np.load('tp.npy')
        mae = np.load('mae.npy')
        r2 = np.load('r2.npy')
        rmse = np.load('rmse.npy')
        mape = np.load('mape.npy')
        os.chdir(current_dir)


        # reshapes
        count_localization = count_localization[np.newaxis].T
        count_regression = count_regression[np.newaxis].T
        count_true = count_true[np.newaxis].T
        count_localization_total = count_localization_total[np.newaxis].T
        count_regression_total = count_regression_total[np.newaxis].T
        count_true_total = count_true_total[np.newaxis].T
        mae = mae[np.newaxis].T
        r2 = r2[np.newaxis].T
        rmse = rmse[np.newaxis].T
        mape = mape[np.newaxis].T

        f1 = f1.reshape(-1, 9)
        recall = recall.reshape(-1, 9)
        precision = precision.reshape(-1, 9)
        fn = fn.reshape(-1, 9)
        fp = fp.reshape(-1, 9)
        tp = tp.reshape(-1, 9)

        mdict = {'count_localization': count_localization,
                 'count_regression': count_regression,
                 'count_true': count_true,
                 'count_localization_total': count_localization_total,
                 'count_regression_total': count_regression_total,
                 'count_true_total': count_true_total,
                 'f1': f1,
                 'precision': precision,
                 'recall': recall,
                 'fn': fn,
                 'fp': fp,
                 'tp': tp,
                 'mae': mae,
                 'r2': r2,
                 'rmse': rmse,
                 'mape': mape}

        savemat(os.path.join(temp_path, 'arrays.mat'), mdict)


    print('end')

if __name__ == '__main__':
    test()