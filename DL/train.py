from data import ImageFolder
import torch
import numpy as np
from matplotlib import pyplot
from looper import Looper
import properties
import sys
sys.path.append('models')
import time
import os
import shutil

def train():
    os.environ['TORCH_HOME'] = os.path.join("..", os.path.curdir)

    from models.my_unet import UNet
    # from models.my_unet_small import UNet

    start_time = time.time()

    if torch.cuda.is_available():
        DEVICE = torch.cuda.current_device ()
        print (DEVICE)
        print (torch.cuda.device_count ())
        print (torch.cuda.get_device_name (DEVICE))
        print (torch.cuda.is_available ())
        device = torch.device('cuda:0')
    else:
        torch.cuda.is_available = lambda: False
        device = torch.device ('cpu')


    dataset = {}     # training and validation
    dataloader = {}  # training and validation

    for mode in ['train', 'valid']:

        dataset[mode] = ImageFolder(mode,
                                    properties.randomHueSaturationValue_prob if mode == 'train' else 0,
                                    properties.randomShiftScaleRotate_prob if mode == 'train' else 0,
                                    properties.randomHorizontalFlip_prob if mode == 'train' else 0,
                                    properties.randomVerticalFlip_prob if mode == 'train' else 0,
                                    properties.randomRotate90_prob if mode == 'train' else 0)

        dataloader[mode] = torch.utils.data.DataLoader(dataset[mode],
                                                       batch_size=properties.batch_size)

    # specify the number of input channels
    input_channels = properties.input_channels

    # initialize a model based on chosen network_architecture
    network = UNet().to(device)

    network = torch.nn.DataParallel(network)

    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of trainable parameters: {params}")

    # initialize loss, optimized and learning rate scheduler
    loss = torch.nn.MSELoss()
    # loss = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=properties.learning_rate,
                                 weight_decay=1e-5)
    # optimizer = torch.optim.SGD(network.parameters(),
    #                             lr=learning_rate,
    #                             momentum=0.9,
    #                             weight_decay=1e-5)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=400,
                                                   gamma=0.1)

    # if plot flag is on, create a live plot (to be updated by Looper)
    if properties.plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2


    '''The following folders and arrays are created in order to save the models and its statistics'''
    # making the folders
    results_dist = os.path.join(os.curdir, 'results')
    if os.path.exists(results_dist):
        shutil.rmtree(results_dist)
    os.mkdir(results_dist)

    # for the arrays
    arrays_dist = os.path.join(results_dist, 'arrays')
    os.mkdir(arrays_dist)

    arrays_dist_train = os.path.join(arrays_dist, 'train')
    arrays_dist_valid = os.path.join(arrays_dist, 'valid')
    os.mkdir(arrays_dist_train)
    os.mkdir(arrays_dist_valid)

    # for the checkpoints
    checkpoints_dist = os.path.join(results_dist, 'checkpoints')
    os.mkdir(checkpoints_dist)

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
    epoch_time = np.array([])

    np.save(os.path.join(arrays_dist_train, 'tp'), tp)
    np.save(os.path.join(arrays_dist_valid, 'tp'), tp)
    np.save(os.path.join(arrays_dist_train, 'fp'), fp)
    np.save(os.path.join(arrays_dist_valid, 'fp'), fp)
    np.save(os.path.join(arrays_dist_train, 'fn'), fn)
    np.save(os.path.join(arrays_dist_valid, 'fn'), fn)
    np.save(os.path.join(arrays_dist_train, 'precision'), precision)
    np.save(os.path.join(arrays_dist_valid, 'precision'), precision)
    np.save(os.path.join(arrays_dist_train, 'recall'), recall)
    np.save(os.path.join(arrays_dist_valid, 'recall'), recall)
    np.save(os.path.join(arrays_dist_train, 'f1'), f1)
    np.save(os.path.join(arrays_dist_valid, 'f1'), f1)
    np.save(os.path.join(arrays_dist_train, 'rmse'), rmse)
    np.save(os.path.join(arrays_dist_valid, 'rmse'), rmse)
    np.save(os.path.join(arrays_dist_train, 'mae'), mae)
    np.save(os.path.join(arrays_dist_valid, 'mae'), mae)
    np.save(os.path.join(arrays_dist_train, 'r2'), r2)
    np.save(os.path.join(arrays_dist_valid, 'r2'), r2)
    np.save(os.path.join(arrays_dist_train, 'mape'), mape)
    np.save(os.path.join(arrays_dist_valid, 'mape'), mape)
    np.save(os.path.join(arrays_dist, 'epoch_time'), epoch_time)

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(network,
                          device,
                          loss,
                          optimizer,
                          dataloader['train'],
                          len(dataset['train']),
                          arrays_dist_train)
    valid_looper = Looper(network,
                          device,
                          loss,
                          optimizer,
                          dataloader['valid'],
                          len(dataset['valid']),
                          arrays_dist_valid,
                          validation=True)

    # current best results (lowest mean absolute error on validation set)
    current_best_mape_train = np.infty
    current_best_mape_valid = np.infty
    current_best_rmse_train = np.infty
    current_best_rmse_valid = np.infty
    current_best_f1_train = np.zeros(len(properties.distance_threshold))
    current_best_f1_valid = np.zeros(len(properties.distance_threshold))

    for epoch in range(properties.epochs):
        print(f"Epoch {epoch}\n")

        # run training epoch and update learning rate
        result_rmse_train, result_mape_train, result_f1_train = train_looper.run()
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result_rmse_valid, result_mape_valid, result_f1_valid = valid_looper.run()


        # WE DON'T NEED TO SAVE THE CHECKPOINTS IN OPTIMIZATION
        # update checkpoint
        # every 50 epochs, the optimizer is also saved in order to be able to continue running the model
        if (epoch+1)%50==0:
            state = {
                'epoch' : epoch,
                'state_dict' : network.state_dict(),
                'optimizer' : optimizer.state_dict()
            }
            torch.save(state, os.path.join(checkpoints_dist, f'{epoch}_complete.pth'))

        # save the checkpoints

        if result_mape_train < current_best_mape_train:
            torch.save(network.state_dict(), os.path.join(checkpoints_dist, f'{epoch}.pth'))
        
        elif result_mape_valid < current_best_mape_valid:
            torch.save(network.state_dict(), os.path.join(checkpoints_dist, f'{epoch}.pth'))
        
        else:
            for i in range(len(properties.distance_threshold)):
                if result_f1_train[i] > current_best_f1_train[i]:
                    torch.save(network.state_dict(), os.path.join(checkpoints_dist, f'{epoch}.pth'))
                    break
        
                if result_f1_valid[i] > current_best_f1_valid[i]:
                    torch.save(network.state_dict(), os.path.join(checkpoints_dist, f'{epoch}.pth'))
                    break


        # Updating the current bests
        if result_mape_train < current_best_mape_train:
            current_best_mape_train = result_mape_train
            print(f"Train > New best MAPE: {current_best_mape_train}")

        if result_mape_valid < current_best_mape_valid:
            current_best_mape_valid = result_mape_valid
            print(f"Valid > New best MAPE: {current_best_mape_valid}")

        if result_rmse_train < current_best_rmse_train:
            current_best_rmse_train = result_rmse_train

        if result_rmse_valid < current_best_rmse_valid:
            current_best_rmse_valid = result_rmse_valid

        for i in range(len(properties.distance_threshold)):
            if result_f1_train[i] > current_best_f1_train[i]:
                current_best_f1_train[i] = result_f1_train[i]
                print(f"Train > New best F1 at {properties.distance_threshold[i]}: {current_best_f1_train[i]}")

            if result_f1_valid[i] > current_best_f1_valid[i]:
                current_best_f1_valid[i] = result_f1_valid[i]
                print(f"Valid > New best F1 at {properties.distance_threshold[i]}: {current_best_f1_valid[i]}")




        epoch_time = time.time() - start_time
        np.save(os.path.join(arrays_dist, 'epoch_time'),  np.append(np.load(os.path.join(arrays_dist, 'epoch_time.npy')), epoch_time))

        print("\n", "-"*80, "\n", sep='')

    print(f"[Training done] Train > Best MAPE: {current_best_mape_train}")
    print(f"[Training done] Train > Best F1: {current_best_f1_train}")
    print(f"[Training done] Valid > Best MAPE: {current_best_mape_valid}")
    print(f"[Training done] Valid > Best F1: {current_best_f1_valid}")
    print(time.time() - start_time)

        # counter for termination for optimization
        # if epoch == 0:
        #     rmse_termination_count = 0

        # if rmse_termination_count == 0:
        #     best_rmse_previous_epoch = current_best_rmse_valid

        # if current_best_rmse_valid == best_rmse_previous_epoch:
        #     rmse_termination_count += 1
        # else:
        #     rmse_termination_count = 0

        # if rmse_termination_count == 5:
        #     print (f"During Iteration Best Valid RMSE: {current_best_rmse_valid}")
        #     return current_best_rmse_valid
    return current_best_rmse_valid

if __name__ == '__main__':
    train()
