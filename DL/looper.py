"""Looper implementation."""
from typing import Optional, List
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import properties
import sklearn.metrics as skm
import os

class Looper():

    def __init__(self,
                 network: torch.nn.Module,
                 device: torch.device,
                 loss: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_size: int,
                 array_dist,
                 validation: bool=False):

        self.network = network
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.loader = data_loader
        self.size = dataset_size
        self.validation = validation
        self.array_dist = array_dist
        self.running_loss = []
    ##############################################################################################################
    def run(self):
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values_regression = []
        self.tps = []
        self.fps = []
        self.fns = []
        self.running_loss.append(0)

        # set a proper mode: train or eval
        self.network.train(not self.validation)

        for image, label, actual_label in self.loader:
            # move images and labels to given device
            image = image.to(self.device)
            label = label.to(self.device)

            # clear accumulated gradient if in train mode
            if not self.validation:
                self.optimizer.zero_grad()

            # get model prediction (a density map)
            result = self.network(image)

            # some interesting plots
            if self.validation and properties.plot_in_looper:
                temp_image = image.cpu().data.numpy()
                temp_image = temp_image[0]
                temp_image = temp_image.transpose(1, 2, 0)
                temp_label = label.cpu().data.numpy()
                temp_label = temp_label[0][0]
                temp_output = result.cpu().data.numpy()
                temp_output = temp_output[0][0]
                plt.subplot(311)
                plt.imshow(temp_image)
                plt.subplot(312)
                plt.imshow(temp_label)
                plt.subplot(313)
                plt.imshow(temp_output)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

            # calculate loss and update running loss
            loss = self.loss(result, label)
            self.running_loss[-1] += image.shape[0] * loss.item() / self.size

            # update weights if in train mode
            if not self.validation:
                loss.backward()
                self.optimizer.step()

            # loop over batch samples
            for true, actual_true, predicted in zip(label, actual_label, result):
                true_counts = torch.sum(true).item() / properties.point_scale
                predicted_counts_regression = torch.sum(predicted).item() / properties.point_scale
                tp, fp, fn = self.evaluator(predicted,
                                            actual_true[0],
                                            properties.point_threshold,
                                            properties.distance_threshold)

                # update current epoch results
                self.true_values.append(true_counts)
                self.predicted_values_regression.append(predicted_counts_regression)
                self.tps.append(list(tp))
                self.fps.append(list(fp))
                self.fns.append(list(fn))


        self.update_errors()

        self.log()

        return self.rmse, self.mape, self.f1
    ##############################################################################################################
    def evaluator(self, predicted, actual_true, point_threshold, distance_threshold):
        ########################################################################
        def get_gt_features(actual_label):
            actual_true_points = []
            temp = np.where(actual_label == properties.point_scale)
            temp_a = temp[0]
            temp_b = temp[1]
            for i in range(len(temp_a)):
                actual_true_points.append((temp_a[i], temp_b[i]))
            return actual_true_points
        ########################################################################
        def get_distance(first, second):
            first = torch.tensor(first, dtype=torch.float16)
            second = torch.tensor(second, dtype=torch.float16)

            distance = torch.zeros(len(first))
            for i in range(len(distance)):
                minimum = 100
                for j in range(len(second)):

                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    first_param = first[i]
                    second_param = second[j]
                    first_param = first_param.to(device)
                    second_param = second_param.to(device)
                    # temp_distance = torch.dist(first_param, second_param)
                    a = first_param.cpu ().detach ().numpy ()
                    b = second_param.cpu ().detach ().numpy ()
                    temp_distance = np.linalg.norm (a - b)
                    # temp_distance = torch.dist(first[i], second[j])
                    if temp_distance < minimum:
                        minimum = temp_distance
                        minimum = minimum.item()
                distance[i] = minimum
            return distance
        ########################################################################
        # todo: talk about the new way that we handled the nms
        def nms(predicted, threshold):
            matrix = predicted.detach().clone()
            matrix = matrix.cpu().numpy()
            matrix = matrix[0]
            points = []
            potential_points = np.where(matrix > threshold)
            x = potential_points[0]
            y = potential_points[1]

            badindex = np.where((x == 0) |
                                (x >= (matrix.shape[0]-1)) |
                                (y == 0) |
                                (y >= (matrix.shape[1]-1)))

            x = np.delete(x, badindex)
            y = np.delete(y, badindex)

            for i in range(len(x)):
                m = matrix[x[i], y[i]]
                m1 = matrix[x[i]-1, y[i]]
                m2 = matrix[x[i]+1, y[i]]
                m3 = matrix[x[i], y[i]-1]
                m4 = matrix[x[i], y[i]+1]
                if m>m1 and m>m2 and m>m3 and m>m4:
                    points.append((x[i], y[i]))

            # output = matrix[tuple(np.array(points).T)]

            return points
        ########################################################################
        predicted_points = nms(predicted, point_threshold)
        actual_true_points = get_gt_features(actual_true)

        if len(predicted_points) == 0 or len(actual_true_points) == 0:
            if len(predicted_points) == len(actual_true_points):
                tp, fp, fn = np.zeros(len(distance_threshold)), \
                             np.zeros(len(distance_threshold)), \
                             np.zeros(len(distance_threshold))
                return tp, fp, fn
            elif len(predicted_points) == 0:
                tp, fp, fn = np.zeros(len(distance_threshold)), \
                             np.zeros(len(distance_threshold)), \
                             np.zeros(len(distance_threshold)) + len(actual_true_points)
                return tp, fp, fn
            else:
                tp, fp, fn = np.zeros(len(distance_threshold)), \
                             np.zeros(len(distance_threshold)) + len(predicted_points), \
                             np.zeros(len(distance_threshold))
                return tp, fp, fn

        distance_pred_true = get_distance(predicted_points, actual_true_points)
        distance_true_pred = get_distance(actual_true_points, predicted_points)

        # todo: Here, the number of samples outside that distance_threshold is maybe considerable
        tp = np.empty(len(distance_threshold))
        fp = np.empty(len(distance_threshold))
        fn = np.empty(len(distance_threshold))
        for iii in range(len(distance_threshold)):
            result_pred_true = distance_pred_true.le(distance_threshold[iii])
            result_true_pred = distance_true_pred.le(distance_threshold[iii])

            tp[iii] = result_pred_true.sum()
            fp[iii] = (~result_pred_true).sum()
            fn[iii] = (~result_true_pred).sum()

        return tp, fp, fn
    ##############################################################################################################
    def update_errors(self):
        ##########################################################################################################
        def calculateF1(tps, fps, fns):

            tp = sum(tps)
            fp = sum(fps)
            fn = sum(fns)

            precision = 0.0
            recall = 0.0
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            f1 = 0.0
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)

            return precision, recall, f1
        ##########################################################################################################
        self.rmse = skm.mean_squared_error(self.true_values, self.predicted_values_regression) ** 0.5
        self.mae = skm.mean_absolute_error(self.true_values, self.predicted_values_regression)
        self.r2 = np.corrcoef(self.true_values, self.predicted_values_regression)[0, 1] ** 2
        self.mape = np.mean(np.abs((np.array(self.true_values) - np.array(self.predicted_values_regression))/
                   np.array(self.true_values))) * 100

        self.precision = np.empty(len(self.tps[0]))
        self.recall = np.empty(len(self.tps[0]))
        self.f1 = np.empty(len(self.tps[0]))
        self.tps = np.array(self.tps)
        self.fps = np.array(self.fps)
        self.fns = np.array(self.fns)
        for jjj in range(len(self.tps[0])):
            self.precision[jjj], self.recall[jjj], self.f1[jjj] = calculateF1(self.tps[:, jjj],
                                                                              self.fps[:, jjj],
                                                                              self.fns[:, jjj])
        ##############################################################################################################
        # save statistics
        np.save(os.path.join(self.array_dist, 'tp'),  np.append(np.load(os.path.join(self.array_dist, 'tp.npy')), sum(self.tps)))
        np.save(os.path.join(self.array_dist, 'fp'),  np.append(np.load(os.path.join(self.array_dist, 'fp.npy')), sum(self.fps)))
        np.save(os.path.join(self.array_dist, 'fn'),  np.append(np.load(os.path.join(self.array_dist, 'fn.npy')), sum(self.fns)))
        np.save(os.path.join(self.array_dist, 'rmse'),  np.append(np.load(os.path.join(self.array_dist, 'rmse.npy')), self.rmse))
        np.save(os.path.join(self.array_dist, 'mae'),  np.append(np.load(os.path.join(self.array_dist, 'mae.npy')), self.mae))
        np.save(os.path.join(self.array_dist, 'r2'),  np.append(np.load(os.path.join(self.array_dist, 'r2.npy')), self.r2))
        np.save(os.path.join(self.array_dist, 'mape'),  np.append(np.load(os.path.join(self.array_dist, 'mape.npy')), self.mape))
        np.save(os.path.join(self.array_dist, 'precision'),  np.append(np.load(os.path.join(self.array_dist, 'precision.npy')), self.precision)) # In evaluation, I should reshape these three with a.reshape(-1, 9)
        np.save(os.path.join(self.array_dist, 'recall'),  np.append(np.load(os.path.join(self.array_dist, 'recall.npy')), self.recall))
        np.save(os.path.join(self.array_dist, 'f1'),  np.append(np.load(os.path.join(self.array_dist, 'f1.npy')), self.f1))

        # print("tp")
        # print(np.load(os.path.join(self.array_dist, 'tp.npy')))
        # print('fp')
        # print(np.load(os.path.join(self.array_dist, 'fp.npy')))
        # print('fn')
        # print(np.load(os.path.join(self.array_dist, 'fn.npy')))
        # print('rmse')
        # print(np.load(os.path.join(self.array_dist, 'rmse.npy')))
        # print('mae')
        # print(np.load(os.path.join(self.array_dist, 'mae.npy')))
        # print('r2')
        # print(np.load(os.path.join(self.array_dist, 'r2.npy')))
        # print('mape')
        # print(np.load(os.path.join(self.array_dist, 'mape.npy')))
        # print('precision')
        # print(np.load(os.path.join(self.array_dist, 'precision.npy')))
        # print('recall')
        # print(np.load(os.path.join(self.array_dist, 'recall.npy')))
        # print('f1')
        # print(np.load(os.path.join(self.array_dist, 'f1.npy')))

    ##############################################################################################################
    def log(self):
        """Print current epoch results."""
        print(f"{'Train' if not self.validation else 'Valid'}:\n"
              f"\tMAE: {self.mae:3.3f}\n"
              f"\tRMSE: {self.rmse:3.3f}\n"
              f"\tMAPE: {self.mape:3.2f}%\n"
              f"\tR2: {self.r2:3.2f}\n"
              f"\tPrecision: {self.precision}\n"
              f"\tRecall: {self.recall}\n"
              f"\tF1: {self.f1}\n"
              f"\tTP: {sum(self.tps)}\n"
              f"\tFP: {sum(self.fps)}\n"
              f"\tFN: {sum(self.fns)}\n"
              )

