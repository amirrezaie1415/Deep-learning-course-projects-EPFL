#!/usr/bin/env python3

import torch
from NN import *
from helpers import *
from misc import * # for plot
import math # to use pi 
import copy
import argparse
import csv 
import os

torch.set_grad_enabled(False)  # disable the autograd

def main(config):
    # set the seed number for the purpose of reproducibility
    torch.manual_seed(config.seed) 
    np.random.seed(config.seed)
    for run_number in range(1, config.nb_run + 1):
        # 1- create the dataset
        X_test, X_train = torch.empty(1000, 2).uniform_(0,1), torch.empty(1000, 2).uniform_(0,1)

        # 1-1 create the target tensor
        y_test, y_train = label_data(X_test), label_data(X_train)

        # 1-2 normalize the design matrices (zero mean and standard deviation of one)
        mu, std = torch.mean(X_train, dim=0), torch.std(X_train, dim=0)
        X_train_norm = X_train.clone().sub_(mu).div_(std)
        X_test_norm = X_test.clone().sub_(mu).div_(std)

        # 2- create a network
        model = Sequential(
                           Linear(2, 25),
                           ReLU(),
                           Linear(25, 25),
                           ReLU(),
                           Linear(25, 25),
                           ReLU(),
                           Linear(25, 2),
                           )

            # 2-2 print the architecture and number of parameters 
        print('_______________________________\n')
        print('###___ model architecture ___ ### ')
        nb_bias_param = 0
        nb_weights_param = 0
        for key, value in model.layers.items():
            print(key + ': ' + str(value))
            nb_bias_param += value._bias.view(-1).size(0)
            nb_weights_param += value._weights.view(-1).size(0)
        print('_______________________________')
        print('number of weights: {}'.format(nb_weights_param))
        print('number of bias: {}'.format(nb_bias_param))
        print('total number of parameters: {}'.format(nb_weights_param + nb_bias_param))
        print('_______________________________\n')

            # 2-3 optimization scheme 
        # set the learning rate
        model.optimizer['eta'] = config.lr
        # define the loss function
        criterion = LossMSE()
        # total number of epochs
        nb_epochs = config.epochs
        # batch size
        batch_size = config.batch_size
        # select the criteria based on which the best model is selected. This can take two values:
            # 'loss': in this cases the best model is selected based on the minimum test loss
            # 'accuracy': in this cases the best model is selected based on the maximum test accuracy
        save_criterion = 'loss' 

        # 3- training and testing
        nb_train = X_train_norm.size(0)  # number of training samples
        nb_test = X_test_norm.size(0)  # number of test samples

        # empty lists to hold values of loss and accuracy for train and test data
        loss_train = []
        loss_test = []
        acc_train = []
        acc_test = []

        minimum_test_loss = 1e10
        max_test_acc = 0

        print('learning rate: %.1e, number of epochs: %d, batch size: %d' %(model.optimizer['eta'], nb_epochs, batch_size))
        for epoch in range(nb_epochs):
            loss_train_batch = 0
            loss_test_batch = 0
            pred_label_train = torch.empty(nb_train)  # initialize the predicted train lables
            pred_label_test = torch.empty(nb_test)  # initialize the predicted test lables
            # 3-1 loop over training samples
            length = 0
            for indices, X_train_batch, y_train_batch in get_mini_batch(X_train_norm, y_train, batch_size):
                length += 1 
                # forward pass
                pred = model(X_train_batch)
                # compute loss
                loss_train_batch += criterion(y_train_batch, pred).item()
                # compute labels
                pred_label_train[indices] = pred.argmax(dim=1).view(-1).float()
                # gradient step and backward pass
                model.grad(y_train_batch)
                model.backward()
            loss_train.append(loss_train_batch / length)
            acc_train.append(accuracy(pred_label_train, y_train.argmax(dim=1)))

            # 3-2 loop over test samples
            pred = model(X_test_norm.view(nb_test, 2, 1))
            loss_test.append(criterion(y_test.view(nb_test, 2, -1), pred).item())
            pred_label_test = pred.argmax(dim=1).view(-1).float()
            acc_test.append(accuracy(pred_label_test, y_test.argmax(dim=1)))

            # copy the best model.
            if save_criterion == 'loss':
                if loss_test[epoch] < minimum_test_loss:
                    model_selected = copy.deepcopy(model)
                    minimum_test_loss = loss_test[epoch]

            if save_criterion == 'accuracy':
                if acc_test[epoch] > max_test_acc:
                    model_selected = copy.deepcopy(model)
                    max_test_acc = acc_test[epoch]

            print('Epoch: [%.d/%.d],  train loss: %.3f, test loss: %.3f, train acc.: %.1f, test acc.: %.1f' % (
                        epoch+1,nb_epochs, loss_train[epoch], loss_test[epoch], acc_train[epoch], acc_test[epoch]))
            result_csv_path = '../logs/'
            if not os.path.exists(result_csv_path):
                os.makedirs(result_csv_path)
            if run_number==1 and epoch == 0:
                f = open(os.path.join(result_csv_path, '%s-%.8f-%d-%d.csv' % (model.layers['act_1']._name, model.optimizer['eta'], nb_epochs, batch_size)), 'w', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow(['Activation func.', 'learning_rate','batch_size','epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'run_number'])
                f.close()
            f = open(os.path.join(result_csv_path, '%s-%.8f-%d-%d.csv' % (model.layers['act_1']._name, model.optimizer['eta'], nb_epochs, batch_size)), 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow(
                [model.layers['act_1']._name, model.optimizer['eta'], batch_size, epoch+1, loss_train[epoch], loss_test[epoch], acc_train[epoch], acc_test[epoch], run_number])
            f.close()
        # 4- performance of the selected model over training and test data
        pred_train, pred_label_train = pred_label(model=model_selected, X_normalized=X_train_norm)
        pred_test, pred_label_test = pred_label(model=model_selected, X_normalized=X_test_norm)
            # 4-1 train data
        nb_train = X_train.size(0)
        pred = model_selected(X_train_norm.view(nb_train, 2, 1))
        loss_train_selected_model = criterion(y_train.view(nb_train, 2, -1), pred).item()
        pred_label_train_selected_model = pred.argmax(dim=1).view(-1).float()
        acc_train_selected_model = accuracy(pred_label_train_selected_model, y_train.argmax(dim=1))
            # 4-2 test data
        nb_test = X_test.size(0)
        pred = model_selected(X_test_norm.view(nb_test, 2, 1))
        loss_test_selected_model = criterion(y_test.view(nb_test, 2, -1), pred).item()
        pred_label_test_selected_model = pred.argmax(dim=1).view(-1).float()
        acc_test_selected_model = accuracy(pred_label_test_selected_model, y_test.argmax(dim=1))

        print('\nPerformance of the selected model:')
        print('--> train loss: %.3f, test loss: %.3f, train acc.: %.1f, test acc.: %.1f' % (
                 loss_train_selected_model, loss_test_selected_model, acc_train_selected_model, acc_test_selected_model))

        # plots
        save_plot = bool(config.save_plot)
        if save_plot:
            print('[INFO] Saving plots ...')
            # plot loss
            plot_loss(loss_train, loss_test, save_plot)
                # plot accuracy
            plot_acc(acc_train, acc_test, save_plot)

                # plot class labels
                    # train data
            plot_classes(X_train, pred_label_train_selected_model, 'train data - predicted labels', save_plot)
            plot_classes(X_train, y_train.argmax(dim=1).numpy(), 'train data - true labels', save_plot)
                    # test data
            plot_classes(X_test, pred_label_test_selected_model, 'test data - predicted labels', save_plot)
            plot_classes(X_test, y_test.argmax(dim=1).numpy(), 'test data - true labels', save_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='seed number')
    parser.add_argument('--nb_run', type=int, default=1, help='number of training models')
    parser.add_argument('--save_plot', type=int, default=0, help='a flag to save plot')
    config = parser.parse_args()
    main(config)
