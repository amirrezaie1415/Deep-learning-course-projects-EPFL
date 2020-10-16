import torch
# import necessary modules for plotting
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import os

font = {'family' : 'sans-serif',
        'size'   : 18}
matplotlib.rc('font', **font)

    
def plot_classes(X, pred_label, data_split, save_plot=False):
    """
    This function plots two graphs:
        - plot of data points color coded based on their true labels
        - plot of data color coded based on predicted class labels
    in both plots, the Euclidean space is divided into 2 regions highlighted 
    with different colors showing the true class labels.
     
    INPUT:
        X = design tensor of the size (N, D), where N is the number of samples
        and D is the number of features.
        
        pred_label = predicted labels {0, 1}
        data_split = a string to be shown as the title of plot
    
    OUTPUT:
        a plot
    """
    # generate a grid of data points based on which the space is colored into two regions. 
    xv, yv = torch.meshgrid([(torch.arange(0,500)).float()/500, (torch.arange(0,500)).float()/500])
    grid = torch.cat((xv.reshape(-1).view(-1,1), yv.reshape(-1).view(-1,1)), 1)
    grid_norm = torch.norm(grid, dim=1)
    label_grid =  1. - ((grid_norm > 1/torch.sqrt(2 * torch.tensor(math.pi))) & (grid_norm < 1.)).float()
    
    
    data_np = X.numpy()
    data_np_0 = data_np[np.where(pred_label == 0)]
    data_np_1 = data_np[np.where(pred_label == 1)]

    plt.figure(figsize=(6,6))
    plt.pcolormesh(xv.numpy(), yv.numpy(), label_grid.numpy().reshape(xv.numpy().shape), cmap=plt.cm.bwr, vmin=-1, vmax=2)
    plt.scatter(data_np_0[:,0], data_np_0[:,1], label='class - 0', color='b')
    plt.scatter(data_np_1[:,0], data_np_1[:,1], label='class - 1', color='r')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    legend_x = 1
    legend_y = 0.9
    plt.legend( loc='center left', bbox_to_anchor=(legend_x, legend_y))
    plt.title(data_split)
    if save_plot:
        save_dir = '../plots/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + data_split + '.png')
    plt.show()
    
def plot_loss(loss_train, loss_test, save_plot=False):
    plt.figure(figsize=(6,6))
    plt.plot(range(len(loss_train)), loss_train, label='train', color='r')
    plt.plot(range(len(loss_test)), loss_test, label='test', color='b')
    plt.ylim([0,1])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save_plot:
        save_dir = '../plots/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + 'loss.pdf')
    plt.show()
    
def plot_acc(acc_train, acc_test, save_plot=False):
    plt.figure(figsize=(6,6))
    plt.plot(range(len(acc_train)), acc_train, label='train', color='r')
    plt.plot(range(len(acc_test)), acc_test, label='test', color='b')
    plt.ylim([0,100])
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if save_plot:
        save_dir = '../plots/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + 'accuracy.pdf')
    plt.show()
