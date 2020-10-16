#!/usr/bin/env python3

import torch
from torch import nn
import dlc_practical_prologue as prologue
from helpers import *
from networks import *
import csv 
import os

torch.manual_seed(42) # set the seed number for the sake of reproducibility

nb_run = 10  # number of training models
accuracies = nb_run * [None]

# set to True to use weight sharing
weight_sharing = True
for run_number in range(nb_run):
    print('-'*5 + 'run #:%d'%(run_number+1))
    # load data
    N = 1000  # number of train and test data
    (train_input, train_target, train_classes,
    test_input, test_target, test_classes) = prologue.generate_pair_sets(N)

    # change the image range to [0, 1]
    train_input.div_(255);
    test_input.div_(255);

    # normalize data using the mean and std of MNIST dataset (taken from slide 4/13 of the lecture 8-5-dataloader)
    train_input.sub_(0.1302).div_(0.3069);
    test_input.sub_(0.1302).div_(0.3069);

    # define model and optimization scheme
    # use either the ConvNet or WeightSharing class
    nb_channels = 32
    auxiliary_loss = True
    model = WeightSharing(nb_channels, auxiliary_loss) if weight_sharing else ConvNet(nb_channels, auxiliary_loss)
    if run_number==0:
        print(model)
    lr = 0.005  # learning rate
    weight_decay = 0  # weight decay
    nb_epochs = 150 # number of epochs
    batch_size = 50  # batch size
    weight_loss = [1., 1.] # weight associated with the main loss (w_{m} in the report) = weight loss[0],
                           # weight associated with the auxiliary loss (w_{a} in the report) = weight loss[1]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    criterion1 = nn.BCEWithLogitsLoss(reduction='mean')  # criteria associated with the main target
    criterion2 = nn.CrossEntropyLoss(reduction='mean')  # criteria associated with the auxiliary target (class of first image)
    criterion3 = nn.CrossEntropyLoss(reduction='mean')  # criteria associated with the auxiliary target (class of second image)
    
    # initialize train/test loss and accuracy
    train_loss = torch.empty(nb_epochs)
    train_acc = torch.empty(nb_epochs)
    test_loss = torch.empty(nb_epochs)
    test_acc = torch.empty(nb_epochs)

    for epoch in range(nb_epochs):
        model.train(True)
        loss_batch = 0
        acc_batch = 0
        batch_nb = 0
        for indices_batch in get_batch_indices(X=train_input, batch_size=batch_size):
            batch_nb += 1
            # forward pass
            output1, output2, output3 = model(train_input[indices_batch])  
            # compute Loss
            loss_goal = criterion1(output1, train_target[indices_batch].view(-1, 1).float())
            loss_aux1 = criterion2(output2, train_classes[indices_batch, 0].long())
            loss_aux2 = criterion3(output3, train_classes[indices_batch, 1].long())
            loss = (weight_loss[0] * loss_goal +  weight_loss[1] * (loss_aux1 + loss_aux2))/ (weight_loss[0] + weight_loss[1])
            loss_batch += loss.item() 
            acc_batch += accuracy(torch.sigmoid(output1), train_target[indices_batch], threshold=0.5)
            # compute gradients and update parameters  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc[epoch] = acc_batch / batch_nb
        train_loss[epoch] = loss_batch / batch_nb 
        
        # predict test data
        model.eval()
        model.train(False)
        with torch.no_grad():
            # forward pass
            output1, output2, output3 = model(test_input)  
            # compute Loss
            loss_goal = criterion1(output1, test_target.view(-1, 1).float())
            loss_aux1 = criterion2(output2, test_classes[:, 0].long())
            loss_aux2 = criterion3(output3, test_classes[:, 1].long())
            loss = (weight_loss[0] * loss_goal +  weight_loss[1] * (loss_aux1 + loss_aux2))/ (weight_loss[0] + weight_loss[1])
            test_loss[epoch] = loss.item()  
            test_acc[epoch] = accuracy(torch.sigmoid(output1), test_target, threshold=0.5)

        print('Epoch: [%.d/%.d],  train Loss: %.2f, test Loss: %.2f, train acc.:%.1f, test acc.: %.1f' % (
                    epoch+1,nb_epochs, train_loss[epoch], test_loss[epoch], train_acc[epoch], test_acc[epoch]))
        # save results as csv files
        result_csv_path = '../logs/weight_sharing/' if weight_sharing else '../logs/convnet/'
        if not os.path.exists(result_csv_path):
            os.makedirs(result_csv_path)
        if run_number==0 and not os.path.exists(os.path.join(result_csv_path, 'result-%d-%.1f-%d-%.8f.csv' % (weight_loss[0], weight_loss[1],
                                                                                                      batch_size,lr))):
            with open(os.path.join(result_csv_path, 'result-%d-%.1f-%d-%.8f.csv' % (weight_loss[0], weight_loss[1], batch_size,lr)), 'w', encoding='utf-8', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['lr','batch_size','weight_decay','weight_loss_main','weight_loss_aux',
                             'train_loss', 'test_loss', 'train_acc','test_acc', 'epoch', 'run_number'])

        with open(os.path.join(result_csv_path, 'result-%d-%.1f-%d-%.8f.csv' % (weight_loss[0], weight_loss[1], batch_size,lr)), 
                    'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([lr, batch_size, weight_decay, weight_loss[0], weight_loss[1], 
                         train_loss[epoch].item(), test_loss[epoch].item(), train_acc[epoch].item(), test_acc[epoch].item(), epoch, run_number+1])

    accuracies[run_number] = test_acc[-1]

print('Accuracies : {}'.format(accuracies))
accuracies = torch.tensor(accuracies)
print('Mean: {} +- {}'.format(accuracies.mean(), accuracies.std()))

