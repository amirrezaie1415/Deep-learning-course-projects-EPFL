import torch
from torch import nn
import dlc_practical_prologue as prologue
from helpers import *
from networks import *
import csv 
import os


# define hyper-parameter spaces
lr_list =[1e-2, 5e-3, 1e-3, 5e-4, 1e-4]  # learning rate
batch_size_list = [10, 20, 50, 100, 200]  # batch size
weight_mainloss = 1.  # wight associated with the main loss 
weight_auxloss_list = [0  , 0.1 , 0.2 , 0.3, 0.4, 0.5, 
                       0.6, 0.7 , 0.8 , 0.9, 1.0, 2.0, 
                       3.0] # weight associated with the auxiliary loss
epochs = 150  # number of epochs
weight_decay = 0  # weight decay

for weight_auxloss in weight_auxloss_list:    
    for batch_size in batch_size_list:
        print('-'*5 + 'batch size:%d'%batch_size)
        for lr in lr_list:
            print('-'*5 + 'lr:%.1e'%lr)
            torch.manual_seed(0) # set the seed number for the sake of reproducibility
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
            model = ConvNet(nb_channels=64, auxiliary_loss=True)
            lr = lr  # learning rate
            weight_decay = weight_decay  # weight decay
            nb_epochs = epochs  # number of epochs
            batch_size = batch_size  # batch size
            weight_loss = [weight_mainloss, weight_auxloss] # weight associated with the main loss (w_{m} in the report) = weight loss[0],
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
                    loss_batch += loss.clone() 
                    acc_batch += accuracy(torch.sigmoid(output1), train_target[indices_batch], threshold=0.5)
                    # compute gradients and update paramters  
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
            result_csv_path = '../logs/'
            if not os.path.exists(result_csv_path):
                os.makedirs(result_csv_path)
            if not os.path.exists(os.path.join(result_csv_path, 'result-%d-%.1f.csv' % (weight_loss[0], weight_loss[1]))):
                f = open(os.path.join(result_csv_path, 'result-%d-%.1f.csv' % (weight_loss[0], weight_loss[1])), 'w', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow(['lr','batch_size','weight_decay','weight_loss_main','weight_loss_aux','test loss', 'max test acc'])
                f.close()

            f = open(os.path.join(result_csv_path, 'result-%d-%.1f.csv' % (weight_loss[0], weight_loss[1])), 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow([lr, batch_size, weight_decay, weight_loss[0], weight_loss[1], 
                         test_loss[test_acc.argmax()].item(),  test_acc.max().item()]) 
            f.close()


