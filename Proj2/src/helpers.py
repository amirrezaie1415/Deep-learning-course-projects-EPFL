import torch
import math 



def get_mini_batch(X, y, batch_size):
    """ This function yeilds mini-batch of data .
    IPUT:
        X = design tensor of the size (N, D), where N is the number of samples
        and D is the number of features. 
        y = target tensor (one-hot encoded) of size (N, C), where C is the number of classes
        batch_size = size of batches
    OUTPUT:
        incides: These are the indices used to create mini-batches
        mini_batch_X: mini_batch of desing tensor of the size (batch_size, D, 1)
        mini_batch_y: mini_batch of target tensor of the size (batch_size, C, 1)
    """
    nb = X.size(0)
    index_shuffuled = torch.randperm(nb)
    for count in torch.arange(0,nb,batch_size):
        if count + batch_size <= nb:
            indices = index_shuffuled[count:count + batch_size]
        else:
            indices = index_shuffuled[count:nb]
        mini_batch_X = X[indices].view(batch_size, 2,1)
        mini_batch_y = y[indices].view(batch_size, 2,-1)
        yield indices, mini_batch_X, mini_batch_y
        
def label_data(X):
    """ This function generates the label (one-hot encoded) for the input design tensor X.
    
    INPUT:
        X = desing tensor of the size (N, D), where N is the number of samples
        and D is the number of features.
    
    OUTPUT:
        y_one_hot = one-hot encoded label tensor of the size (N, C), where C is the number of classes
    """
    norm = torch.norm(X - torch.tensor([0.5, 0.5]), dim=1)
    # create the target tensor
    label =  1. - ((norm > 1/torch.sqrt(2 * torch.tensor(math.pi))) & (norm < 1.)).float()
    y_one_hot = torch.cat((1 - label.view(-1,1), label.view(-1,1)), 1) # one-hot encoded
    return y_one_hot

def train_test_split(X, y, ratio, random_state=42):
    """
    This function split the dataset into train and test data.
    
    INPUT:
        X = design tensor of the size (N, D), where N is the number of samples
        and D is the number of features. 
        y = target tensor (one-hot encoded) of size (N, C), where C is the number of classes
        ratio = ratio of the test data number to N. For example: ratio = 0.2 means that 
        80% of data is used for training and 20% for test.  
        random_state = a number set as the seed for random number generator 
        
    OUTPUT:
        X_train = train design tensor 
        X_test = test design tensor
        y_train = train target tensor 
        y_test = test target tensor
    """
    # fix the seed number
    torch.manual_seed(random_state)
    
    indices_shuffuled = torch.randperm(y.size(0))
    test_ind = indices_shuffuled[0:int(y.size(0) * ratio)]
    X_test = X[test_ind]
    y_test = y[test_ind]

    train_ind = indices_shuffuled[int(y.size(0) * ratio):]
    X_train = X[train_ind]
    y_train = y[train_ind]
    
    return X_train, X_test, y_train, y_test

def accuracy(pred, target):
    """
    Compute the accuracy.
    INPUT:
    pred  = predicted labels of size (N), where N is the number of samples.
    target = true class labels of size (N), where N is the number of samples.

     OUTPUT:
     Accuracy = (true positive + true negative) / (true positive + true negative + false positive + false negative)
    """

    corr = torch.sum(pred == target)
    size = pred.size(0)
    acc = float(corr) / float(size) * 100
    return acc

def pred_label (model, X_normalized):
    """ This function is used to pred the labels given the model and
        the normalized desing tensor (X_normalized).
    IPUT:
        model = a fully-connected network as an instance of Sequantial class
        X_normalized = normalized desing tensor of the size (N, D), where N is the number of samples
        and D is the number of features. 
    OUTPUT:
        pred = predicted values
        pred_label = predicted labels as the argmax of pred.
        Example: for a single data point:
            pred = tensor([2.5, 0.2])
            pred_label = tensor([1])
    """
    nb = X_normalized.size(0)
    pred = model(X_normalized.view(nb, 2, 1))
    pred_label = pred.argmax(dim=1).view(-1).float()
    return pred, pred_label


