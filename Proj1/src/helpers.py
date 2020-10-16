import torch


def get_batch_indices(X, batch_size):
    """
    This function outputs shuffled indices of mini-batch set.

    INPUT:
        X: design tensor of size (N, C, H, W) N: number of samples, C: number of channels, H: height of image, W: width of image

    OUTPUT:
        indices: a 1d tensor containing shuffled indices of mini-batch set.
    """
    nb = X.size(0)
    index_shuffled = torch.randperm(nb)
    for count in torch.arange(0,nb,batch_size):
        if count + batch_size <= nb:
            indices = index_shuffled[count:count + batch_size]
        else:
            indices = index_shuffled[count:nb]
        yield indices


def accuracy(output_prob, target, threshold=0.5):
    """
    Compute the accuracy [in %].
    INPUT:
    threshold  = threshold on the probability

     OUTPUT:
     Accuracy = (true positive + true negative) / (true positive + true negative + false positive + false negative) * 100
    """
    output_prob = output_prob.view(-1)
    target = target.view(-1)
    output_prob = (output_prob > threshold).float()
    corr = torch.sum(output_prob == target.float())
    size = output_prob.size(0)
    acc = float(corr) / float(size) * 100
    return acc
