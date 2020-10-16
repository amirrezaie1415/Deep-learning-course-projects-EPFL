import torch

class Module(object):
    """
    This is the base class for other defined classes.
    """
    
    def __init__(self):
        self.layers = {}
        self.optimizer = {}
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    #def forward(self, *input ):
        #raise NotImplementedError
    
    #def backward(self, *gradwrtoutput):
       # raise NotImplementedError
    
    #def param(self):
        #return []
    
    def add_layer(self, idx, layer):
        self.layers[idx] = layer


class Linear(Module):
    """
    This class defines a fully connected layer.
    Example: 
        fc = Linear(input_features=2, output_features=10)     
    """
    def __init__(self, input_features, output_features, _gain=1.):
        self._name = 'linear'
        self.input_features = input_features
        self.output_features = output_features
        self._weights = torch.empty(input_features, output_features)
        self._bias = torch.empty(self.output_features, 1)
        self._gain = _gain
        self._initialize_parameters()
        self._delta = None
        self._d_weights = None
        self._d_bias = None
    
    def _initialize_parameters(self):
        """ Initialize weights and bias terms.
         - weights are initialized as: Normaldist.(mean=0, std=1)/sqrt(input_features)
         - bias terms are initialized as zero values.  
         """
        std = self._gain * torch.sqrt(2. / torch.tensor(float(self.input_features + self.output_features)))
        self._weights = torch.empty_like(self._weights).normal_(0,std)
        self._bias = torch.zeros_like(self._bias)
        
    def forward(self, x):
        """ Compute the pre-activation values"""
        pre_activation = self._weights.t().matmul(x) + self._bias.clone().unsqueeze(0)
        return pre_activation 

    def __repr__(self):
        return 'Fully-connected layer: in_features:{}, out_features:{}'.format(self.input_features,
                                                                              self.output_features)
    
class ReLU(Module):
    """ This class defines the ReLU activation function, which is
        ReLU(x) = max(0, x)
    """
    def __init__(self):
        self._name = 'relu'
        self._weights = torch.tensor([])
        self._bias = torch.tensor([])
    
    def forward(self, x):
        """ Compute and return the activation values"""
        return torch.max(torch.tensor(0.), x)
    
    def derivative(self, x):
        """Compute the derivative of ReLU activation function."""
        grad = x.clone()
        grad[grad>0] = 1
        grad[grad<=0] = 0
        return grad
    
    def __repr__(self):
        return 'ReLU activation'
    

class Tanh(Module):
    """ This class defines the Tanh activation function, which is
        Tanh(x) = 2 /(1+exp(-2x)) -1
    """
    def __init__(self):
        self._name = 'tanh'
        self._weights = torch.tensor([])
        self._bias = torch.tensor([])
        
    def forward(self, x):
        return 2 * (1 / (1 + torch.exp(-2*x))) - 1
    
    def derivative(self, x):
        """Compute the derivative of Tanh activation function."""
        z = x.clone()
        tan_z = 2 * (1 / (1 + torch.exp(-2*z))) - 1
        grad = 1 - tan_z.pow(2)
        return grad
    
    def __repr__(self):
        return 'Tanh activation'
    

class LossMSE:
    """ This class defines the sum of squared errors for a single sample.
        Example: 
            For a single data point we have:
                target = tensor([t1, t2])
                pred = tensor([p1, p2])
                criterion = LossSE()
                loss = criterion(target, pred) = (t1-p1)^2 + (t2-p2)^2
    """
    def __call__(self, target, pred):
        return ((target - pred).pow(2)).sum(dim=1).mean()

    
class Sequential(Module):
    """ This class defines the architecture of a fully-connected network.
    
        Example: a fully-connected netwrok with 10 inputs, 2 hidden layers each with 50 units,
                1 output layer, and ReLU activation function is defined as:
                model = Sequential( 
                                    Linear(10, 50),
                                    ReLU(),
                                    Linear(50, 50),
                                    ReLU(),
                                    Linear(50, 1)
                                    )
                                    
            """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        # add layers
        self._nb_layers = 0  # initialize the  number of Linear layers
        act_layer = 0       # initialize the  number of activation layers
        for idx, module in enumerate(args):
            if module.__dict__['_name'] == 'linear':
                self._nb_layers += 1
                self.add_layer('fc_' + str(self._nb_layers), module)
                try:
                    if args[idx+1].__dict__['_name'] == 'relu':
                        module._gain = torch.sqrt(torch.tensor(2.)).item()
                    if args[idx+1].__dict__['_name'] == 'tanh':
                        module._gain = 5/3
                except IndexError: # for the case where the last layer does not have activation 
                    module._gain = torch.sqrt(torch.tensor(1.)).item()
            elif module.__dict__['_name'] in ['relu', 'tanh']:
                act_layer += 1
                self.add_layer('act_' + str(act_layer), module)
                
    def forward(self, x):
        """ apply the forward pass"""
        self.input = x
        nb_layer = 0
        for idx, layer in enumerate(self.layers.values()):
            x = layer(x)
            if layer.__dict__['_name'] == 'linear':
                nb_layer += 1
                self.layers['fc_' + str(nb_layer)].pre_activation = x
            elif layer.__dict__['_name'] in ['relu', 'tanh']:
                self.layers['fc_' + str(nb_layer)].activation = x
            self._nb_layers = nb_layer
        return x

  
    def grad(self, target):
        self.target = target
        try:
            self.layers['fc_'+ str(self._nb_layers)]._delta = (- 2 * (self.target - self.layers['fc_'+ str(self._nb_layers)].activation).mean(dim=0) * self.layers['act_'+ str(self._nb_layers)].derivative(self.layers['fc_'+ str(self._nb_layers)].pre_activation))
        except AttributeError: # for the case where the last layer does not have activation
            self.layers['fc_'+ str(self._nb_layers)]._delta = - 2 * (self.target - self.layers['fc_'+ str(self._nb_layers)].pre_activation).mean(dim=0) * torch.ones_like(self.layers['fc_' + str(self._nb_layers)].pre_activation)
        for ind in range(self._nb_layers - 1, 0, -1):
            self.layers['fc_'+ str(ind)]._delta = self.layers['fc_' + str(ind+1)]._weights.matmul(self.layers['fc_' + str(ind+1)]._delta).mul(self.layers['act_'+ str(ind)].derivative(self.layers['fc_' + str(ind)].pre_activation))
        
        for ind in range(self._nb_layers, 0, -1):
            self.layers['fc_'+ str(ind)]._d_bias = self.layers['fc_'+ str(ind)]._delta.sum(dim=0)
            
        for ind in range(self._nb_layers, 0, -1):
            if ind-1 != 0:
                self.layers['fc_'+ str(ind)]._d_weights = (self.layers['fc_'+ str(ind-1)].activation.matmul(self.layers['fc_'+ str(ind)]._delta.transpose(1,2))).sum(dim=0)
            else:
                self.layers['fc_'+ str(ind)]._d_weights = (self.input.matmul(self.layers['fc_'+ str(ind)]._delta.transpose(1,2))).sum(dim=0)
                
    def backward(self):
        for ind in range(self._nb_layers, 0, -1):
            self.layers['fc_'+ str(ind)]._weights = self.layers['fc_'+ str(ind)]._weights - self.optimizer['eta'] * self.layers['fc_'+ str(ind)]._d_weights 
            self.layers['fc_'+ str(ind)]._bias = self.layers['fc_'+ str(ind)]._bias - self.optimizer['eta'] * self.layers['fc_'+ str(ind)]._d_bias
    