import os
import csv
import shutil
import copy
import glob
import json
import os
from collections import OrderedDict
from functools import partial

from torch import nn


import torch
import yaml
from torch.utils.data import DataLoader
import re
import torchvision.transforms as transforms

#import modules
import training_utils

import csv
import math

import matplotlib.colors as colors
import numpy as np
import scipy.ndimage
import scipy.special
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


######################################### META LEARNING MODULES #############################################
########## MODULE FROM GITHUB: https://github.com/vsitzmann/metasdf/blob/master/meta_modules.py #############
#############################################################################################################

class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules.
    """
    def meta_named_parameters(self, prefix='', recurse=True):
        """Return named parameters for meta-learning."""
        gen = self._named_members(lambda module: module._parameters.items() if isinstance(module, MetaModule) else [], prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

class MetaSequential(nn.Sequential, MetaModule):
    """
    Sequential container that supports meta-learning.
    """
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module '
                    '(inheriting from `nn.Module`), or a `MetaModule`. '
                    'Got type: `{0}`'.format(type(module)))
        return input

class MAML(nn.Module):
    '''MAML module from https://github.com/vsitzmann/metasdf'''

    def __init__(self, num_meta_steps, hypo_module, loss, init_lr,
                first_order=False, l1_lambda=0):
        super().__init__()

        self.hypo_module = hypo_module  # The module who's weights we want to meta-learn.
        self.first_order = first_order
        self.loss = loss
        self.log = []
        self.l1_lambda = l1_lambda

        self.register_buffer('num_meta_steps', torch.Tensor([num_meta_steps]).int())

        self.lr = nn.ModuleList([])
        for name, param in hypo_module.meta_named_parameters():
            self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                for _ in range(num_meta_steps)]))
        param_count = 0
        for param in self.parameters():
            param_count += np.prod(param.shape)
        print(param_count)

    def _update_step(self, loss, param_dict, step):
        """Inner Loop Update Step: gradient descent on a specific task"""
        grads = torch.autograd.grad(loss, param_dict.values(), create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            lr = self.lr[i][step]
            params[name] = param - lr * grad
        return params, grads

    def forward_with_params(self, query_x, fast_params, **kwargs):
        output = self.hypo_module({'coords': query_x}, params=fast_params)
        return output

    def generate_params(self, context_dict):
        """Adapt the model using the context set with input 'x' and target 'y'."""
        x = context_dict.get('x').cuda()
        y = context_dict.get('y').cuda()
        meta_batch_size = x.shape[0]

        with torch.enable_grad():
            # Initialize parameters by replicating the initial values for each batch element
            fast_params = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

            prev_loss = 1e6
            intermed_predictions = []
            for j in range(self.num_meta_steps):
                # Forward pass with current parameters
                predictions = self.hypo_module({'coords': x}, params=fast_params)

                # Compute loss
                loss = self.loss(predictions, y)
                if self.l1_lambda > 0:
                    # Apply L1 regularization
                    l1_loss = model_l1_dictdiff(self.hypo_module.state_dict(), fast_params, self.l1_lambda)
                    loss += l1_loss['l1_loss']
                intermed_predictions.append(predictions['model_out'])

                # Using the computed loss, update the fast parameters.
                fast_params, grads = self._update_step(loss, fast_params, j)
                prev_loss = loss

        return fast_params, intermed_predictions

    def forward(self, meta_batch, **kwargs):
        # The meta_batch conists of the "context" set (the observations we're conditioning on)
        # and the "query" inputs (the points where we want to evaluate the specialized model)
        context = meta_batch['context']
        query_x = meta_batch['query']['x'].cuda()

        # Specialize the model with the "generate_params" function.
        fast_params, intermed_predictions = self.generate_params(context)

        # Compute the final outputs.
        model_output = self.hypo_module({'coords': query_x}, params=fast_params)['model_out']
        out_dict = {'model_out': model_output, 'intermed_predictions': intermed_predictions, 'fast_params': fast_params}

        return out_dict
    
class BatchLinear(nn.Linear, MetaModule):
    """
    A linear meta-layer that can handle batched weight matrices and biases.
    This layer is designed to be used with hypernetworks, where weights are dynamically generated.
    Based on the implementation from: https://github.com/vsitzmann/siren
    """
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        # If no parameters are provided, use the layer's own weights and biases
        if params is None:
            params = OrderedDict(self.named_parameters())
        # Extract weight and bias from the parameter dictionary
        bias = params.get('bias', None) # Use None if bias is not provided
        weight = params['weight'] # 'weight' must be provided
        # The weight tensor is permuted to ensure that the last two dimensions are (in_features, out_features).
        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        if bias is not None:
            output += bias.unsqueeze(-2)
        return output


######################################### INR MODULES #########################################

class INRNet(MetaModule):
    '''
    Implicit Neural Representation (INR) network with support for Fourier Features.
    This class maps input coordinates to output features using positional encoding and a fully connected neural network.
    '''

    def __init__(self, out_features=1, type='sine', in_features=2,
                  hidden_features=256, num_hidden_layers=3, ff_dims=None, **kwargs):
        super().__init__()

        # Initialize positional encoding (Based on NeRF - source [40] in the article)
        num_frequencies = ff_dims
        self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                    sidelength=kwargs.get('sidelength', None),
                                                    fn_samples=kwargs.get('fn_samples', None),
                                                    use_nyquist=kwargs.get('use_nyquist', True),
                                                    num_frequencies=num_frequencies,
                                                    scale=kwargs.get('encoding_scale', 2.0))
        # Update input features to match the output dimension of positional encoding
        in_features = self.positional_encoding.out_dim

        # Fully connected neural network (FCBlock)
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Clone and detach coordinates to compute gradients w.r.t. input
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        # Apply positional encoding
        coords = self.positional_encoding(coords)
        # Pass encoded coordinates through the neural network
        output = self.net(coords, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def predict(self, model_input):
        """Alias for the forward method, for convenience."""
        return self.forward(model_input)

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}
    

class FCBlock(MetaModule):
    """
    Fully connected neural network with support for meta-learning (weights swapping).
    """

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Define nonlinearity (default: sine activation with specific initialization)
        nl, nl_weight_init, first_layer_init = (Sine(), sine_init, first_layer_sine_init)

        # Use custom weight initialization if provided
        self.weight_init = weight_init if weight_init is not None else nl_weight_init

        # Define the network layers
        self.net = []
        self.net.append(MetaSequential(BatchLinear(in_features, hidden_features), nl))

        # Hidden layers
        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(BatchLinear(hidden_features, hidden_features), nl))

        # Output layer (linear or with nonlinearity)
        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features), nl))

        # Combine all layers into a MetaSequential module
        self.net = MetaSequential(*self.net)

        # Apply weight initialization
        if self.weight_init is not None:
            self.net.apply(self.weight_init)
        # Special initialization for the first layer
        self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x

        # Forward pass through each layer and record activations
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)
                if retain_grad:
                    x.retain_grad()
                # Store activation
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations
    

######################################### FOURIER FEATURE ENCODINGS #########################################
################### MODULE TO ADD POSITIONAL ENCODING AS IN NERF (CHOSEN IN THE ARTICLE) ####################
########### FUNCTION FROM SIREN GITHUB: https://github.com/vsitzmann/siren/blob/master/modules.py ###########
#############################################################################################################

class PosEncodingNeRF(nn.Module):
    """Module to add positional encoding as in NeRF.
    From SIREN github: https://github.com/vsitzmann/siren/blob/master/modules.py

    Args:
        in_features (int): Number of input features (1D, 2D, or 3D coordinates).
        sidelength (int or tuple, optional): Spatial resolution of the input (required if in_features=2).
        fn_samples (int, optional): Number of samples (required if in_features=1).
        use_nyquist (bool, optional): Whether to use Nyquist frequency to determine the number of frequencies.
        num_frequencies (int, optional): Number of frequency bands to use. If None, it's determined based on input dimension.
        scale (float, optional): Scaling factor for the frequencies.
    """

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True, num_frequencies=None, scale=2):
        super().__init__()
        self.in_features = in_features # Number of input features (1, 2, or 3)
        self.scale = scale  # Scaling factor for frequency bands
        self.sidelength = sidelength # Spatial resolution (for 2D inputs)

        # Determine the number of frequencies if not explicitly provided
        if num_frequencies == None:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                assert fn_samples is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = num_frequencies
        self.out_dim = in_features + in_features * 2 * self.num_frequencies  

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]
                sin = torch.unsqueeze(torch.sin((self.scale ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((self.scale ** i) * np.pi * c), -1)
                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc


##################################### ACTIVATION & WEIGHT INITIALIZATION #########################################
class Sine(nn.Module):
    """
    A custom sine activation function (sine function scaled by a factor of 30, see page 5 of the article)
    It is often used in Implicit Neural Representations (INRs) like SIREN (Sine-based Representational Networks).
    """
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return torch.sin(30 * input)

def sine_init(m):
    """
    Initializes the weights of a layer using a uniform distribution for the SIREN model.
    The range is determined by the number of input features and scaled by 1/30.
    This initialization helps SIREN models converge faster by ensuring that input signals pass through sine activations with the proper amplitude and frequency.
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

############################################## UTILS ##############################################

def get_subdict(dictionary, key=None):
    """
    Extracts a subset of a dictionary whose keys start with a given prefix.

    This function is useful in meta-learning models where each layer's parameters
    are stored in a single dictionary with names like 'net.0.weight' or 'net.1.bias'.
    By specifying a prefix, you can extract the parameters for a specific layer.
    """
    if dictionary is None:
        return None
    # If no key is provided, return the entire dictionary
    if (key is None) or (key == ''):
        return dictionary
    # Use regular expressions to match keys starting with the specified prefix
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    # Extract and return key-value pairs where the key matches the prefix
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value) in dictionary.items() if key_re.match(k) is not None)

########################## UTILS FUNCTIONS #########################

def model_l1_dictdiff(ref_model_dict, model_dict, l1_lambda):
    "Computes the L1 norm of the difference between the parameters of two model state dictionaries. Used to regulaize models during training."
    l1_norm = sum((p.squeeze() - ref_p.squeeze()).abs().sum() for (p, ref_p) in zip(ref_model_dict.values(), model_dict.values()))
    return {'l1_loss': l1_lambda * l1_norm}

def model_l1(model, l1_lambda):
    "Computes the L1 norm of the models parameters and weights it with l1_lambda"
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return {'l1_loss': l1_lambda * l1_norm}