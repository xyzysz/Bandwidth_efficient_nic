import math
import warnings
import sys
import torch
import torch.nn as nn
from torch import autograd
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Union

from compressai.ops.parametrizers import NonNegativeParametrizer

from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class Conv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation:int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
    ) -> None:
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        self.acceleration = False
        self.weight_bit_depth = 8
        self.activation_bit_depth = 8

        self.sparse = False
        self.sparse_M = 4
        self.sparse_N = 2

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.acceleration:
            wbq = (2**(self.weight_bit_depth))//2 - 1
            abq = (2**(self.activation_bit_depth))//2 - 1
            in_scale = get_activation_scale(input,abq)
            weight_scale = get_weight_scale(weight,wbq)
            q_input = ste.apply((input*in_scale).clamp(-1*abq,abq))
            q_weight = ste.apply((self.weight*weight_scale).clamp(-1*wbq,wbq))
            out = F.conv2d(q_input, q_weight, None, self.stride,
                            self.padding, self.dilation, self.groups)
            out = out/(in_scale*weight_scale.reshape(1,self.out_channels,1,1)) +bias.reshape(1,self.out_channels,1,1)
        
        else:
            out = F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        return out

    def forward(self, input):
 
        if self.sparse:
            weight = Sparse_NHWC.apply(self.weight, self.sparse_N, self.sparse_M)
        else:
            weight = self.weight
        out = self._conv_forward(input,weight, self.bias)
        return out


class ConvTranspose2d(nn.ConvTranspose2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros',
    ) -> None:
        super(ConvTranspose2d, self).__init__(
        in_channels, out_channels, kernel_size, stride, padding, 
            output_padding, groups, bias, dilation,padding_mode)
        self.acceleration = False
        self.weight_bit_depth = 8
        self.activation_bit_depth = 8
        self.sparse = False
        self.sparse_M = 4
        self.sparse_N = 2

    def _conv_forward(self, input: Tensor, weight, bias, output_size: Optional[List[int]] = None) -> Tensor:
        if self.acceleration:
            wbq = (2**(self.weight_bit_depth))//2 - 1
            abq = (2**(self.activation_bit_depth))//2 - 1
            in_scale = get_activation_scale(input,abq)
            weight_scale = get_weight_scale(weight,wbq,transpose=True)
            q_input = ste.apply((input*in_scale).clamp(-1*abq,abq))
            q_weight = ste.apply((self.weight*weight_scale).clamp(-1*wbq,wbq))
            out = F.conv_transpose2d(q_input, q_weight, None, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)
            out = out/(in_scale*weight_scale.reshape(1,self.out_channels,1,1)) + bias.reshape(1,self.out_channels,1,1)
        else:
            out = F.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)
        return out

    def forward(self, input):
        if self.sparse:
            weight = Sparse_NHWC.apply(self.weight, self.sparse_N, self.sparse_M)
        else:
            weight = self.weight
        out = self._conv_forward(input,weight, self.bias)
        return out

        
class activation_encoder(nn.Module):
    def __init__(
        self,
        channels: int,
        after_relu = False,
    ) -> None:
        super(activation_encoder, self).__init__()
        self.channels = channels
        self.after_relu = after_relu
        self.a_bits = 0
        self.compress_activation = False
        self.compression_bit_depth = 8
        self.adaptive_bit_depth_allocation = False
        self.adaptive_bit_depth_start = 3
        self.adaptive_bit_depth_end = 12
        self.adaptive_bit_depth_ref = nn.Parameter(torch.arange(self.adaptive_bit_depth_start,self.adaptive_bit_depth_end+1),requires_grad=False)
        self.adaptive_bit_depth = nn.Parameter(torch.ones(self.adaptive_bit_depth_end-self.adaptive_bit_depth_start+1))
        self.self_information =  GaussianConditional(None)
        self.use_affine = False
        self.scale = nn.Conv2d(channels,channels,kernel_size=1,stride=1)
        self.scale_t = nn.Conv2d(channels,channels,kernel_size=1,stride=1)
        self.h_a = nn.Sequential(nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
                                nn.Conv2d(channels,channels,kernel_size=5,stride=2,padding=2),
                                nn.Conv2d(channels,channels,kernel_size=5,stride=2,padding=2),                
                        )
        self.h_s = nn.Sequential(nn.ConvTranspose2d(channels,2*channels,kernel_size=5,stride=2,padding=2,output_padding=1),
                                 nn.ConvTranspose2d(2*channels,2*channels,kernel_size=5,stride=2,padding=2,output_padding=1),
                                 nn.Conv2d(2*channels,2*channels,kernel_size=3,stride=1,padding=1),                
                        )
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.ep = 'Guassian'
        self.use_rans = False
        self.tau = 1
    
    def get_scale_table(self):
        return torch.exp(torch.linspace(math.log(0.11), math.log(2**self.compression_bit_depth), (2**self.compression_bit_depth)//2))
        
        
    def forward(self, input):
        if self.compress_activation:
            b,c,h,w = input.size()
            half=0.5
            a_t = self.scale(input) if self.use_affine else input
            if self.adaptive_bit_depth_allocation:
                if self.training:
                    bit_depth_sample = gumbel_softmax(self.adaptive_bit_depth, 0, self.tau)
                    bit_depth = (self.adaptive_bit_depth_ref*bit_depth_sample).sum()
                else:
                    bit_depth = torch.argmax(self.adaptive_bit_depth, dim=0)+3
            else:
                bit_depth = self.compression_bit_depth

            a_t, amax, amin = quant_scaling(a_t,bit_depth)  
            a_t_hat = ste.apply(a_t)        
            if self.ep == 'Gaussian':
                sigma = torch.std(a_t_hat.reshape(b,c,-1),dim=2).unsqueeze(-1).unsqueeze(-1)
                mu = torch.mean(a_t_hat.reshape(b,c,-1),dim=2).unsqueeze(-1).unsqueeze(-1)
                if self.use_rans:
                    self.self_information.update_scale_table(scale_table=self.get_scale_table())
                    indexes = self.self_information.build_indexes(sigma*torch.ones_like(a_t_hat))
                    y_strings = self.self_information.compress(a_t_hat, indexes,means=mu*torch.ones_like(a_t_hat))
                    self.a_bits = y_strings 
                else:
                    _, a_t_likelihoods = self.self_information(a_t,sigma,mu)
                    bits = torch.log(a_t_likelihoods).sum() / (-math.log(2)) + self.channels*64
                self.a_bits = bits
            elif self.ep == 'EG':
                if self.training:
                    q = floor_ste.apply(a_t_hat/(2**4))#k=4
                    q_length = 2*(floor_ste.apply(torch.log(q+1) / math.log(2)) +1)-1
                    r_length = 4
                else:
                    q = torch.floor(a_t_hat/(2**4))#k=4
                    q_length = 2*(torch.floor(torch.log(q+1) / math.log(2)) +1)-1
                    r_length = 4
                code_length = q_length + r_length + 1
                self.a_bits = code_length.sum()
            elif self.ep == 'Symmetrical_EG':
                a_t_avg = (torch.median(a_t_hat.reshape(b,c,-1),dim=2)[0]).unsqueeze(-1).unsqueeze(-1)
                res = a_t_hat-a_t_avg
                res_code = torch.where(res>=0,2*res,2*abs(res)+1)
                if self.training:
                    binary_length = floor_ste.apply(torch.log(res_code+1) / math.log(2)) +1
                else:
                    binary_length = torch.floor(torch.log(res_code+1) / math.log(2))+1
                code_length = 2*binary_length - 1
                bits = code_length.sum() +self.compression_bit_depth*b*c
                self.a_bits = bits
            elif self.ep=='Sparse_EG':
                a_t_hat_temp = torch.where(a_t_hat-1>0,a_t_hat-1,0)
                if self.training:
                    q = floor_ste.apply(a_t_hat_temp/(2**4))#k=4
                    q_length = 2*(floor_ste.apply(torch.log(q+1) / math.log(2)) +1)-1
                    r_length = 4
                else:
                    q = torch.floor(a_t_hat_temp/(2**4))#k=4
                    q_length = 2*(torch.floor(torch.log(q+1) / math.log(2)) +1)-1
                    r_length = 4
                egk_length = q_length + r_length + 1
                code_length = torch.where(a_t_hat==0,1,egk_length)
                self.a_bits = code_length.sum()
            elif self.ep=='Uniform':
                self.a_bits = b*c*h*w*bit_depth
            elif self.ep=='Hyperprior':
                hyper = self.h_a(a_t_hat)
                hyper, hyper_max, hyper_min = quant_scaling(hyper,bit_depth)  
                hyper_hat = ste.apply(hyper)      
                h_sigma = torch.std(hyper_hat.reshape(b,c,-1),dim=2).unsqueeze(-1).unsqueeze(-1)
                h_mu = torch.mean(hyper_hat.reshape(b,c,-1),dim=2).unsqueeze(-1).unsqueeze(-1)
                _, hyper_likelihoods = self.self_information(hyper,h_sigma,h_mu)
                hyper_hat = quant_scaling(hyper_hat, bit_depth,hyper_max, hyper_min,True)
                mu, sigma = self.h_s(hyper_hat).chunk(2,1)
                _, a_t_likelihoods = self.self_information(a_t,sigma,mu)
                bits = torch.log(a_t_likelihoods).sum() / (-math.log(2)) + torch.log(hyper_likelihoods).sum() / (-math.log(2))
                self.a_bits = bits
            else:
                raise ValueError('invalid entropy coding method')

            a_out = quant_scaling(a_t_hat, bit_depth,amax,amin,True)
            out = self.scale_t(a_out) if self.use_affine else a_out
            return out 
        else:
            return input
        
           

class ste(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None

class floor_ste(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None
class L1_norm(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.norm(input,p=1)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        input, = ctx.saved_tensors
        grad = torch.where(input>0,1e-7,-(1e-7))
        grad_input = torch.where(grad,0,grad)
        return grad_input, None
        
def quant_scaling(x, bit_depth, xmax_in=None, xmin_in=None, inverse=False):
    xmax = xmax_in if xmax_in is not None else torch.max(x)
    xmin = xmin_in if xmin_in is not None else torch.min(x)
    qmax = 2**(bit_depth)-1
    # qmax = 255
    if inverse:
        re_x = (x/qmax)*(xmax-xmin)+xmin
        return re_x 
    else:
        scaled_x = ((x-xmin)/(xmax-xmin))*qmax
        return scaled_x, xmax, xmin
  
def get_activation_scale(x,abq):
    scale = abq/torch.max(abs(x))
    return scale

def get_weight_scale(x,wbq, transpose=False):
    if transpose:
        x = x.permute(1,0,2,3)
    o,i,h,w = x.size()
    scale = wbq/torch.max(abs(x.reshape(o,-1)),dim=1)[0]
    if transpose:
        return scale.reshape(1,o,1,1)
    else:
        return scale.reshape(o,1,1,1)
        
class Sparse_NHWC(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):

        ctx.save_for_backward(weight)
        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
        w_b = w_b.permute(0,3,1,2)

        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b

    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None 

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)
    

    return x