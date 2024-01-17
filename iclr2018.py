import sys
import math
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import conv, deconv, update_registered_buffers
from layers import conv1x1, conv5x5, deconv5x5, conv3x3, deconv3x3, conv1x1_relu,conv3x3_relu, deconv3x3_relu,conv,deconv
from modules import Conv2d, ConvTranspose2d, activation_encoder
from compressai.models import CompressionModel
from typing import Type, Any, Callable, Union, List, Optional

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
 
        
class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, 
            N, 
            M, 
            **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        class RB(nn.Module):
            def __init__(self, C):
                super().__init__()

                self.RB = nn.Sequential(
                conv1x1_relu(C,C//2, 1),
                conv3x3_relu(C//2,C//2, 1),
                conv1x1_relu(C//2,C, 1),
                )

            def forward(self, x):
                y = self.RB(x) + x
                return y

        self.g_a = nn.Sequential(
            conv5x5(3, N, 2),
            RB(N),
            conv5x5(N, N, 2),
            RB(N),
            conv5x5(N, N, 2),
            RB(N),
            conv5x5(N, M, 2),
        )

        self.g_s = nn.Sequential(
            deconv5x5(M, N, 2),
            RB(N),
            deconv5x5(N, N, 2),
            RB(N),
            deconv5x5(N, N, 2),
            RB(N),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv3x3_relu(M, N, 1),
            conv3x3_relu(N, N, 2),
            conv3x3(N, N, 2),
        )

        self.h_s = nn.Sequential(
            deconv3x3_relu(N, N, 2),
            deconv3x3_relu(N, N, 2),
            conv3x3_relu(N, M, 1),
        )
    
        self.gaussian_conditional = GaussianConditional(None)
        

        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
    
    def set_acceleration(self, acceleration) :
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d) :
                m.acceleration = acceleration['if_acceleration']
                if acceleration['if_acceleration']:
                    m.weight_bit_depth = acceleration['weight_bit_depth']
                    m.activation_bit_depth = acceleration['activation_bit_depth']
    def set_nm_sparse(self, nm_sparse):
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d) :
                m.sparse = nm_sparse['if_nm_sparse']
                if nm_sparse['if_nm_sparse']:
                    m.sparse_M = nm_sparse['sparse_M']
                    m.sparse_N = nm_sparse['sparse_N']
    def set_compress_activation(self, compress_activation):
        for m in self.modules():
            if isinstance(m, activation_encoder):
                m.compress_activation = compress_activation['if_compress_activation']
                if compress_activation['if_compress_activation']:
                    m.compression_bit_depth = compress_activation['compression_bit_depth']
                    m.use_affine = compress_activation['use_affine']
                    m.ep = compress_activation['ep']
                    m.use_rans = compress_activation['use_rans']
                    m.adaptive_bit_depth_allocation = compress_activation['adaptive_bit_depth_allocation']
                    if compress_activation['adaptive_bit_depth_allocation']:
                        m.adaptive_bit_depth_start = compress_activation['adaptive_bit_depth_start']
                        m.adaptive_bit_depth_end = compress_activation['adaptive_bit_depth_end']

    
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}