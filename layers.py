# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import torch
import torch.nn as nn
from modules import Conv2d, ConvTranspose2d,activation_encoder

__all__ = [
    "conv5x5",
    'deconv5x5',
    "subpel_conv3x3",
    'deconv3x3',
    "conv3x3",
    "conv1x1",
    'deconv3x3_relu',
    "conv3x3_relu",
    "conv1x1_relu",
    "conv",
    "deconv"
]

def conv5x5(in_planes: int, out_planes: int, stride: int) -> Conv2d:
    """5x5 convolution with padding"""
    return nn.Sequential(
        Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,padding=2, bias=True),
        activation_encoder(channels=out_planes,after_relu=False)
    )
        

def deconv5x5(in_planes: int, out_planes: int, stride: int) -> ConvTranspose2d:
    """5x5 deconvolution with padding"""
    return nn.Sequential(
        ConvTranspose2d(in_planes, out_planes, kernel_size=5, stride=stride,padding=2,  output_padding=stride-1, bias=True),
        activation_encoder(channels=out_planes,after_relu=False)
    )             


def subpel_conv3x3(in_ch: int, out_ch: int, r: int) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, stride=1,
                     padding=1, bias=True, weight_decoder=weight_decoder, bias_decoder=bias_decoder), 
        nn.PixelShuffle(r),
        activation_encoder(channels=out_planes,after_relu=False)
    )

def deconv3x3(in_planes: int, out_planes: int, stride: int) -> ConvTranspose2d:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1,  output_padding=stride-1, bias=True),
        activation_encoder(channels=out_planes,after_relu=False)
    )
def conv3x3(in_planes: int, out_planes: int, stride: int) -> Conv2d:
    """3x3 convolution with padding"""
    return nn.Sequential(
        Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=True),
        activation_encoder(channels=out_planes,after_relu=False)
    )


def conv1x1(in_planes: int, out_planes: int, stride: int) -> Conv2d:
    """1x1 convolution"""
    return nn.Sequential(
        Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True,),
        activation_encoder(channels=out_planes,after_relu=False)
    )

def deconv3x3_relu(in_planes: int, out_planes: int, stride: int) -> ConvTranspose2d:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1,  output_padding=stride-1, bias=True),
        nn.ReLU(inplace=True),
        activation_encoder(channels=out_planes,after_relu=True)
    )
def conv3x3_relu(in_planes: int, out_planes: int, stride: int ) -> Conv2d:
    """3x3 convolution with padding"""
    return nn.Sequential(
        Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=True),
        nn.ReLU(inplace=True),
        activation_encoder(channels=out_planes,after_relu=True)
    )


def conv1x1_relu(in_planes: int, out_planes: int, stride: int) -> Conv2d:
    """1x1 convolution"""
    return nn.Sequential(
        Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True),
        nn.ReLU(inplace=True),
        activation_encoder(channels=out_planes,after_relu=True)
    )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )