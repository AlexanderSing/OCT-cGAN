import functools
from typing import Tuple

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 depth: int,
                 generator_filters_base: int = 64,
                 normalization_layer: nn.Module = nn.BatchNorm2d,
                 use_dropout: bool = False,
                 layers_per_level: int = 1):
        super(Generator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(outer_nc=generator_filters_base * 16,
                                             inner_nc=generator_filters_base * 16,
                                             input_channels=None,
                                             submodule=None,
                                             normalization_layer=normalization_layer,
                                             innermost=True)
        for i in range(depth - 6):
            unet_block = UnetSkipConnectionBlock(outer_nc=generator_filters_base * 16,
                                                 inner_nc=generator_filters_base * 16,
                                                 input_channels=None,
                                                 submodule=unet_block,
                                                 normalization_layer=normalization_layer,
                                                 use_dropout=use_dropout,
                                                 layers_per_level=layers_per_level,
                                                 padding=(1, 0))
        unet_block = UnetSkipConnectionBlock(outer_nc=generator_filters_base * 8,
                                             inner_nc=generator_filters_base * 16,
                                             input_channels=None,
                                             submodule=unet_block,
                                             normalization_layer=normalization_layer,
                                             layers_per_level=layers_per_level,
                                             padding=(1, 1))
        unet_block = UnetSkipConnectionBlock(outer_nc=generator_filters_base * 4,
                                             inner_nc=generator_filters_base * 8,
                                             input_channels=None,
                                             submodule=unet_block,
                                             normalization_layer=normalization_layer,
                                             layers_per_level=layers_per_level,
                                             padding=(1, 0))
        unet_block = UnetSkipConnectionBlock(outer_nc=generator_filters_base * 2,
                                             inner_nc=generator_filters_base * 4,
                                             input_channels=None,
                                             submodule=unet_block,
                                             normalization_layer=normalization_layer,
                                             layers_per_level=layers_per_level,
                                             padding=(1, 0))
        unet_block = UnetSkipConnectionBlock(outer_nc=generator_filters_base,
                                             inner_nc=generator_filters_base * 2,
                                             input_channels=None,
                                             submodule=unet_block,
                                             normalization_layer=normalization_layer,
                                             layers_per_level=layers_per_level,
                                             padding=(1, 1))
        unet_block = UnetSkipConnectionBlock(outer_nc=output_channels,
                                             inner_nc=generator_filters_base,
                                             input_channels=input_channels,
                                             submodule=unet_block,
                                             outermost=True,
                                             normalization_layer=normalization_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 outer_nc: int,
                 inner_nc: int,
                 input_channels: int = None,
                 submodule: nn.Module = None,
                 outermost: bool = False,
                 innermost: bool = False,
                 normalization_layer: nn.Module = nn.BatchNorm2d,
                 use_dropout: bool = False,
                 layers_per_level: int = 1,
                 padding: Tuple[int, int] = (1, 1)):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(normalization_layer) == functools.partial:
            use_bias = normalization_layer.func == nn.InstanceNorm2d
        else:
            use_bias = normalization_layer == nn.InstanceNorm2d
        if input_channels is None:
            input_channels = outer_nc
        downconv = nn.Conv2d(input_channels, inner_nc, kernel_size=4,
                             stride=2, padding=padding, bias=use_bias, padding_mode='reflect')
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = normalization_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = normalization_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=6, stride=(2, 1), padding=2)
            downconv = nn.Conv2d(input_channels, inner_nc, kernel_size=6, stride=(2, 1), padding=2, padding_mode='reflect')
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=padding, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=padding, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)