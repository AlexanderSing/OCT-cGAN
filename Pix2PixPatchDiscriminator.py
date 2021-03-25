import functools

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,
                 input_channels: int,
                 discriminator_filters_base: int = 64,
                 n_layers: int = 3,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 use_sigmoid: bool = False,
                 convs_per_layer: int = 1):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padws = [(2, 2), (1, 1), (1, 0), (1, 0)]
        input_conv = nn.Conv2d(input_channels, discriminator_filters_base, kernel_size=6, stride=(2, 1), padding=padws[0], padding_mode='reflect')
        sequence = [
            input_conv,
            nn.LeakyReLU(0.2, True)
        ]
        input_conv.register_forward_hook(self.add_intermediate_output)

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            intermediate_conv = nn.Conv2d(discriminator_filters_base * nf_mult_prev, discriminator_filters_base * nf_mult,
                          kernel_size=kw, stride=2, padding=padws[n], padding_mode='reflect', bias=use_bias)
            sequence += [
                intermediate_conv,
                norm_layer(discriminator_filters_base * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            intermediate_conv.register_forward_hook(self.add_intermediate_output)

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        intermediate_conv_2 = nn.Conv2d(discriminator_filters_base * nf_mult_prev, discriminator_filters_base * nf_mult,
                      kernel_size=kw, stride=1, padding=padws[-1], padding_mode='reflect', bias=use_bias)
        sequence += [
            intermediate_conv_2,
            norm_layer(discriminator_filters_base * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        intermediate_conv_2.register_forward_hook(self.add_intermediate_output)

        last_conv = nn.Conv2d(discriminator_filters_base * nf_mult, 1, kernel_size=3, stride=1, padding=0)

        sequence += [nn.Conv2d(discriminator_filters_base * nf_mult, 1, kernel_size=3, stride=1, padding=0)]
        last_conv.register_forward_hook(self.add_intermediate_output)

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.intermediate_outputs = []

    def forward(self, input):
        self.intermediate_outputs = []
        return self.model(input)

    def add_intermediate_output(self, conv, input, output):
        self.intermediate_outputs.append(output.detach())

    def get_intermediate_output(self):
        return self.intermediate_outputs