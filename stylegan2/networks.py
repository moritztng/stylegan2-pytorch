import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, n_conv, n_feature_maps, equal_lr):
        super().__init__()
        self.conv_rgb = Conv(3, n_feature_maps[0], 1, equal_lr=equal_lr)
        self.blocks = nn.Sequential(*[DiscriminatorBlock(n_feature_maps[i], n, n_conv, equal_lr) for i, n in enumerate(n_feature_maps[1:])])
        self.conv_3 = Conv(n_feature_maps[-1] + 1, n_feature_maps[-1], 3, padding=1, equal_lr=equal_lr)
        self.conv_4 = Conv(n_feature_maps[-1], n_feature_maps[-1], 4, equal_lr=equal_lr)
        self.lrelu = nn.LeakyReLU(0.2)
        self.fc = FC(n_feature_maps[-1], 1, equal_lr=equal_lr)
    def forward(self, x):
        x = self.conv_rgb(x)
        x = self.blocks(x)
        mean_std_batch = torch.sqrt(x.var(0) + 1e-8).mean().expand(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat((x, mean_std_batch), 1)
        x = self.lrelu(self.conv_3(x))
        x = self.lrelu(self.conv_4(x))
        return self.fc(x.view(x.size()[:2]))

class DiscriminatorBlock(nn.Module):
    def __init__(self, n_feature_maps_in, n_feature_maps_out, n_conv, equal_lr):
        super().__init__()
        self.layers = []
        for _ in range(n_conv - 1):
            self.layers += [Conv(n_feature_maps_in, n_feature_maps_in, 3, padding=1, equal_lr=equal_lr), nn.LeakyReLU(0.2)]
        self.layers += [Conv(n_feature_maps_in, n_feature_maps_out, 3, padding=1, equal_lr=equal_lr), nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*self.layers)
        self.downsample = lambda x: nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        self.conv_res = Conv(n_feature_maps_in, n_feature_maps_out, 1, equal_lr=equal_lr)
    def forward(self, x):
        return self.conv_res(self.downsample(x)) + self.downsample(self.layers(x))

class Generator(nn.Module):
    def __init__(self, n_layers_mapper, n_dim_mapper, n_conv_blocks, n_feature_maps, n_dim_const, equal_lr):
        super().__init__()
        self.mapper = Mapper(n_layers_mapper, n_dim_mapper, equal_lr)
        self.synthesizer = Synthesizer(n_dim_mapper, n_conv_blocks, n_feature_maps, n_dim_const, equal_lr)
    def forward(self, z, weight_truncation=None):
        w = self.mapper(z)
        return self.synthesizer(w, weight_truncation=weight_truncation)

class Mapper(nn.Module):
    def __init__(self, n_layers, n_dim, equal_lr):
        super().__init__()
        self.layers = [PixelNorm()]
        for _ in range(n_layers):
            self.layers.append(FC(n_dim, n_dim, equal_lr=equal_lr))
            self.layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*self.layers)
    def forward(self, z):
        return self.layers(z)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z):
        return z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)

class Synthesizer(nn.Module):
    def __init__(self, n_dim_w, n_conv_blocks, n_feature_maps, n_dim_const, equal_lr):
        super().__init__()
        self.blocks = [SynthesizerBlock(n_conv_blocks, True, n_dim_w, n_feature_maps[0], equal_lr, n_dim_const=n_dim_const)]
        for i, n in enumerate(n_feature_maps[1:]):
            self.blocks.append(SynthesizerBlock(n_conv_blocks, False, n_dim_w, n, equal_lr, n_feature_maps_in=n_feature_maps[i]))
        self.blocks = nn.Sequential(*self.blocks)
        self.register_buffer('avg_w', torch.zeros(n_dim_w))
    def forward(self, w, weight_truncation=None):
        if self.training:
            self.avg_w.copy_(w.detach().mean(dim=0).lerp(self.avg_w, 0.995))
        if weight_truncation:
            w = self.avg_w.lerp(w, weight_truncation)
        return self.blocks((w, None, None))[2]

class SynthesizerBlock(nn.Module):
    def __init__(self, n_blocks, first, n_dim_w, n_feature_maps_out, equal_lr, n_feature_maps_in=None, n_dim_const=None):
        super().__init__()
        self.first = first
        if first:
            self.const = nn.Parameter(torch.randn(1, n_feature_maps_out, n_dim_const, n_dim_const))
            self.blocks = []
        else:
            self.upsample = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=False)
            self.blocks = [ConvBlock(n_dim_w, n_feature_maps_in, n_feature_maps_out, equal_lr)]
        for _ in range(n_blocks - 1):
            self.blocks.append(ConvBlock(n_dim_w, n_feature_maps_out, n_feature_maps_out, equal_lr))
        self.blocks = nn.Sequential(*self.blocks)
        self.conv_rgb = Conv(n_feature_maps_out, 3, 1, equal_lr=equal_lr)
        self.affine_transform = AffineTransform(n_dim_w, n_feature_maps_out, equal_lr)
    def forward(self, args):
        w, x, rgb = args
        if self.first:
            x = self.const.expand(w.size(0), -1, -1, -1)
        else:
            x = self.upsample(x)
            rgb = self.upsample(rgb)
        w, x = self.blocks((w, x))
        rgb_new = self.conv_rgb(x, y_s=self.affine_transform(w), demod=False)
        rgb = rgb_new if self.first else rgb + rgb_new
        return w, x, rgb

class ConvBlock(nn.Module):
    def __init__(self, n_dim_w, n_feature_maps_in, n_feature_maps_out, equal_lr):
        super().__init__()
        self.conv = Conv(n_feature_maps_in, n_feature_maps_out, 3, padding=1, equal_lr=equal_lr)
        self.affine_transform = AffineTransform(n_dim_w, n_feature_maps_in, equal_lr)
        self.scale_noise = nn.Parameter(torch.tensor(0.))
        self.lrelu = nn.LeakyReLU(0.2)
    def forward(self, args):
        w, x = args
        x = self.conv(x, y_s=self.affine_transform(w), demod=True) 
        x = x + self.scale_noise * torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x = 2**0.5 * self.lrelu(x)
        return w, x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, equal_lr=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        norm_const = (in_channels * kernel_size**2)**-0.5
        scale_init = 1 if equal_lr else norm_const
        self.scale_forward = norm_const if equal_lr else 1
        self.weight = nn.Parameter(scale_init * torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
    def forward(self, x, y_s=None, demod=False):
        weight = self.scale_forward * self.weight
        bias = self.bias
        groups = 1
        batch_size = x.size(0)
        if y_s is not None:
            weight = y_s.view(y_s.size(0), 1, y_s.size(1), 1, 1) * weight.unsqueeze(0)
            if demod:
                x_s = ((weight ** 2).sum(dim=(2, 3, 4)) + 1e-8) ** 0.5
                weight = weight / x_s.view(*x_s.size(), 1, 1, 1)
            weight = weight.view(-1, *weight.size()[2:])
            bias = bias.expand(batch_size, -1).reshape(-1)
            groups = batch_size
            x = x.reshape(1, -1, *x.size()[2:])
        x = nn.functional.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=groups)
        return x.view(batch_size, -1, *x.size()[2:])

class AffineTransform(nn.Module):
    def __init__(self, n_dim_w, n_feature_maps, equal_lr):
        super().__init__()
        self.fc = FC(n_dim_w, n_feature_maps, equal_lr=equal_lr)
        nn.init.ones_(self.fc.bias)
    def forward(self, w):
        return self.fc(w)

class FC(nn.Module):
    def __init__(self, n_dim_in, n_dim_out, equal_lr=True):
        super().__init__()
        norm_const = n_dim_in**-0.5
        scale_init = 1 if equal_lr else norm_const
        self.scale_forward = norm_const if equal_lr else 1 
        self.weight = nn.Parameter(scale_init * torch.randn(n_dim_out, n_dim_in))
        self.bias = nn.Parameter(torch.zeros(n_dim_out))
    def forward(self, x):
        return nn.functional.linear(x, self.scale_forward * self.weight, bias=self.bias)
