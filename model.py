import torch
import torch.nn as nn
import torch.nn.functional as F
from wtconv2d import WTConv2d
from util import wavelet

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return x * self.weight


class SepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        bias=True,
        padding_mode="zeros",
        wt_levels=3,
        wt_type="db1",
        wt_init=0.1,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.wt_levels = wt_levels

        self.base_conv = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channel,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.base_scale = _ScaleModule([1, in_channel, 1, 1], init_scale=1.0)

        wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(
            wt_type, in_channel, in_channel, torch.float
        )
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channel * 4,
                    in_channel * 4,
                    kernel_size,
                    padding="same",
                    stride=1,
                    dilation=1,
                    groups=in_channel * 4,
                    bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channel * 4, 1, 1], init_scale=wt_init) for _ in range(self.wt_levels)]
        )

        self.wt_alpha = nn.Parameter(torch.zeros(1))

        self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride) if stride > 1 else None

        self.pw = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias)

    def _wt_branch(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        return next_x_ll

    def forward(self, x):
        base = self.base_scale(self.base_conv(x))
        wt = self._wt_branch(x)
        alpha = torch.sigmoid(self.wt_alpha)
        fused = base + alpha * wt

        if self.do_stride is not None:
            fused = self.do_stride(fused)

        return self.pw(fused)


class WaveletEnhanceBlockWT(nn.Module):
    def __init__(self, channels, wt_levels=2, wt_type="db1", refine_kernel=3):
        super().__init__()
        self.wt = WTConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            stride=1,
            bias=True,
            wt_levels=wt_levels,
            wt_type=wt_type,
        )
        self.refine = SepConv(channels, channels, kernel_size=refine_kernel, bias=False)

        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.wt(x)
        y = self.refine(y)
        a = torch.sigmoid(self.alpha)
        return x + a * y


class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, relu_slope=0.1):
        super(BasicBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = SepConv(in_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = SepConv(out_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out = out + self.identity(x)
        return out


class GetGradient(nn.Module):
    def __init__(self, dim=3, mode="sobel"):
        super(GetGradient, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode == "sobel":
            kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

            kernel_y = torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            kernel_x = torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            self.register_buffer("kernel_y", kernel_y.repeat(self.dim, 1, 1, 1))
            self.register_buffer("kernel_x", kernel_x.repeat(self.dim, 1, 1, 1))
        elif mode == "laplacian":
            kernel_laplace = [[0.25, 1, 0.25], [1, -5, 1], [0.25, 1, 0.25]]
            kernel_laplace = torch.tensor(kernel_laplace, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer("kernel_laplace", kernel_laplace.repeat(self.dim, 1, 1, 1))

    def forward(self, x):
        if self.mode == "sobel":
            grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=self.dim)
            grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=self.dim)
            grad_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-6)
        else:
            grad_magnitude = F.conv2d(x, self.kernel_laplace, padding=1, groups=self.dim)
            grad_magnitude = torch.abs(grad_magnitude)
        return grad_magnitude


class REAMGuideLite(nn.Module):
    def __init__(self, mid_ch):
        super().__init__()
        self.conv_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.edge_head = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1, 1, 0),
        )

    def forward(self, mid):
        avg_mid = torch.mean(mid, dim=1, keepdim=True)
        max_mid, _ = torch.max(mid, dim=1, keepdim=True)
        a = torch.cat([avg_mid, max_mid], dim=1)

        gate = torch.sigmoid(self.conv_gate(a))
        edge = torch.sigmoid(self.edge_head(mid))
        return gate, edge


class SGFB(nn.Module):
    def __init__(self, feature_channels=48):
        super(SGFB, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.frdb1 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.frdb2 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.get_gradient = GetGradient(feature_channels, mode="sobel")
        self.conv_grad = nn.Sequential(
            SepConv(feature_channels, feature_channels, kernel_size=3, bias=False),
            nn.Sigmoid(),
        )

        self.beta_gate = nn.Parameter(torch.zeros(1))
        self.beta_edge = nn.Parameter(torch.zeros(1))

    def forward(self, x, guide_gate=None, guide_edge=None):
        grad = self.get_gradient(x)
        grad = self.conv_grad(grad)

        if guide_gate is not None:
            g = F.interpolate(guide_gate, size=x.shape[-2:], mode="bilinear", align_corners=False)
            g = g.expand(-1, x.shape[1], -1, -1)
            w = torch.sigmoid(self.beta_gate)
            grad = grad * ((1 - w) + w * g)

        if guide_edge is not None:
            e = F.interpolate(guide_edge, size=x.shape[-2:], mode="bilinear", align_corners=False)
            e = e.expand(-1, x.shape[1], -1, -1)
            w2 = torch.sigmoid(self.beta_edge)
            grad = grad * ((1 - w2) + w2 * e)

        x = self.frdb1(x)
        alpha = torch.sigmoid(self.alpha)
        x = alpha * grad * x + (1 - alpha) * x
        x = self.frdb2(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, feature_channels=48, wt_levels=2, wt_type="db1"):
        super(BasicLayer, self).__init__()
        self.web = WaveletEnhanceBlockWT(
            feature_channels, wt_levels=wt_levels, wt_type=wt_type, refine_kernel=3
        )
        self.sgfb = SGFB(feature_channels)

    def forward(self, x, guide_gate=None, guide_edge=None):
        res = x
        x = self.web(x)
        x = self.sgfb(x, guide_gate=guide_gate, guide_edge=guide_edge)
        return 0.5 * x + 0.5 * res


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MiniNAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dwc = c * dw_expand
        self.conv1 = nn.Conv2d(c, dwc, 1, 1, 0)
        self.dwconv = nn.Conv2d(dwc, dwc, 3, 1, 1, groups=dwc)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(dwc // 2, c, 1, 1, 0)

        ffnc = c * ffn_expand
        self.ffn1 = nn.Conv2d(c, ffnc, 1, 1, 0)
        self.ffn2 = nn.Conv2d(ffnc // 2, c, 1, 1, 0)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.conv2(self.sg(self.dwconv(self.conv1(x))))
        x = x + y * self.beta
        y2 = self.ffn2(self.sg(self.ffn1(x)))
        x = x + y2 * self.gamma
        return x


class ColorPriorWhiteBalance(nn.Module):
    def __init__(self, ch=3, hidden=24, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.prior_enc = nn.Sequential(
            nn.Conv2d(ch, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            MiniNAFBlock(hidden),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.to_gain = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, ch, 1, 1, 0),
        )
        self.to_bias = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, ch, 1, 1, 0),
        )

        self.alpha = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True).expand_as(x)
        prior = self.prior_enc(x_mean)

        gain = torch.sigmoid(self.to_gain(prior))
        gain = 0.5 + gain
        bias = torch.tanh(self.to_bias(prior)) * 0.1

        x_wb = x * gain + bias
        x_wb = torch.clamp(x_wb, 0.0, 1.0)

        a = torch.sigmoid(self.alpha)
        out = a * x_wb + (1 - a) * x
        return out


class myModel(nn.Module):
    def __init__(self, in_channels=3, feature_channels=32, use_white_balance=False, wt_levels=2, wt_type="db1"):
        super(myModel, self).__init__()
        self.use_white_balance = use_white_balance
        if self.use_white_balance:
            self.wb = ColorPriorWhiteBalance(ch=in_channels)

        self.first = nn.Conv2d(in_channels, feature_channels, kernel_size=3, stride=1, padding=1)

        self.encoder1 = BasicLayer(feature_channels, wt_levels=wt_levels, wt_type=wt_type)
        self.down1 = Downsample(feature_channels)

        self.encoder2 = BasicLayer(feature_channels * 2**1, wt_levels=wt_levels, wt_type=wt_type)
        self.down2 = Downsample(feature_channels * 2**1)

        self.bottleneck = BasicLayer(feature_channels * 2**2, wt_levels=wt_levels, wt_type=wt_type)

        self.ream_guide = REAMGuideLite(mid_ch=feature_channels * 2**2)

        self.up1 = Upsample(feature_channels * 2**2)
        self.decoder1 = BasicLayer(feature_channels * 2**1, wt_levels=wt_levels, wt_type=wt_type)

        self.up2 = Upsample(feature_channels * 2**1)
        self.decoder2 = BasicLayer(feature_channels, wt_levels=wt_levels, wt_type=wt_type)

        self.out = nn.Conv2d(feature_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        if self.use_white_balance:
            x = self.wb(x)

        x1 = self.first(x)

        x1 = self.encoder1(x1)

        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        guide_gate, guide_edge = self.ream_guide(x3)

        x = self.up1(x3) + x2
        x = self.decoder1(x, guide_gate=guide_gate, guide_edge=guide_edge)

        x = self.up2(x) + x1
        x = self.decoder2(x, guide_gate=guide_gate, guide_edge=guide_edge)

        out = self.out(x) + res
        return out


if __name__ == "__main__":
    dummy_img = torch.rand(1, 3, 128, 128)
    model = myModel(use_white_balance=True, wt_levels=2, wt_type="db1")
    output_img = model(dummy_img)
    print("Output image shape:", output_img.shape)