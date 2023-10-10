import torch
import torch.nn as nn
import torch.nn.functional as F

import utilities

CONV2D = {
    "conv": nn.Conv2d,
    "batchnorm": nn.BatchNorm2d,
    "tconv": nn.ConvTranspose2d,
    "stride": 2,
    "outputpadding": 1,
}

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


def create_convolution(
    input_channels,
    output_channels,
    kernel_size,
    stride,
    layer,
    weight_norm,
    batch_norm=None,
    output_padding=None,
):
    output = weight_norm(
        layer(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
        )
        if not output_padding
        else layer(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            output_padding=output_padding,
        )
    )
    if batch_norm is None:
        return output

    return nn.Sequential(*[output, batch_norm(output_channels), nn.ReLU(inplace=True)])


class ConvEncoderDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # pylint: disable=unnecessary-lambda
        weight_norm = lambda x: nn.utils.weight_norm(x)
        # pylint: enable=unnecessary-lambda
        kernel_size = 3
        layers = CONV2D
        self.args = args

        self.encoder1 = create_convolution(
            args.input_channels,
            args.dimension,
            kernel_size,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
        )
        self.encoder2 = create_convolution(
            args.dimension,
            2 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
        )
        self.encoder3 = create_convolution(
            2 * args.dimension,
            4 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
        )
        self.encoder4 = create_convolution(
            4 * args.dimension,
            8 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
        )

        self.decoder1 = create_convolution(
            8 * args.dimension,
            4 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["tconv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            output_padding=layers["outputpadding"],
        )
        self.projection1 = create_convolution(
            8 * args.dimension,
            4 * args.dimension,
            1,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
        )
        self.decoder2 = create_convolution(
            4 * args.dimension,
            2 * args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["tconv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            output_padding=layers["outputpadding"],
        )
        self.projection2 = create_convolution(
            4 * args.dimension,
            2 * args.dimension,
            1,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
        )
        self.decoder3 = create_convolution(
            2 * args.dimension,
            args.dimension,
            kernel_size,
            layers["stride"],
            layer=layers["tconv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
            output_padding=layers["outputpadding"],
        )
        self.projection3 = create_convolution(
            2 * args.dimension,
            args.dimension,
            1,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=layers["batchnorm"],
        )

        self.restoration = create_convolution(
            args.dimension,
            args.output_channels,
            kernel_size,
            1,
            layer=layers["conv"],
            weight_norm=weight_norm,
            batch_norm=None,
        )

        if args.global_channels > 0:
            self.fc = nn.Linear(args.global_channels, 8 * args.dimension)

    def forward(self, x, z=None, statistics=None):
        x_e1 = self.encoder1(x)
        x_e2 = self.encoder2(x_e1)
        x_e3 = self.encoder3(x_e2)
        x_e4 = self.encoder4(x_e3)

        if z is not None:
            z = F.relu(self.fc(z)).unsqueeze(-1).unsqueeze(-1)
            x_e4 = x_e4 + z

        x_d1 = self.decoder1(x_e4)
        if 1 in self.args.skip:
            x_d1 = self.projection1(torch.cat([x_d1, x_e3], dim=1))

        x_d2 = self.decoder2(x_d1)
        if 2 in self.args.skip:
            x_d2 = self.projection2(torch.cat([x_d2, x_e2], dim=1))

        x_d3 = self.decoder3(x_d2)
        if 3 in self.args.skip:
            x_d3 = self.projection3(torch.cat([x_d3, x_e1], dim=1))

        x = self.restoration(x_d3)
        if statistics is not None:
            if isinstance(statistics, tuple):
                mean, std = statistics
                return utilities.denormalize_array(x, mean, std)
            elif isinstance(statistics, list):
                means = [l[0] for l in statistics]
                stds = [l[1] for l in statistics]
                return utilities.denormalize_multi_dim_array(x, means, stds)
        return x
    

class JointSVFAndTmrtModel(nn.Module):
    svf_veg_indices = [9, 12, 15, 18, 20]
    svf_all_veg_indices = [8, 9, 11, 12, 14, 15, 17, 18, 19, 20]

    svf_without_aveg_indices = [8, 10, 12, 14, 15]

    def __init__(self, svf_model: nn.Module, tmrt_model: nn.Module) -> None:
        super().__init__()
        self.svf_model = svf_model
        self.tmrt_model = tmrt_model

        if tmrt_model.args.input_channels == 16:
            self.svf_indices = self.svf_without_aveg_indices
        elif tmrt_model.args.input_channels == 21:
            if svf_model.args.output_channels == 5:
                self.svf_indices = self.svf_veg_indices
            else:
                self.svf_indices = self.svf_all_veg_indices
        else:
            raise NotImplementedError

    def forward(self, dsm_veg, spatial, temporal=None, statistics=None):
        new_svf_veg = self.forward_veg_to_svf(dsm_veg=dsm_veg)
        spatial[:, self.svf_indices] = new_svf_veg
        return self.forward_tmrt(spatial=spatial, temporal=temporal, statistics=statistics)

    def forward_veg_to_svf(self, dsm_veg):
        return self.svf_model(dsm_veg)

    def forward_tmrt(self, spatial, temporal=None, statistics=None):
        if temporal is None:
            return self.tmrt_model(spatial, statistics=statistics)
        return self.tmrt_model(spatial, temporal, statistics=statistics)
