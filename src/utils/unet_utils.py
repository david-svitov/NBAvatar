import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """Adaptive Instance Normalization conditioned on a pose embedding."""
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(cond_dim, num_features * 2)
        # Identity init: start as plain InstanceNorm (gamma=1, beta=0)
        nn.init.zeros_(self.fc.weight)
        nn.init.ones_(self.fc.bias[:num_features])
        nn.init.zeros_(self.fc.bias[num_features:])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        params = self.fc(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 2C, 1, 1)
        gamma, beta = params.chunk(2, dim=1)
        return gamma * x + beta


class PoseEncoder(nn.Module):
    """Maps concatenated pose params to a fixed-size conditioning embedding."""
    def __init__(self, pose_dim: int, embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout_p=0.0, use_residual=False):
        super(down, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.adain = AdaIN(out_ch, cond_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.use_residual = use_residual
        # 1x1 proj with stride=2 to match spatial dims after stride-2 conv
        self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=2) if use_residual else None

    def forward(self, x, cond):
        out = self.conv(x)
        out = self.adain(out, cond)
        out = self.act(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + self.proj(x)
        return out


class up(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, output_pad=0, concat=True, final=False,
                 dropout_p=0.0):
        super(up, self).__init__()
        self.concat = concat
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pad = nn.ZeroPad2d((0, output_pad, 0, output_pad)) if output_pad > 0 else nn.Identity()
        self.adain = AdaIN(out_ch, cond_dim)
        self.act = nn.Tanh() if final else nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x1, x2, cond):
        if self.concat:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x1 = torch.cat((x2, x1), dim=1)
        x1 = self.upsample(x1)
        x1 = self.conv(x1)
        x1 = self.pad(x1)
        x1 = self.adain(x1, cond)
        x1 = self.act(x1)
        x1 = self.dropout(x1)
        return x1


class UNet(nn.Module):
    def __init__(self, input_channels, pose_dim, embed_dim=256):
        super(UNet, self).__init__()
        self.pose_encoder = PoseEncoder(pose_dim, embed_dim)

        self.down1 = down(input_channels, 64, embed_dim)
        self.down2 = down(64, 128, embed_dim)
        self.down3 = down(128, 256, embed_dim)
        self.down4 = down(256, 512, embed_dim)
        self.down5 = down(512, 512, embed_dim)
        self.up1 = up(512, 512, embed_dim, output_pad=1, concat=False)
        self.up2 = up(1024, 512, embed_dim)
        self.up3 = up(768, 256, embed_dim)
        self.up4 = up(384, 128, embed_dim)
        self.up5 = up(192, 64, embed_dim)

        self.rgb_out = nn.Conv2d(64, 3, 3, padding=1)
        self.alpha_out = nn.Conv2d(64, 1, 3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x, pose_cond):
        cond = self.pose_encoder(pose_cond)  # (B, embed_dim)

        x1 = self.down1(x, cond)
        x2 = self.down2(x1, cond)
        x3 = self.down3(x2, cond)
        x4 = self.down4(x3, cond)
        x5 = self.down5(x4, cond)
        x = self.up1(x5, None, cond)
        x = self.up2(x, x4, cond)
        x = self.up3(x, x3, cond)
        x = self.up4(x, x2, cond)
        x = self.up5(x, x1, cond)

        rgb = self.activation(self.rgb_out(x))
        alpha = self.activation(self.alpha_out(x))
        return rgb, alpha


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.adain = AdaIN(out_ch, cond_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, cond):
        return self.act(self.adain(self.conv(x), cond))


class MinimalRenderer(nn.Module):
    """Fully convolutional renderer with no downsampling. RF = 1 + num_layers*2 pixels."""
    def __init__(self, input_channels, pose_dim, embed_dim=256, num_layers=4):
        super().__init__()
        self.pose_encoder = PoseEncoder(pose_dim, embed_dim)
        self.convs = nn.ModuleList(
            [ConvBlock(input_channels, 64, embed_dim)] +
            [ConvBlock(64, 64, embed_dim) for _ in range(num_layers - 1)]
        )
        self.rgb_out = nn.Conv2d(64, 3, 1)
        self.alpha_out = nn.Conv2d(64, 1, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, pose_cond):
        cond = self.pose_encoder(pose_cond)
        for conv in self.convs:
            x = conv(x, cond)
        return self.activation(self.rgb_out(x)), self.activation(self.alpha_out(x))


class UNetSmall(nn.Module):
    def __init__(self, input_channels, pose_dim, embed_dim=256, dropout_p=0.0,
                 use_residual=False):
        super().__init__()
        self.pose_encoder = PoseEncoder(pose_dim, embed_dim)

        self.down1 = down(input_channels, 64, embed_dim, dropout_p=dropout_p,
                          use_residual=use_residual)
        self.down2 = down(64, 128, embed_dim, dropout_p=dropout_p,
                          use_residual=use_residual)
        self.down3 = down(128, 256, embed_dim, dropout_p=dropout_p,
                          use_residual=use_residual)
        self.up1 = up(256, 256, embed_dim, concat=False, dropout_p=dropout_p)
        self.up2 = up(384, 128, embed_dim, dropout_p=dropout_p)   # 256 + 128 skip
        self.up3 = up(192, 64, embed_dim, dropout_p=dropout_p)    # 128 + 64 skip

        self.rgb_out = nn.Conv2d(64, 3, 3, padding=1)
        self.alpha_out = nn.Conv2d(64, 1, 3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x, pose_cond):
        cond = self.pose_encoder(pose_cond)

        x1 = self.down1(x, cond)
        x2 = self.down2(x1, cond)
        x3 = self.down3(x2, cond)
        x = self.up1(x3, None, cond)
        x = self.up2(x, x2, cond)
        x = self.up3(x, x1, cond)

        rgb = self.activation(self.rgb_out(x))
        alpha = self.activation(self.alpha_out(x))
        return rgb, alpha
