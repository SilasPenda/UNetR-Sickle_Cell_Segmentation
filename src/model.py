import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from src.utils import get_config


class ConvBlock(nn.Module):  # Output size=(Input size − Kernel size + 2⋅Padding)+1
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):  # Output size=(Input size−1)⋅Stride−2⋅Padding+Kernel size
    def __init__(self, in_c, out_c):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)


class UNETR_2D(nn.Module):
    def __init__(self, config_filepath=os.path.join(os.getcwd(), "config.yaml")):
        super().__init__()
        config = get_config(config_filepath)
        self.image_size = config.get('image_size', 256)
        self.num_layers = config.get('num_layers', 12)
        self.hidden_dim = config.get('hidden_dim', 768)
        self.mlp_dim = config.get('mlp_dim', 3072)
        self.num_heads = config.get('num_heads', 12)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.num_patches = config.get('num_patches', 256)
        self.patch_size = config.get('patch_size', 16)
        self.num_channels = config.get('num_channels', 3)

        # Preprocessing layer: Creating patches and flattening them
        self.patch_dim = self.patch_size * self.patch_size * self.num_channels
        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.positions = nn.Parameter(torch.randn(self.num_patches, self.hidden_dim))

        """ Transformer Encoder """
        self.trans_encoder_layers = nn.ModuleList()

        for i in range(self.num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.mlp_dim,
                dropout=self.dropout_rate,
                activation=nn.GELU(),
                batch_first=True
            )
            self.trans_encoder_layers.append(layer)

        """ CNN Decoder """
        # Decoder 1
        self.d1 = DeconvBlock(self.hidden_dim, 512)
        self.s1 = nn.Sequential(
            DeconvBlock(self.hidden_dim, 512),
            ConvBlock(512, 512)
        )
        self.c1 = nn.Sequential(
            ConvBlock(512 + 512, 512),
            ConvBlock(512, 512)
        )

        # Decoder 2
        self.d2 = DeconvBlock(512, 256)
        self.s2 = nn.Sequential(
            DeconvBlock(self.hidden_dim, 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 256),
            ConvBlock(256, 256)
        )
        self.c2 = nn.Sequential(
            ConvBlock(256 + 256, 256),
            ConvBlock(256, 256)
        )

        # Decoder 3
        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(self.hidden_dim, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128)
        )
        self.c3 = nn.Sequential(
            ConvBlock(128 + 128, 128),
            ConvBlock(128, 128)
        )

        # Decoder 4
        self.d4 = DeconvBlock(128, 64)
        self.s4 = nn.Sequential(
            ConvBlock(self.num_channels, 64),
            ConvBlock(64, 64)
        )
        self.c4 = nn.Sequential(
            ConvBlock(64 + 64, 64),
            ConvBlock(64, 64)
        )

        # Output
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Patch + Position Embeddings """
        batch_size = inputs.size(0)
        x = self.patch_embedding(inputs)  # [B, hidden_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        x = x + self.positions

        """ Transformer Encoder """
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        for i in range(self.num_layers):
            layer = self.trans_encoder_layers[i]
            x = layer(x)

            if (i + 1) in skip_connection_index:
                skip_connections.append(x)

        """ CNN Decoder """
        z3, z6, z9, z12 = skip_connections

        # Reshaping
        z0 = inputs
        z3 = z3.permute(0, 2, 1).view(batch_size, self.hidden_dim, self.patch_size, self.patch_size)
        z6 = z6.permute(0, 2, 1).view(batch_size, self.hidden_dim, self.patch_size, self.patch_size)
        z9 = z9.permute(0, 2, 1).view(batch_size, self.hidden_dim, self.patch_size, self.patch_size)
        z12 = z12.permute(0, 2, 1).view(batch_size, self.hidden_dim, self.patch_size, self.patch_size)

        # Decoder 1
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        # Decoder 2
        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        # Decoder 3
        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        # Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        # Return output
        return self.output(x)


if __name__ == "__main__":
    config = {
        "image_size": 256,
        "num_layers": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "num_patches": 256,
        "patch_size": 16,
        "num_channels": 3
    }

    x = torch.randn((8, config["num_channels"], config["image_size"], config["image_size"]))
    model = UNETR_2D()
    output = model(x)
    print(output.shape)
