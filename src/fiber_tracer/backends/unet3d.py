"""Lightweight 3D U-Net for fiber segmentation.

The network is intentionally small so it can be trained on CPU with the
synthetic phantom generator shipped in ``fiber_tracer.validation.phantoms``.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Two 3×3×3 convolutions with ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # type: ignore[no-any-return]


class UNet3D(nn.Module):
    """3D U-Net for binary semantic segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale XCT).
    out_channels : int
        Number of output channels (1 for binary foreground probability).
    features : Sequence[int]
        Number of features at each encoder level, e.g. ``(8, 16, 32)``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (8, 16, 32),
    ) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        current = in_channels
        for feature in features:
            self.encoder_blocks.append(_ConvBlock(current, feature))
            current = feature

        self.bottleneck = _ConvBlock(features[-1], features[-1] * 2)

        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        for i in range(len(reversed_features)):
            in_feat = reversed_features[i] * 2 if i == 0 else reversed_features[i - 1]
            out_feat = reversed_features[i]
            self.up_convs.append(nn.ConvTranspose3d(in_feat, out_feat, kernel_size=2, stride=2))
            self.decoder_blocks.append(_ConvBlock(out_feat * 2, out_feat))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_conv, decoder_block, enc_out in zip(
            self.up_convs, self.decoder_blocks, reversed(encoder_outputs)
        ):
            x = up_conv(x)
            # If spatial sizes do not match due to odd input dimensions, crop
            # the larger feature map to the smaller one.
            if x.shape != enc_out.shape:
                diff_d = enc_out.shape[2] - x.shape[2]
                diff_h = enc_out.shape[3] - x.shape[3]
                diff_w = enc_out.shape[4] - x.shape[4]
                x = nn.functional.pad(
                    x,
                    [
                        diff_w // 2,
                        diff_w - diff_w // 2,
                        diff_h // 2,
                        diff_h - diff_h // 2,
                        diff_d // 2,
                        diff_d - diff_d // 2,
                    ],
                )
            x = torch.cat([enc_out, x], dim=1)
            x = decoder_block(x)

        return torch.sigmoid(self.final_conv(x))

    @torch.no_grad()
    def predict_volume(
        self,
        volume: np.ndarray,
        patch_size: tuple[int, int, int] = (32, 32, 32),
        overlap: int = 16,
        batch_size: int = 1,
    ) -> np.ndarray:
        """Run sliding-window inference on a 3D volume.

        Overlapping predictions are averaged. The input is expected to be a
        3D numpy array; it is normalized internally to ``[0, 1]``.
        """
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D")

        device = next(self.parameters()).device
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        full_shape = volume_norm.shape

        # Pad volume so that every dimension is at least one patch long.
        pad_d = max(0, patch_size[0] - full_shape[0])
        pad_h = max(0, patch_size[1] - full_shape[1])
        pad_w = max(0, patch_size[2] - full_shape[2])
        padded = np.pad(
            volume_norm,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
        )
        padded_shape = padded.shape

        output = np.zeros(padded_shape, dtype=np.float32)
        counts = np.zeros(padded_shape, dtype=np.float32)

        stride = tuple(max(1, ps - overlap) for ps in patch_size)
        for d in range(0, padded_shape[0] - patch_size[0] + 1, stride[0]):
            for h in range(0, padded_shape[1] - patch_size[1] + 1, stride[1]):
                for w in range(0, padded_shape[2] - patch_size[2] + 1, stride[2]):
                    patch = padded[
                        d : d + patch_size[0],
                        h : h + patch_size[1],
                        w : w + patch_size[2],
                    ]
                    patch_tensor = (
                        torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    )
                    pred = self(patch_tensor).squeeze().cpu().numpy()
                    output[
                        d : d + patch_size[0],
                        h : h + patch_size[1],
                        w : w + patch_size[2],
                    ] += pred
                    counts[
                        d : d + patch_size[0],
                        h : h + patch_size[1],
                        w : w + patch_size[2],
                    ] += 1

        # Avoid division by zero; counts is always positive for valid inputs.
        output /= counts
        # Crop back to original size.
        return output[: full_shape[0], : full_shape[1], : full_shape[2]]
