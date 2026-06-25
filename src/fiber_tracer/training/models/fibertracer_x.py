"""FiberTracer-X: a task-conditioned fiber analysis foundation model.

The design follows the research plan: a compact shared 3D encoder is
pre-trained on a large, domain-randomized synthetic corpus, and lightweight
task- / material-specific adapters are attached for downstream tasks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from fiber_tracer.backends.unet3d import _ConvBlock


class FiberTracerXEncoder(nn.Module):
    """Shared 3D encoder used by all FiberTracer-X task adapters.

    Parameters
    ----------
    in_channels:
        Number of input channels (1 for grayscale XCT).
    features:
        Number of features at each encoder level, e.g. ``(16, 32, 64)``.
    dropout:
        Spatial dropout probability after each conv block.
    norm:
        ``"batch"`` or ``"instance"`` normalization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] = (16, 32, 64),
        dropout: float = 0.0,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.features = tuple(features)
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        current = in_channels
        for feature in features:
            self.encoder_blocks.append(_ConvBlock(current, feature, dropout=dropout, norm=norm))
            current = feature

        self.bottleneck = _ConvBlock(features[-1], features[-1] * 2, dropout=dropout, norm=norm)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return encoder feature maps and bottleneck tensor."""
        encoder_outputs: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        return encoder_outputs, x


class Segmentation3DAdapter(nn.Module):
    """3D U-Net decoder adapter for semantic/instance segmentation."""

    def __init__(
        self,
        features: Sequence[int],
        out_channels: int = 1,
        dropout: float = 0.0,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        reversed_features = list(reversed(features))
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(reversed_features)):
            in_feat = reversed_features[i] * 2 if i == 0 else reversed_features[i - 1]
            out_feat = reversed_features[i]
            self.up_convs.append(nn.ConvTranspose3d(in_feat, out_feat, kernel_size=2, stride=2))
            self.decoder_blocks.append(
                _ConvBlock(out_feat * 2, out_feat, dropout=dropout, norm=norm)
            )
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    @staticmethod
    def _match_sizes(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape == target.shape:
            return x
        diff_d = target.shape[2] - x.shape[2]
        diff_h = target.shape[3] - x.shape[3]
        diff_w = target.shape[4] - x.shape[4]
        return nn.functional.pad(
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

    def forward(
        self,
        bottleneck: torch.Tensor,
        encoder_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        x = bottleneck
        for up_conv, decoder_block, enc_out in zip(
            self.up_convs, self.decoder_blocks, reversed(encoder_features)
        ):
            x = up_conv(x)
            x = self._match_sizes(x, enc_out)
            x = torch.cat([enc_out, x], dim=1)
            x = decoder_block(x)
        return self.final_conv(x)


class OrientationRegressorAdapter(nn.Module):
    """Regress the second-order orientation tensor A2 from the bottleneck.

    The output is the 6 unique components of the symmetric A2 tensor:
    (a11, a12, a13, a22, a23, a33).  These can be assembled into a full
    3x3 tensor by the loss function.
    """

    def __init__(self, features: Sequence[int], hidden_dim: int = 128) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        in_features = features[-1] * 2
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        _encoder_features: Sequence[torch.Tensor],
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pool(bottleneck)
        return self.mlp(x)

    @staticmethod
    def components_to_matrix(components: torch.Tensor) -> torch.Tensor:
        """Assemble (B, 6) components into a symmetric (B, 3, 3) tensor."""
        b = components.size(0)
        a11, a12, a13, a22, a23, a33 = components.unbind(dim=1)
        return torch.stack(
            [
                torch.stack([a11, a12, a13], dim=1),
                torch.stack([a12, a22, a23], dim=1),
                torch.stack([a13, a23, a33], dim=1),
            ],
            dim=1,
        ).reshape(b, 3, 3)


class FiberTracerX(nn.Module):
    """Task-conditioned FiberTracer-X model.

    Parameters
    ----------
    in_channels:
        Input channel count.
    features:
        Encoder feature channels.
    tasks:
        Mapping from task name to adapter configuration.  Example:
        ``{"segment": {"out_channels": 3}, "orient": {}}``.
    """

    def __init__(
        self,
        tasks: dict[str, dict[str, Any]] | None = None,
        in_channels: int = 1,
        features: Sequence[int] = (16, 32, 64),
        dropout: float = 0.0,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.encoder = FiberTracerXEncoder(
            in_channels=in_channels,
            features=features,
            dropout=dropout,
            norm=norm,
        )
        self.tasks = tasks or {}
        self.adapters = nn.ModuleDict()
        for task_name, task_cfg in self.tasks.items():
            self.adapters[task_name] = self._build_adapter(task_name, task_cfg, features)

    def _build_adapter(
        self,
        task_name: str,
        task_cfg: dict[str, Any],
        features: Sequence[int],
    ) -> nn.Module:
        if task_name == "segment":
            return Segmentation3DAdapter(
                features=features,
                out_channels=task_cfg.get("out_channels", 1),
            )
        if task_name == "orient":
            return OrientationRegressorAdapter(
                features=features,
                hidden_dim=task_cfg.get("hidden_dim", 128),
            )
        raise ValueError(f"Unknown task adapter: {task_name}")

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        if task not in self.adapters:
            raise KeyError(f"Task '{task}' not registered. Available: {list(self.adapters.keys())}")
        encoder_features, bottleneck = self.encoder(x)
        adapter = self.adapters[task]
        if task == "segment":
            return adapter(bottleneck, encoder_features)
        if task == "orient":
            return adapter(encoder_features, bottleneck)
        raise ValueError(f"Forward logic missing for task: {task}")

    def predict_segment(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method returning foreground probabilities."""
        logits = self.forward(x, task="segment")
        return torch.sigmoid(logits)
