from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _ensure_repo_imports(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class SingleImageVDA(nn.Module):
    """Single-image VDA wrapper that keeps the pretrained VDA weights intact.

    This wrapper removes the original sliding-window sequence logic from the
    inference path and runs the model with temporal length fixed to 1. It also
    exposes a global-feature extractor for the distillation head.
    """

    def __init__(
        self,
        repo_root: str | Path,
        checkpoint_path: str | Path,
        encoder: str = "vitl",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        input_size: int = 518,
        max_res: int = 1280,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.input_size = int(input_size)
        self.max_res = int(max_res)
        self.device = torch.device(device)

        _ensure_repo_imports(self.repo_root)
        from video_depth_anything.video_depth import VideoDepthAnything
        from video_depth_anything.util.transform import (  # type: ignore
            NormalizeImage,
            PrepareForNet,
            Resize,
        )
        from torchvision.transforms import Compose

        self._Resize = Resize
        self._NormalizeImage = NormalizeImage
        self._PrepareForNet = PrepareForNet
        self._Compose = Compose

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": features, "out_channels": list(out_channels)},
        }
        if encoder not in model_configs:
            raise ValueError(f"Unsupported encoder: {encoder}")

        self.vda = VideoDepthAnything(**model_configs[encoder], metric=False)
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        self.vda.load_state_dict(state_dict, strict=True)
        self.vda = self.vda.to(self.device).eval()
        self.encoder_name = encoder
        self.feature_dim = self.vda.pretrained.embed_dim
        self.transform = self._build_transform(self.input_size)

    def _build_transform(self, input_size: int):
        return self._Compose(
            [
                self._Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                self._NormalizeImage(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                self._PrepareForNet(),
            ]
        )

    def freeze_backbone(self) -> None:
        for param in self.vda.pretrained.parameters():
            param.requires_grad = False

    def freeze_decoder(self) -> None:
        for param in self.vda.head.parameters():
            param.requires_grad = False

    def freeze_all(self) -> None:
        self.freeze_backbone()
        self.freeze_decoder()

    @staticmethod
    def load_rgb_image(image_path: str | Path) -> np.ndarray:
        image_path = str(image_path)
        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _maybe_resize_long_side(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.max_res <= 0:
            return image_rgb
        height, width = image_rgb.shape[:2]
        long_side = max(height, width)
        if long_side <= self.max_res:
            return image_rgb
        scale = self.max_res / float(long_side)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        return cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def _prepare_single(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        image_rgb = self._maybe_resize_long_side(image_rgb)
        height, width = image_rgb.shape[:2]
        transformed = self.transform({"image": image_rgb.astype(np.float32) / 255.0})["image"]
        tensor = torch.from_numpy(transformed).unsqueeze(0).to(self.device)
        return tensor, (height, width)

    def prepare_batch(
        self,
        images: Sequence[np.ndarray],
    ) -> tuple[torch.Tensor, List[tuple[int, int]]]:
        tensors: List[torch.Tensor] = []
        original_sizes: List[tuple[int, int]] = []
        for image in images:
            tensor, original_size = self._prepare_single(image)
            tensors.append(tensor)
            original_sizes.append(original_size)
        squeezed = [tensor.squeeze(0) for tensor in tensors]
        max_h = max(tensor.shape[-2] for tensor in squeezed)
        max_w = max(tensor.shape[-1] for tensor in squeezed)
        padded: List[torch.Tensor] = []
        for tensor in squeezed:
            pad_h = max_h - tensor.shape[-2]
            pad_w = max_w - tensor.shape[-1]
            if pad_h > 0 or pad_w > 0:
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            padded.append(tensor)
        batch = torch.stack(padded, dim=0)
        return batch, original_sizes

    def extract_global_features(self, batch: torch.Tensor) -> torch.Tensor:
        self.vda.eval()
        with torch.no_grad():
            features = self.vda.pretrained.get_intermediate_layers(
                batch,
                self.vda.intermediate_layer_idx[self.encoder_name],
                return_class_token=True,
            )
        last_patch_tokens = features[-1][0]
        return last_patch_tokens.mean(dim=1)

    def predict_relative_depth(self, batch: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = batch.shape[-2] // 14, batch.shape[-1] // 14
        with torch.no_grad():
            features = self.vda.pretrained.get_intermediate_layers(
                batch,
                self.vda.intermediate_layer_idx[self.encoder_name],
                return_class_token=True,
            )
            depth = self.vda.head(features, patch_h, patch_w, frame_length=1)[0]
            depth = F.interpolate(
                depth,
                size=(batch.shape[-2], batch.shape[-1]),
                mode="bilinear",
                align_corners=True,
            )
            depth = F.relu(depth)
        return depth.squeeze(1)

    def infer_single_image(
        self,
        image_rgb: np.ndarray,
    ) -> dict[str, np.ndarray | torch.Tensor | tuple[int, int]]:
        batch, original_sizes = self.prepare_batch([image_rgb])
        global_features = self.extract_global_features(batch)
        relative_depth = self.predict_relative_depth(batch)[0]
        original_h, original_w = original_sizes[0]
        relative_depth = F.interpolate(
            relative_depth.unsqueeze(0).unsqueeze(0),
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        return {
            "global_features": global_features[0],
            "relative_depth": relative_depth,
            "original_size": original_sizes[0],
        }
