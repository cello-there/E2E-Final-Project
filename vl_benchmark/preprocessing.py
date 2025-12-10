# vl_benchmark/preprocessing.py

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


@dataclass
class PreprocessConfig:
    """
    Shared (model-agnostic) image preprocessing config.

    This is used when cfg.mode == "shared", to put all models
    under the same preprocessing regime for controlled comparison.
    """
    image_size: int = 224
    resize_shorter_side: int = 256
    center_crop: bool = True

    # CLIP-style normalization; good generic choice
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


def build_image_transform(cfg: PreprocessConfig | None = None) -> transforms.Compose:
    """
    Builds a torchvision transform that:
      - converts PIL or uint8 tensor to float tensor in [0, 1]
      - resizes the shorter side to cfg.resize_shorter_side
      - center-crops (or resizes) to cfg.image_size
      - normalizes with cfg.mean/std
    """
    if cfg is None:
        cfg = PreprocessConfig()

    t_list: List[transforms.Compose] = [
        # Ensure we have a float tensor image in [0, 1]
        transforms.Lambda(
            lambda img: transforms.functional.to_tensor(img)
            if isinstance(img, Image.Image) or img.dtype != torch.float32
            else img
        ),
        transforms.Resize(
            cfg.resize_shorter_side,
            interpolation=InterpolationMode.BICUBIC,
        ),
    ]

    if cfg.center_crop:
        t_list.append(transforms.CenterCrop(cfg.image_size))
    else:
        t_list.append(
            transforms.Resize(
                (cfg.image_size, cfg.image_size),
                interpolation=InterpolationMode.BICUBIC,
            )
        )

    t_list.extend(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )

    return transforms.Compose(t_list)


class TextPreprocessor:
    """
    Generic wrapper around tokenizers, used for shared mode.

    tokenizers: dict mapping model_key -> tokenizer object.
      For HuggingFace-like tokenizers, we call them with
        (texts, padding=True, truncation=True, return_tensors="pt")
      For OpenCLIP-like tokenizers, we call them as token_ids = tok(texts)
    """

    def __init__(self, tokenizers: Dict[str, Any], max_length: int = 77):
        self.tokenizers = tokenizers
        self.max_length = max_length

    def __call__(
        self,
        texts: List[str],
        model_key: str,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        tok = self.tokenizers[model_key]

        # HuggingFace-style tokenizer
        if hasattr(tok, "__call__") and hasattr(tok, "pad_token_id"):
            out = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {k: v.to(device) for k, v in out.items() if isinstance(v, torch.Tensor)}

        # OpenCLIP-style tokenizer (returns a tensor or list of ids)
        token_ids = tok(texts)
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        return {"input_ids": token_ids.to(device)}
