# vl_benchmark/models.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Callable

import time
import torch
import torch.nn as nn

from .preprocessing import PreprocessConfig, build_image_transform, TextPreprocessor


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@dataclass
class ModelConfig:
    """
    Shared config for all VLM wrappers.

    mode:
      - "native": use model-specific preprocessing/tokenization
      - "shared": use a unified preprocessing pipeline
    """
    name: str                       # "openclip", "siglip2", "vlm2vecv2"
    device: torch.device
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    mode: Literal["native", "shared"] = "native"
    shared_preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


class VLModelWrapper(nn.Module):
    """
    Unified wrapper for a vision-language model.

    It:
      - chooses between native vs shared preprocessing based on cfg.mode
      - exposes encode_images / encode_texts
      - optionally measures latency + peak VRAM
    """

    def __init__(
        self,
        cfg: ModelConfig,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        native_image_preprocess: Callable[[Any], torch.Tensor],
        native_text_preprocess: Callable[[List[str], torch.device], Dict[str, torch.Tensor]],
        shared_image_preprocess: Callable[[Any], torch.Tensor],
        shared_text_preprocess: Callable[[List[str], torch.device], Dict[str, torch.Tensor]],
    ):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        self.image_encoder = image_encoder.to(self.device)
        self.text_encoder = text_encoder.to(self.device)

        # Store both native and shared preprocessing pipelines
        self.native_image_pre = native_image_preprocess
        self.native_text_pre = native_text_preprocess
        self.shared_image_pre = shared_image_preprocess
        self.shared_text_pre = shared_text_preprocess

        # Choose active preprocessing mode
        if cfg.mode == "native":
            self.image_preprocess = self.native_image_pre
            self.text_preprocess = self.native_text_pre
        else:
            self.image_preprocess = self.shared_image_pre
            self.text_preprocess = self.shared_text_pre

        self._autocast_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": None,
        }[cfg.precision]

    @property
    def model_key(self) -> str:
        return self.cfg.name

    # ----------------------- Public API -----------------------

    def encode_images(
        self,
        images: Any,               # list[PIL.Image] or batch tensor
        normalize: bool = True,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        images:
          - Whatever the active image_preprocess expects:
            * native mode: usually list of PIL.Image
            * shared mode: also list of PIL.Image (recommended)

        Returns: (embeddings [B, D], stats or None)
        """
        self.eval()
        stats = None

        with torch.no_grad():
            # --- Preprocessing on CPU ---
            imgs = self.image_preprocess(images)  # [B, C, H, W] on CPU
            if imgs.device.type != self.device.type:
                imgs = imgs.to(self.device, non_blocking=True)

            # --- Efficiency measurement setup ---
            if return_stats and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                start = time.perf_counter()

            # --- Forward with autocast if enabled ---
            if self._autocast_dtype is not None and self.device.type == "cuda":
                ctx = torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype)
            else:
                ctx = torch.cpu.amp.autocast(enabled=False)

            with ctx:
                img_embeds = self._encode_images_impl(imgs)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            if return_stats and self.device.type == "cuda":
                end = time.perf_counter()
                peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                stats = {
                    "latency_s": end - start,
                    "peak_mem_mb": peak_mem,
                    "batch_size": imgs.size(0),
                }

        if normalize:
            img_embeds = l2_normalize(img_embeds, dim=-1)

        return img_embeds, stats

    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        texts: list of raw strings

        Uses the active text_preprocess (native or shared).
        """
        self.eval()
        stats = None

        with torch.no_grad():
            toks = self.text_preprocess(texts, self.device)

            if return_stats and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                start = time.perf_counter()

            if self._autocast_dtype is not None and self.device.type == "cuda":
                ctx = torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype)
            else:
                ctx = torch.cpu.amp.autocast(enabled=False)

            with ctx:
                txt_embeds = self._encode_texts_impl(toks)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            if return_stats and self.device.type == "cuda":
                end = time.perf_counter()
                peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                stats = {
                    "latency_s": end - start,
                    "peak_mem_mb": peak_mem,
                    "batch_size": len(texts),
                }

        if normalize:
            txt_embeds = l2_normalize(txt_embeds, dim=-1)

        return txt_embeds, stats

    # ----------------- Internal hooks (overrideable) -----------------

    def _encode_images_impl(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Default image encoding path.
        """
        return self.image_encoder(imgs)

    def _encode_texts_impl(self, toks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Default text encoding path.
        """
        return self.text_encoder(**toks)


# =====================================================================
#  OpenCLIP wrapper
# =====================================================================

class OpenCLIPTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.encode_text(input_ids)


def build_openclip_wrapper(
    cfg: ModelConfig,
    model_name: str,
    pretrained: str,
) -> VLModelWrapper:
    import open_clip

    # Load model + native transforms
    model, _, native_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=cfg.device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # --------------------------
    # Native preprocessing
    # --------------------------
    def native_img_pre(images: Any) -> torch.Tensor:
        # images: list[PIL.Image] or single PIL.Image
        if not isinstance(images, (list, tuple)):
            images = [images]
        return torch.stack([native_preprocess(img) for img in images])

    def native_txt_pre(texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        token_ids = tokenizer(texts)
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        return {"input_ids": token_ids.to(device)}

    # --------------------------
    # Shared preprocessing
    # --------------------------
    shared_transform = build_image_transform(cfg.shared_preprocess)
    shared_text_proc = TextPreprocessor({"openclip": tokenizer})

    def shared_img_pre(images: Any) -> torch.Tensor:
        if not isinstance(images, (list, tuple)):
            images = [images]
        return torch.stack([shared_transform(img) for img in images])

    def shared_txt_pre(texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        return shared_text_proc(texts, "openclip", device)

    image_encoder = model.visual
    text_encoder = OpenCLIPTextEncoder(model)

    return VLModelWrapper(
        cfg=cfg,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        native_image_preprocess=native_img_pre,
        native_text_preprocess=native_txt_pre,
        shared_image_preprocess=shared_img_pre,
        shared_text_preprocess=shared_txt_pre,
    )


# =====================================================================
#  SigLIP-2 wrapper (HuggingFace-style)
# =====================================================================

def build_siglip2_wrapper(
    cfg: ModelConfig,
    hf_name: str,
) -> VLModelWrapper:
    """
    SigLIP-2 wrapper via HuggingFace.

    You may need to adapt the internal calls depending on the exact
    model class (e.g., SiglipModel, VisionTextDualEncoderModel, etc.).
    """
    from transformers import AutoProcessor, AutoModel, AutoTokenizer
    # Prefer fast processors/tokenizers for fair runtime comparison
    try:
        processor = AutoProcessor.from_pretrained(hf_name, use_fast=True)
    except TypeError:
        processor = AutoProcessor.from_pretrained(hf_name)

    model = AutoModel.from_pretrained(hf_name).to(cfg.device)

    # --------------------------
    # Native preprocessing
    # --------------------------
    def native_img_pre(images: Any) -> torch.Tensor:
        if not isinstance(images, (list, tuple)):
            images = [images]
        proc = processor(
            images=images,
            return_tensors="pt",
        )
        return proc["pixel_values"]  # [B, C, H, W] on CPU

    def native_txt_pre(texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        proc = processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in proc.items() if isinstance(v, torch.Tensor)}

    # --------------------------
    # Shared preprocessing
    # --------------------------
    shared_transform = build_image_transform(cfg.shared_preprocess)

    def shared_img_pre(images: Any) -> torch.Tensor:
        if not isinstance(images, (list, tuple)):
            images = [images]
        return torch.stack([shared_transform(img) for img in images])

    # Shared tokenizer: use processor.tokenizer if present, else AutoTokenizer
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        try:
            tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        except TypeError:
            tok = AutoTokenizer.from_pretrained(hf_name)
    shared_text_proc = TextPreprocessor({"siglip2": tok}, max_length=77)


    def shared_txt_pre(texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        return shared_text_proc(texts, "siglip2", device)

    class SigLIP2Wrapper(VLModelWrapper):
        def _encode_images_impl(self, imgs: torch.Tensor) -> torch.Tensor:
            # This may need adjustment depending on exact SigLIP-2 HF model
            if hasattr(model, "vision_model"):
                out = model.vision_model(pixel_values=imgs)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    return out.pooler_output
                return out.last_hidden_state[:, 0]
            # Fallback: call full model and use vision part
            out = model(pixel_values=imgs)
            if hasattr(out, "image_embeds"):
                return out.image_embeds
            return out.last_hidden_state[:, 0]

        def _encode_texts_impl(self, toks: Dict[str, torch.Tensor]) -> torch.Tensor:
            if hasattr(model, "text_model"):
                out = model.text_model(**toks)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    return out.pooler_output
                return out.last_hidden_state[:, 0]
            out = model(**toks)
            if hasattr(out, "text_embeds"):
                return out.text_embeds
            return out.last_hidden_state[:, 0]

    return SigLIP2Wrapper(
        cfg=cfg,
        image_encoder=model,
        text_encoder=model,
        native_image_preprocess=native_img_pre,
        native_text_preprocess=native_txt_pre,
        shared_image_preprocess=shared_img_pre,
        shared_text_preprocess=shared_txt_pre,
    )






# =====================================================================
#  VLM2Vec-V2 wrapper (HF-style / repo-style)
# =====================================================================

def build_vlm2vecv2_wrapper(
    cfg: ModelConfig,
    hf_name: str = "VLM2Vec/VLM2Vec-V2.0",
) -> VLModelWrapper:
    """
    VLM2Vec-V2 wrapper using the published HF weights.

    - Uses AutoProcessor for text (native + shared).
    - Uses Qwen2-VL's vision/image processor + get_image_features
      to obtain image embeddings (bypasses full LM forward with
      <image> placeholders).

    NOTE: This is a pragmatic integration for your benchmark, not a
    full reimplementation of their MMEBModel pipeline.
    """
    from transformers import AutoProcessor, AutoModel, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(hf_name, use_fast=True)
    except TypeError:
        processor = AutoProcessor.from_pretrained(hf_name)

    hf_model = AutoModel.from_pretrained(hf_name).to(cfg.device)


    # -------------------------------------------------
    # Native preprocessing (TEXT)
    # -------------------------------------------------
    def native_img_pre(images: Any) -> torch.Tensor:
        # For VLM2Vec we override encode_images, so this should not be used.
        raise RuntimeError(
            "VLM2VecWrapper overrides encode_images; native_img_pre should not be called."
        )

    def native_txt_pre(texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        proc = processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in proc.items() if isinstance(v, torch.Tensor)}

    # -------------------------------------------------
    # Shared preprocessing (TEXT)
    # -------------------------------------------------
    shared_transform = build_image_transform(cfg.shared_preprocess)

    def shared_img_pre(images: Any) -> torch.Tensor:
        # For VLM2Vec we override encode_images, so this should not be used.
        raise RuntimeError(
            "VLM2VecWrapper overrides encode_images; shared_img_pre should not be called."
        )

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        try:
            tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        except TypeError:
            tok = AutoTokenizer.from_pretrained(hf_name)
    shared_text_proc = TextPreprocessor({"vlm2vecv2": tok}, max_length=77)

    def shared_txt_pre(texts: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        return shared_text_proc(texts, "vlm2vecv2", device)

    # -------------------------------------------------
    # Wrapper class – custom encode_images using image_processor + get_image_features
    # -------------------------------------------------
    class VLM2VecWrapper(VLModelWrapper):
        def encode_images(
            self,
            images: Any,
            normalize: bool = True,
            return_stats: bool = False,
        ):
            """
            Use Qwen2-VL's image_processor + get_image_features to obtain
            image embeddings. We avoid calling the multimodal processor
            with text=None to sidestep placeholder / token-mismatch issues.
            """
            self.eval()
            stats = None

            with torch.no_grad():
                img_list = images if isinstance(images, (list, tuple)) else [images]

                # Vision-only processing: use the underlying image processor.
                vision_inputs = processor.image_processor(
                    images=img_list,
                    return_tensors="pt",
                )
                pixel_values = vision_inputs["pixel_values"].to(self.device)
                image_grid_thw = vision_inputs.get("image_grid_thw")
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(self.device)

                # Efficiency measurement
                if return_stats and self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                    start = time.perf_counter()

                # Autocast
                if self._autocast_dtype is not None and self.device.type == "cuda":
                    ctx = torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype)
                else:
                    ctx = torch.cpu.amp.autocast(enabled=False)

                with ctx:
                    # Prefer model.get_image_features; fall back to model.model.get_image_features
                    if hasattr(hf_model, "get_image_features"):
                        feats = hf_model.get_image_features(
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                        )
                    elif hasattr(hf_model, "model") and hasattr(hf_model.model, "get_image_features"):
                        feats = hf_model.model.get_image_features(
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                        )
                    else:
                        raise RuntimeError(
                            "VLM2Vec-V2 underlying model has no get_image_features method."
                        )

                    # Normalize output shape to [B, D]
                    if isinstance(feats, (list, tuple)):
                        # List of [Ni, D] – mean-pool tokens per image
                        img_embeds = torch.stack([f.mean(dim=0) for f in feats], dim=0)
                    elif isinstance(feats, torch.Tensor):
                        if feats.dim() == 3:
                            # [B, T, D] – pool over T
                            img_embeds = feats.mean(dim=1)
                        elif feats.dim() == 2:
                            # [N, D]; if N == B treat as-is, else pool everything
                            if feats.size(0) == pixel_values.size(0):
                                img_embeds = feats
                            else:
                                img_embeds = feats.mean(dim=0, keepdim=True)
                        else:
                            # Fallback: flatten and pool
                            img_embeds = feats.view(pixel_values.size(0), -1).mean(dim=1)
                    else:
                        raise RuntimeError(
                            f"Unexpected type from get_image_features: {type(feats)}"
                        )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

                if return_stats and self.device.type == "cuda":
                    end = time.perf_counter()
                    peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                    logical_batch = len(img_list)  # <-- THIS is what we care about
                    stats = {
                        "latency_s": end - start,
                        "peak_mem_mb": peak_mem,
                        "batch_size": logical_batch,
                    }

            if normalize:
                img_embeds = l2_normalize(img_embeds, dim=-1)

            return img_embeds, stats

        def _encode_images_impl(self, imgs: torch.Tensor) -> torch.Tensor:
            """
            Not used; kept only to satisfy the base class interface.
            """
            raise RuntimeError(
                "VLM2VecWrapper uses its own encode_images implementation; "
                "_encode_images_impl should not be called directly."
            )

        def _encode_texts_impl(self, toks: Dict[str, torch.Tensor]) -> torch.Tensor:
            """
            toks: dict from text_preprocess (input_ids, attention_mask, etc.).
            """
            out = hf_model(**toks)

            if hasattr(out, "text_embeds"):
                return out.text_embeds
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state.mean(dim=1)

            raise RuntimeError("VLM2Vec-V2: could not find text embeddings in model output.")

    # We still pass hf_model as both image_encoder/text_encoder to satisfy base class,
    # but VLM2VecWrapper uses hf_model via its own encode_images/_encode_texts_impl.
    return VLM2VecWrapper(
        cfg=cfg,
        image_encoder=hf_model,
        text_encoder=hf_model,
        native_image_preprocess=native_img_pre,
        native_text_preprocess=native_txt_pre,
        shared_image_preprocess=shared_img_pre,
        shared_text_preprocess=shared_txt_pre,
    )
