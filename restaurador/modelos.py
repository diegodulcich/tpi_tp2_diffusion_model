"""Carga bajo demanda de los modelos de difusión y análisis de Hugging Face."""

from __future__ import annotations

import os
from functools import lru_cache

MODELO_INPAINTING = "stable-diffusion-v1-5/stable-diffusion-inpainting"
REVISION_INPAINTING = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
MODELO_DETECCION = "PekingU/rtdetr_r50vd"
REVISION_DETECCION = "df939e661d8c52e80608d1ec566561aabd25a4e7"
MODELO_DESCRIPCION = "Salesforce/blip-image-captioning-base"
REVISION_DESCRIPCION = "82a37760796d32b1411fe092ab5d4e227313294b"


def obtener_dispositivo():
    """Usa CUDA si está disponible y CPU como alternativa."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _limitar_hilos_cpu() -> None:
    """Evita que la inferencia en CPU ocupe todos los hilos por defecto."""
    import torch

    hilos = max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(hilos)


@lru_cache(maxsize=1)
def cargar_inpainting():
    """Carga Stable Diffusion en CUDA/FP16 o CPU/FP32 y conserva la caché."""
    import torch
    from diffusers import AutoPipelineForInpainting

    dispositivo = obtener_dispositivo()
    if dispositivo.type == "cpu":
        _limitar_hilos_cpu()
    tuberia = AutoPipelineForInpainting.from_pretrained(
        MODELO_INPAINTING,
        revision=REVISION_INPAINTING,
        torch_dtype=(
            torch.float16 if dispositivo.type == "cuda" else torch.float32
        ),
    )
    tuberia = tuberia.to(dispositivo)
    tuberia.enable_attention_slicing()
    return tuberia


@lru_cache(maxsize=1)
def cargar_detector():
    """Carga RT-DETR en CUDA cuando está disponible, o en CPU."""
    from transformers import pipeline

    dispositivo = obtener_dispositivo()
    return pipeline(
        "object-detection",
        model=MODELO_DETECCION,
        revision=REVISION_DETECCION,
        device=0 if dispositivo.type == "cuda" else -1,
    )


@lru_cache(maxsize=1)
def cargar_descriptor():
    """Carga BLIP en CUDA cuando está disponible, o en CPU."""
    from transformers import pipeline

    dispositivo = obtener_dispositivo()
    return pipeline(
        "image-to-text",
        model=MODELO_DESCRIPCION,
        revision=REVISION_DESCRIPCION,
        device=0 if dispositivo.type == "cuda" else -1,
    )
