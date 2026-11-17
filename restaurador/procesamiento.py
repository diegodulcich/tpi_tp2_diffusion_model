"""Funciones que ejecutan el pipeline completo de reparación y análisis."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageFilter

from restaurador.imagenes import (
    componer_solo_mascara,
    crear_mapa_diferencias,
    normalizar_mascara,
)
from restaurador.modelos import (
    cargar_descriptor,
    cargar_detector,
    cargar_inpainting,
    obtener_dispositivo,
)


@dataclass
class ResultadoPipeline:
    """Agrupa imágenes y análisis para que la interfaz los presente juntos."""

    imagen_original: Image.Image
    mascara: Image.Image
    imagen_reparada: Image.Image
    mapa_diferencias: Image.Image
    porcentaje_cambiado: float
    detecciones_originales: list[dict]
    detecciones_reparadas: list[dict]
    descripcion_original: str
    descripcion_reparada: str


def reparar_imagen(
    imagen: Image.Image,
    mascara: Image.Image,
    prompt: str,
    semilla: int,
    pasos: int,
    fuerza: float,
) -> Image.Image:
    """Genera una propuesta dentro de la máscara y conserva intacto el exterior."""
    import torch

    imagen = imagen.convert("RGB")
    mascara = normalizar_mascara(mascara, imagen.size)
    if mascara.getbbox() is None:
        raise ValueError("Dibujá con el pincel la zona que querés reconstruir.")

    tuberia = cargar_inpainting()
    generador = torch.Generator(device=obtener_dispositivo()).manual_seed(semilla)
    generada = tuberia(
        prompt=prompt,
        image=imagen,
        mask_image=mascara,
        num_inference_steps=pasos,
        strength=fuerza,
        generator=generador,
    ).images[0]
    return componer_solo_mascara(imagen, generada, mascara)


def reparar_imagen_basica(imagen: Image.Image, mascara: Image.Image) -> Image.Image:
    """Suaviza la región marcada con un filtro local, sin cargar modelos."""
    imagen = imagen.convert("RGB")
    mascara = normalizar_mascara(mascara, imagen.size)
    if mascara.getbbox() is None:
        raise ValueError("Dibujá con el pincel la zona que querés procesar.")

    suavizada = imagen.filter(ImageFilter.MedianFilter(size=5))
    return componer_solo_mascara(imagen, suavizada, mascara)


def analizar_imagen(imagen: Image.Image) -> tuple[list[dict], str]:
    """Obtiene detecciones de objetos y una descripción en lenguaje natural."""
    detector = cargar_detector()
    descriptor = cargar_descriptor()

    # Ambos análisis reciben RGB porque sus procesadores esperan imágenes de
    # tres canales; el resultado visual de la app se mantiene en blanco y negro.
    imagen_rgb = imagen.convert("RGB")
    detecciones = detector(imagen_rgb)
    descripcion_generada = descriptor(imagen_rgb, max_new_tokens=40)
    descripcion = descripcion_generada[0]["generated_text"].strip()
    return detecciones, descripcion


def ejecutar_pipeline_basico(
    imagen: Image.Image,
    mascara: Image.Image,
) -> ResultadoPipeline:
    """Procesa una imagen localmente con un filtro y sin inferencia de modelos."""
    imagen = imagen.convert("RGB")
    mascara = normalizar_mascara(mascara, imagen.size)
    reparada = reparar_imagen_basica(imagen, mascara)
    mapa, porcentaje = crear_mapa_diferencias(imagen, reparada)

    return ResultadoPipeline(
        imagen_original=imagen,
        mascara=mascara,
        imagen_reparada=reparada,
        mapa_diferencias=mapa,
        porcentaje_cambiado=porcentaje,
        detecciones_originales=[],
        detecciones_reparadas=[],
        descripcion_original="",
        descripcion_reparada="",
    )


def ejecutar_pipeline(
    imagen: Image.Image,
    mascara: Image.Image,
    prompt: str,
    semilla: int,
    pasos: int,
    fuerza: float,
) -> ResultadoPipeline:
    """Ejecuta secuencialmente difusión, análisis visual y comparación."""
    imagen = imagen.convert("RGB")
    mascara = normalizar_mascara(mascara, imagen.size)
    reparada = reparar_imagen(imagen, mascara, prompt, semilla, pasos, fuerza)
    mapa, porcentaje = crear_mapa_diferencias(imagen, reparada)

    # Se analiza una versión por vez para limitar el pico de memoria.
    detecciones_originales, descripcion_original = analizar_imagen(imagen)
    detecciones_reparadas, descripcion_reparada = analizar_imagen(reparada)

    return ResultadoPipeline(
        imagen_original=imagen,
        mascara=mascara,
        imagen_reparada=reparada,
        mapa_diferencias=mapa,
        porcentaje_cambiado=porcentaje,
        detecciones_originales=detecciones_originales,
        detecciones_reparadas=detecciones_reparadas,
        descripcion_original=descripcion_original,
        descripcion_reparada=descripcion_reparada,
    )
