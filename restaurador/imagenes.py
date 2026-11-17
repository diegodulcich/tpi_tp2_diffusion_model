"""Operaciones de imagen y máscara independientes de los modelos de IA."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

MAX_LADO_PROCESAMIENTO = 512
FACTOR_REDIMENSION = 8


def preparar_imagen(imagen: Image.Image) -> Image.Image:
    """Orienta, convierte a escala de grises y reduce una imagen a un máximo de 512 px.

    El resultado se devuelve en RGB porque los modelos de difusión esperan tres
    canales, aunque los tres representen la misma intensidad de gris.
    """
    imagen = ImageOps.exif_transpose(imagen).convert("L")
    ancho, alto = imagen.size
    escala = min(1.0, MAX_LADO_PROCESAMIENTO / max(ancho, alto))

    # Stable Diffusion espera dimensiones divisibles por 8; el ajuste conserva
    # la proporción con una diferencia máxima pequeña debida al redondeo.
    nuevo_ancho = max(
        FACTOR_REDIMENSION,
        int(ancho * escala) // FACTOR_REDIMENSION * FACTOR_REDIMENSION,
    )
    nuevo_alto = max(
        FACTOR_REDIMENSION,
        int(alto * escala) // FACTOR_REDIMENSION * FACTOR_REDIMENSION,
    )
    if (nuevo_ancho, nuevo_alto) != imagen.size:
        imagen = imagen.resize(
            (nuevo_ancho, nuevo_alto),
            Image.Resampling.LANCZOS,
        )
    return imagen.convert("RGB")


def extraer_mascara_canvas(canvas_rgba: np.ndarray) -> Image.Image:
    """Convierte los trazos rojos del lienzo en una máscara binaria.

    Blanco (255) significa «el modelo puede reconstruir» y negro (0) significa
    «conservar». El color rojo se eligió porque la imagen de origen es en gris.
    """
    rojo = canvas_rgba[:, :, 0].astype(np.int16)
    verde = canvas_rgba[:, :, 1].astype(np.int16)
    azul = canvas_rgba[:, :, 2].astype(np.int16)
    trazos = (rojo > 100) & (rojo > verde + 35) & (rojo > azul + 35)
    return Image.fromarray(trazos.astype(np.uint8) * 255)


def normalizar_mascara(mascara: Image.Image, tamano: tuple[int, int]) -> Image.Image:
    """Alinea la máscara con la imagen y la convierte a blanco/negro sin grises."""
    mascara = mascara.convert("L")
    if mascara.size != tamano:
        mascara = mascara.resize(tamano, Image.Resampling.NEAREST)
    return mascara.point(lambda valor: 255 if valor >= 128 else 0, mode="L")


def superponer_mascara(imagen: Image.Image, mascara: Image.Image) -> Image.Image:
    """Muestra en rojo la región que se enviará al modelo."""
    imagen = imagen.convert("RGB")
    mascara = normalizar_mascara(mascara, imagen.size)
    rojo = Image.new("RGB", imagen.size, (255, 0, 0))
    capa = Image.composite(rojo, imagen, mascara)
    return Image.blend(imagen, capa, 0.45)


def componer_solo_mascara(
    original: Image.Image,
    generada: Image.Image,
    mascara: Image.Image,
) -> Image.Image:
    """Usa píxeles generados solo donde la máscara lo permite.

    Esta composición hace que las zonas externas a la máscara queden idénticas
    a la entrada, aunque la tubería de difusión haya alterado otros píxeles.
    """
    original = original.convert("RGB")
    generada = generada.convert("L").convert("RGB")
    if generada.size != original.size:
        generada = generada.resize(original.size, Image.Resampling.LANCZOS)
    mascara = normalizar_mascara(mascara, original.size)
    return Image.composite(generada, original, mascara)


def crear_mapa_diferencias(
    original: Image.Image,
    resultado: Image.Image,
) -> tuple[Image.Image, float]:
    """Calcula una visualización de diferencias y el porcentaje de píxeles cambiados.

    El porcentaje usa un umbral de intensidad de 10/255 y describe cambios
    visibles, no calidad ni autenticidad de la reconstrucción.
    """
    antes = np.asarray(original.convert("RGB"), dtype=np.int16)
    despues = np.asarray(resultado.convert("RGB"), dtype=np.int16)
    diferencia = np.abs(antes - despues).mean(axis=2).astype(np.uint8)
    porcentaje_cambiado = float(np.mean(diferencia > 10) * 100)

    escala_grises = Image.fromarray(diferencia)
    mapa = ImageOps.colorize(
        escala_grises,
        black="#160b39",
        mid="#d2486a",
        white="#fee08b",
    )
    return mapa, porcentaje_cambiado
