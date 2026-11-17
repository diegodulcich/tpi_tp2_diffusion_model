import numpy as np
from PIL import Image

from restaurador.imagenes import crear_mapa_diferencias


def test_porcentaje_cuenta_diferencias_medias_reales_sobre_el_umbral():
    original = Image.fromarray(np.zeros((1, 2, 3), dtype=np.uint8))
    resultado = Image.fromarray(
        np.array([[[11, 10, 11], [10, 10, 10]]], dtype=np.uint8)
    )

    _, porcentaje = crear_mapa_diferencias(original, resultado)

    assert porcentaje == 50.0
