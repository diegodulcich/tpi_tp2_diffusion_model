"""Interfaz Gradio del MVP educativo de restauración con difusión."""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError

from restaurador.imagenes import (
    extraer_mascara_canvas,
    normalizar_mascara,
    preparar_imagen,
    superponer_mascara,
)
from restaurador.procesamiento import (
    ResultadoPipeline,
    ejecutar_pipeline,
    ejecutar_pipeline_basico,
)

CARPETA_EJEMPLOS = Path(__file__).parent / "modelos_difusion" / "datos"
EJEMPLOS = {
    "Retrato dañado (1860)": CARPETA_EJEMPLOS / "danado_retrato_1860.jpg",
    "Puente dañado (1888)": CARPETA_EJEMPLOS / "danado_puente_1888.jpg",
}
EJEMPLOS_DISPONIBLES = {
    nombre: ruta for nombre, ruta in EJEMPLOS.items() if ruta.is_file()
}
CARPETA_DESCARGAS = tempfile.TemporaryDirectory(prefix="restauracion-gradio-")

MODO_BASICO = "Demostración local (sin IA)"
MODO_IA = "Pipeline completo de IA (GPU/CPU)"


def preparar_editor(datos_editor: dict[str, Any] | None) -> dict[str, Any] | None:
    """Prepara la fotografía y la coloca como fondo limpio del editor."""
    if not isinstance(datos_editor, dict):
        return None

    fondo = datos_editor.get("background")
    if not isinstance(fondo, Image.Image):
        return None

    try:
        imagen = preparar_imagen(fondo)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise gr.Error(f"No se pudo preparar la imagen: {error}") from error

    return {"background": imagen, "layers": [], "composite": imagen.copy()}


def cargar_ejemplo(nombre: str | None) -> dict[str, Any] | None:
    """Carga un ejemplo disponible en el formato que espera ImageEditor."""
    if nombre is None:
        return None

    ruta = EJEMPLOS_DISPONIBLES.get(nombre)
    if ruta is None:
        raise gr.Error("El ejemplo seleccionado no está disponible.")

    try:
        with Image.open(ruta) as archivo:
            imagen = preparar_imagen(archivo)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise gr.Error(f"No se pudo abrir la imagen seleccionada: {error}") from error

    return {"background": imagen, "layers": [], "composite": imagen.copy()}


def obtener_imagen_y_mascara(
    datos_editor: dict[str, Any] | None,
) -> tuple[Image.Image | None, Image.Image | None]:
    """Extrae el fondo preparado y convierte los trazos rojos en máscara."""
    if not isinstance(datos_editor, dict):
        return None, None

    fondo = datos_editor.get("background")
    if not isinstance(fondo, Image.Image):
        return None, None

    imagen = preparar_imagen(fondo)
    compuesto = datos_editor.get("composite")
    if not isinstance(compuesto, Image.Image):
        mascara = Image.new("L", imagen.size, 0)
    else:
        trazos_rgba = np.asarray(compuesto.convert("RGBA"))
        mascara = extraer_mascara_canvas(trazos_rgba)
        mascara = normalizar_mascara(mascara, imagen.size)

    return imagen, mascara


def actualizar_vista_mascara(
    datos_editor: dict[str, Any] | None,
) -> Image.Image | None:
    """Devuelve una vista previa roja de la región que se procesará."""
    imagen, mascara = obtener_imagen_y_mascara(datos_editor)
    if imagen is None or mascara is None:
        return None
    return superponer_mascara(imagen, mascara)


def clave_solicitud(
    imagen: Image.Image,
    mascara: Image.Image,
    modo: str,
    prompt: str,
    semilla: int,
    pasos: int,
    fuerza: float,
) -> str:
    """Identifica los parámetros exactos para no mostrar resultados desactualizados."""
    resumen = (
        f"{modo}|{prompt}|{semilla}|{pasos}|{fuerza}|"
        f"{imagen.size}|{hashlib.sha256(imagen.tobytes()).hexdigest()}|"
        f"{hashlib.sha256(mascara.tobytes()).hexdigest()}"
    )
    return hashlib.sha256(resumen.encode("utf-8")).hexdigest()


def mostrar_detecciones(titulo: str, detecciones: list[dict[str, Any]]) -> str:
    """Presenta etiquetas y puntajes sin tratarlos como verificación de identidad."""
    confiables = [item for item in detecciones if item["score"] >= 0.5]
    if not confiables:
        return f"**{titulo}**\n\nNo se detectaron objetos con puntaje igual o mayor a 0,50."
    resultados = "\n".join(
        f"- {item['label']}: {item['score']:.2f}" for item in confiables
    )
    return f"**{titulo}**\n\n{resultados}"


def invalidar_resultado_desactualizado(
    datos_editor: dict[str, Any] | None,
    modo: str,
    prompt: str,
    pasos: int,
    fuerza: float,
    semilla: int,
    resultado_guardado: ResultadoPipeline | None,
    solicitud_guardada: str,
) -> tuple[Any, Any, Any, Any]:
    """Oculta un resultado cuando cambia la imagen o alguno de sus parámetros."""
    if not isinstance(resultado_guardado, ResultadoPipeline):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()

    imagen, mascara = obtener_imagen_y_mascara(datos_editor)
    if imagen is None or mascara is None:
        solicitud_actual = None
    else:
        solicitud_actual = clave_solicitud(
            imagen, mascara, modo, prompt, int(semilla), int(pasos), float(fuerza)
        )

    if solicitud_actual == solicitud_guardada:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()

    return (
        None,
        "",
        "Cambiaste la imagen o los parámetros. Volvé a ejecutar para actualizar el resultado.",
        gr.update(visible=False),
    )


def procesar(
    datos_editor: dict[str, Any] | None,
    modo: str,
    prompt: str,
    pasos: int,
    fuerza: float,
    semilla: int,
) -> tuple[Any, ...]:
    """Ejecuta el modo elegido y actualiza la comparación y el archivo descargable."""
    imagen, mascara = obtener_imagen_y_mascara(datos_editor)
    if imagen is None or mascara is None:
        raise gr.Error("Cargá una fotografía antes de iniciar el procesamiento.")

    if mascara.getbbox() is None:
        raise gr.Error("Dibujá con el pincel la zona que querés procesar.")

    modo_basico = modo == MODO_BASICO
    solicitud = clave_solicitud(
        imagen, mascara, modo, prompt, int(semilla), int(pasos), float(fuerza)
    )

    try:
        if modo_basico:
            resultado = ejecutar_pipeline_basico(imagen=imagen, mascara=mascara)
        else:
            resultado = ejecutar_pipeline(
                imagen=imagen,
                mascara=mascara,
                prompt=prompt,
                semilla=int(semilla),
                pasos=int(pasos),
                fuerza=float(fuerza),
            )
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        raise gr.Error(f"No se pudo completar el procesamiento: {error}") from error

    nombre_archivo = (
        "resultado_local.png" if modo_basico else "reconstruccion_ia.png"
    )
    ruta_descarga = Path(CARPETA_DESCARGAS.name) / f"{uuid.uuid4().hex}-{nombre_archivo}"
    resultado.imagen_reparada.save(ruta_descarga, format="PNG")

    pixeles_marcados = int(np.count_nonzero(np.asarray(mascara, dtype=np.uint8)))
    pixeles_totales = imagen.width * imagen.height
    porcentaje_mascara = pixeles_marcados / pixeles_totales * 100
    porcentaje_mascara_texto = f"{porcentaje_mascara:.1f}".replace(".", ",")
    porcentaje_cambiado_texto = f"{resultado.porcentaje_cambiado:.1f}".replace(
        ".", ","
    )
    fuerza_texto = f"{float(fuerza):.2f}".replace(".", ",")

    if modo_basico:
        descripcion = (
            "La demostración local no ejecuta análisis de objetos ni genera "
            "descripciones. Seleccioná el pipeline completo para esas funciones."
        )
        explicacion = (
            f"**Acción realizada:** demostración local sin IA sobre una imagen de "
            f"{imagen.width} × {imagen.height} píxeles.\n\n"
            f"1. Tomó como zona de trabajo los {pixeles_marcados} píxeles blancos "
            f"de la máscara ({porcentaje_mascara_texto}% de la imagen); los píxeles "
            "negros quedaron fuera de la intervención.\n"
            "2. Aplicó un filtro mediano de 5 × 5, que sustituye cada píxel por "
            "un valor representativo de sus vecinos. El filtro suaviza detalles "
            "locales; no genera ni recupera contenido ausente.\n"
            "3. Combinó el filtrado solo en la zona marcada y conservó exactamente "
            "los píxeles originales fuera de ella.\n\n"
            f"**Cambio observado:** {porcentaje_cambiado_texto}% de los "
            "píxeles superó 10 niveles (10/255) en la diferencia absoluta media "
            "de sus canales RGB. Es una medida de cambio, no de calidad."
        )
        descripcion_original = descripcion_reparada = ""
        detecciones_originales = detecciones_reparadas = ""
    else:
        descripcion = (
            "BLIP genera descripciones en inglés. Se muestran tal cual para no "
            "presentar una traducción automática como si la hubiera producido el modelo."
        )
        explicacion = (
            f"**Acción realizada:** propuesta de reconstrucción con el pipeline "
            f"completo de IA sobre una imagen de {imagen.width} × {imagen.height} "
            "píxeles.\n\n"
            f"1. Usó los {pixeles_marcados} píxeles blancos de la máscara "
            f"({porcentaje_mascara_texto}% de la imagen) como zona editable; los "
            "píxeles negros se reservaron para conservar la foto original.\n"
            f"2. Solicitó a Stable Diffusion Inpainting una propuesta guiada por "
            f"la instrucción configurada, con semilla **{int(semilla)}**, "
            f"**{int(pasos)} pasos** y fuerza **{fuerza_texto}**. La generación "
            "usa la imagen y la máscara como contexto, pero puede inventar detalles.\n"
            "3. Copió la propuesta generada únicamente dentro de la máscara y "
            "conservó exactamente los píxeles originales fuera de ella.\n"
            "4. Analizó la foto original y la propuesta con RT-DETR (objetos) y "
            "BLIP (descripción). Se muestran detecciones con puntaje desde 0,50; "
            "estos análisis resumen contenido general y no verifican lo reconstruido.\n\n"
            f"**Cambio observado:** {porcentaje_cambiado_texto}% de los "
            "píxeles superó 10 niveles (10/255) en la diferencia absoluta media "
            "de sus canales RGB. Es una medida de cambio, no de calidad ni autenticidad."
        )
        descripcion_original = resultado.descripcion_original or (
            "El modelo no generó una descripción."
        )
        descripcion_reparada = resultado.descripcion_reparada or (
            "El modelo no generó una descripción."
        )
        detecciones_originales = mostrar_detecciones(
            "Objetos detectados", resultado.detecciones_originales
        )
        detecciones_reparadas = mostrar_detecciones(
            "Objetos detectados", resultado.detecciones_reparadas
        )

    estado = (
        "Demostración local terminada. Se aplicó suavizado; no se generó una "
        "reconstrucción con IA."
        if modo_basico
        else "Proceso de IA terminado. La reconstrucción es una propuesta "
        "plausible, no una recuperación verificada."
    )
    caption_mapa = (
        "**Cómo leer el mapa:** los tonos oscuros (violeta casi negro) indican "
        "poca o ninguna diferencia; los tonos claros indican cambios mayores.\n\n"
        "**Umbral 10/255:** se calcula la diferencia absoluta media de los "
        "canales rojo, verde y azul de cada píxel, en una escala de 0 a 255. "
        "Un píxel cuenta como cambiado si esa media supera 10 niveles.\n\n"
        f"**Cambio notable:** {porcentaje_cambiado_texto}% de los píxeles "
        "superó el umbral. El porcentaje se calcula sobre toda la imagen y "
        "describe cambios de intensidad; no mide calidad ni autenticidad."
    )
    return (
        resultado,
        solicitud,
        estado,
        gr.update(visible=True),
        resultado.imagen_original,
        resultado.imagen_reparada,
        resultado.mapa_diferencias,
        caption_mapa,
        descripcion,
        descripcion_original,
        detecciones_originales,
        descripcion_reparada,
        detecciones_reparadas,
        explicacion,
        str(ruta_descarga),
    )


def actualizar_modo(modo: str) -> tuple[Any, Any, Any]:
    """Actualiza los controles y la información del modo seleccionado."""
    modo_basico = modo == MODO_BASICO
    texto_boton = "Probar en local" if modo_basico else "Reconstruir y analizar"
    return (
        gr.update(visible=not modo_basico),
        gr.update(visible=modo_basico),
        gr.update(value=texto_boton),
    )


with gr.Blocks(title="Restauración educativa con IA") as demo:
    gr.Markdown(
        "# Restauración fotográfica con difusión\n"
        "MVP local y educativo para experimentar con una zona marcada."
    )
    gr.Markdown(
        "> En el modo de IA, lo reconstruido es una invención plausible del modelo, "
        "no una recuperación garantizada de la foto ni evidencia histórica."
    )
    gr.Markdown(
        "No hace falta una GPU para abrir la app. La demostración local funciona "
        "sin modelos; el pipeline completo usa una GPU compatible si está disponible "
        "y, si no, se ejecuta en CPU con mayor demora."
    )

    with gr.Row():
        if EJEMPLOS_DISPONIBLES:
            ejemplo = gr.Dropdown(
                choices=list(EJEMPLOS_DISPONIBLES),
                value=None,
                label="O elegí una imagen de ejemplo",
            )
        else:
            ejemplo = None
            gr.Markdown("No hay ejemplos locales disponibles. Subí una fotografía.")

    gr.Markdown(
        "## 1. Elegí la zona a procesar\n"
        "Subí una fotografía JPG/PNG o elegí un ejemplo. Pintá en rojo sobre el "
        "área dañada: blanco indica la zona que se procesará y negro la que se conservará."
    )
    editor = gr.ImageEditor(
        label="Fotografía y máscara",
        type="pil",
        image_mode="RGB",
        sources=["upload"],
        brush=gr.Brush(colors=["#ff0000"], default_size=10),
        eraser=gr.Eraser(default_size=5),
        transforms=[],
        height=520,
    )
    mascara_preview = gr.Image(
        label="Vista previa de la máscara: la zona roja se procesará",
        type="pil",
    )

    gr.Markdown("## 2. Elegí el modo de procesamiento")
    modo = gr.Radio(
        choices=[MODO_BASICO, MODO_IA],
        value=MODO_BASICO,
        label="Modo",
        info=(
            "La demostración usa un filtro local y no descarga modelos. El pipeline "
            "completo usa CUDA si está disponible; si no, ejecuta los modelos en CPU."
        ),
    )
    informacion_demo = gr.Markdown(
        "La demostración aplica un filtro mediano dentro de la máscara. Sirve para "
        "probar el flujo local, pero no usa IA ni reconstruye contenido como el modelo "
        "de difusión.",
        visible=True,
    )

    with gr.Group(visible=False) as controles_ia:
        gr.Markdown("## 3. Configurá el experimento")
        prompt = gr.Textbox(
            label="Instrucción para la región reconstruida",
            value="black and white archival photograph, natural photographic texture",
            max_length=200,
            info=(
                "La instrucción guía la generación; no aporta evidencia sobre cómo "
                "era la foto."
            ),
        )
        with gr.Row():
            pasos = gr.Slider(
                minimum=5,
                maximum=20,
                step=1,
                value=5,
                label="Pasos de difusión",
                info=(
                    "Más pasos suelen tardar más y pueden refinar la propuesta, "
                    "pero no hacen que sea más verdadera. En CPU, empezá con el mínimo."
                ),
            )
            fuerza = gr.Slider(
                minimum=0.3,
                maximum=1.0,
                step=0.05,
                value=0.85,
                label="Fuerza de reconstrucción",
                info="Un valor alto permite mayor libertad para generar dentro de la máscara.",
            )
            semilla = gr.Number(
                minimum=0,
                maximum=999_999,
                precision=0,
                value=42,
                label="Semilla reproducible",
            )

    boton = gr.Button("Probar en local", variant="primary")
    estado = gr.Markdown()

    with gr.Column(visible=False) as panel_resultados:
        gr.Markdown("## 4. Comparación")
        with gr.Row():
            imagen_original = gr.Image(label="Antes", type="pil")
            imagen_reparada = gr.Image(label="Después", type="pil")
        mapa_diferencias = gr.Image(label="Mapa de diferencias", type="pil")
        caption_mapa = gr.Markdown()

        gr.Markdown("## 5. Análisis visual automático")
        descripcion_general = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Descripción de la imagen original**")
                descripcion_original = gr.Markdown()
                detecciones_originales = gr.Markdown()
            with gr.Column():
                gr.Markdown("**Descripción de la propuesta reconstruida**")
                descripcion_reparada = gr.Markdown()
                detecciones_reparadas = gr.Markdown()

        gr.Markdown("## Operación realizada y cómo funciona")
        explicacion = gr.Markdown()
        descarga = gr.DownloadButton("Descargar resultado (PNG)")

    resultado_estado = gr.State(None)
    solicitud_estado = gr.State("")

    if ejemplo is not None:
        ejemplo.change(fn=cargar_ejemplo, inputs=ejemplo, outputs=editor)

    editor.upload(fn=preparar_editor, inputs=editor, outputs=editor)
    editor.change(
        fn=actualizar_vista_mascara,
        inputs=editor,
        outputs=mascara_preview,
    )
    modo.change(
        fn=actualizar_modo,
        inputs=modo,
        outputs=[controles_ia, informacion_demo, boton],
    )

    entradas_solicitud = [editor, modo, prompt, pasos, fuerza, semilla]
    for componente in entradas_solicitud:
        componente.change(
            fn=invalidar_resultado_desactualizado,
            inputs=[
                *entradas_solicitud,
                resultado_estado,
                solicitud_estado,
            ],
            outputs=[
                resultado_estado,
                solicitud_estado,
                estado,
                panel_resultados,
            ],
        )

    boton.click(
        fn=procesar,
        inputs=entradas_solicitud,
        outputs=[
            resultado_estado,
            solicitud_estado,
            estado,
            panel_resultados,
            imagen_original,
            imagen_reparada,
            mapa_diferencias,
            caption_mapa,
            descripcion_general,
            descripcion_original,
            detecciones_originales,
            descripcion_reparada,
            detecciones_reparadas,
            explicacion,
            descarga,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        inbrowser=True,
        show_error=True,
    )
