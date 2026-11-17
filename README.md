# Restauración fotográfica con difusión

MVP local y educativo para experimentar con la restauración de una zona marcada
de una fotografía. Permite comparar una demostración clásica sin IA con un
pipeline de inpainting por difusión y análisis visual automático.

> **Importante:** la salida del modelo de difusión es una propuesta plausible.
> No recupera necesariamente lo que había en la fotografía y no debe
> interpretarse como evidencia histórica ni como verificación de identidad.

## Qué hace la aplicación

1. El usuario carga una imagen JPG o PNG desde la interfaz Gradio.
2. La aplicación corrige su orientación EXIF, la convierte a escala de grises y
   limita el lado mayor a 512 píxeles. La salida se representa en RGB para que
   sea compatible con los modelos.
3. El usuario dibuja en rojo la región que desea procesar. Esos trazos se
   convierten en una máscara binaria: blanco significa «procesar» y negro,
   «conservar».
4. El usuario elige un modo:
   - **Demostración local (sin IA):** aplica un filtro mediano de 5 × 5 en la
     zona marcada. Puede suavizar defectos pequeños, pero no genera contenido.
   - **Pipeline completo de IA:** Stable Diffusion Inpainting genera una
     propuesta condicionada por la imagen, la máscara y una instrucción de
     texto. Luego RT-DETR y BLIP analizan tanto la imagen original como la
     propuesta reconstruida.
5. Se presentan el antes y el después, un mapa de diferencias, un porcentaje de
   píxeles modificados y, en el modo de IA, detecciones y descripciones.
6. La interfaz detalla la operación ejecutada: el modo, el tamaño de la zona
   marcada, el procedimiento aplicado y las limitaciones de la salida. En el
   modo de IA también informa la semilla, los pasos de difusión, la fuerza y los
   análisis realizados.
7. El resultado se puede descargar como PNG.

En ambos modos se recompone la imagen con la máscara: los píxeles fuera de la
región marcada se conservan exactamente. El mapa de diferencias comunica
cambios entre píxeles, no calidad ni autenticidad. El porcentaje cuenta los
píxeles cuya diferencia absoluta media en RGB supera 10 niveles, sobre la escala
de 0 a 255. En el mapa, los tonos oscuros indican poca o ninguna diferencia y
los tonos claros, diferencias mayores. El porcentaje se calcula sobre todos los
píxeles de la imagen.

## Arquitectura

```text
Usuario
  │
  ▼
app.py (interfaz Gradio, estado y validaciones)
  ├── restaurador/imagenes.py (preparación, máscara y comparación)
  └── restaurador/procesamiento.py
        ├── modo básico: filtro mediano
        └── modo IA:
              ├── restaurador/modelos.py → Stable Diffusion Inpainting
              ├── RT-DETR → detecciones de objetos
              └── BLIP → descripción de imagen
  │
  ▼
ResultadoPipeline → comparación, análisis y descarga
```

## Documentación del proyecto

- [`PIPELINE.md`](./PIPELINE.md): recorrido detallado de la ejecución y las
  llamadas entre módulos.
- [`COLAB.GITHUB.md`](./COLAB.GITHUB.md): pasos para clonar el repositorio y
  ejecutar la app en Google Colab.
- [`colab.ipynb`](./colab.ipynb): alternativa para ejecutar en Colab subiendo
  un ZIP del proyecto.
- [`INFORME_MVP.md`](./INFORME_MVP.md): alcance educativo, stack, modelos,
  decisiones de interacción y limitaciones.

### Responsabilidad de cada archivo

- **`app.py`**: construye la interfaz Gradio y conecta sus eventos con las
  funciones de procesamiento. Prepara imágenes y ejemplos, convierte los trazos
  del editor en una máscara, valida las entradas, actualiza la vista previa y
  presenta los resultados. Calcula una clave a partir de la imagen, la máscara y
  los parámetros para ocultar salidas que ya no correspondan a la solicitud
  actual. Guarda los PNG descargables en un directorio temporal.
- **`restaurador/imagenes.py`**: contiene operaciones de imagen independientes
  de los modelos: orientación EXIF, escala de grises, redimensionamiento,
  extracción y normalización de máscara, superposición visual, composición del
  resultado y mapa de diferencias.
- **`restaurador/modelos.py`**: define los identificadores y revisiones fijadas
  de los modelos de Hugging Face y ofrece cargadores bajo demanda. Usa CUDA si
  PyTorch la detecta y CPU como alternativa. Los modelos quedan en caché durante
  la ejecución para evitar cargarlos de nuevo en cada solicitud.
- **`restaurador/procesamiento.py`**: implementa los dos modos y agrupa sus
  resultados en `ResultadoPipeline`. En el modo de IA, genera la reconstrucción,
  calcula las diferencias y analiza la imagen original y la reconstruida una
  por vez. Los modelos cargados quedan en caché durante la ejecución.
- **`restaurador/__init__.py`**: identifica `restaurador` como paquete Python;
  no agrega lógica de procesamiento.
- **`pyproject.toml`**: declara metadatos, versión de Python, dependencias y
  configuración de pytest.
- **`colab.ipynb`**: prepara el proyecto en Google Colab, instala las
  dependencias necesarias, verifica el runtime y lanza la interfaz Gradio.
- **`images/`**: contiene seis fotografías de muestra. Actualmente no están
  conectadas al selector de ejemplos de la interfaz; se pueden cargar
  manualmente desde la app.

### Flujo del modo de IA

1. Se normaliza la máscara al tamaño de la imagen y se verifica que tenga al
   menos una región marcada.
2. Se carga Stable Diffusion Inpainting y se ejecuta con el prompt, la semilla,
   la cantidad de pasos y la fuerza configurados en la interfaz.
3. Se mezclan los píxeles generados únicamente dentro de la máscara. Así, la
   zona exterior queda idéntica a la entrada aunque el pipeline haya cambiado
   píxeles adicionales.
4. Se genera el mapa de diferencias y el porcentaje de píxeles modificados.
5. RT-DETR detecta objetos y BLIP genera una descripción en cada imagen. La
   interfaz solo muestra detecciones con puntaje igual o superior a 0,50. Las
   descripciones de BLIP se presentan en inglés, tal como las genera el modelo.

Las descripciones y detecciones resumen contenido general; no comprueban los
detalles inventados dentro del área dañada.

### Parámetros del experimento

En la interfaz, el modo de IA expone estos controles:

| Parámetro | Valor inicial | Rango |
| --- | --- | --- |
| Instrucción (`prompt`) | `black and white archival photograph, natural photographic texture` | Hasta 200 caracteres |
| Pasos de difusión | 5 | 5–20, de a 1 |
| Fuerza de reconstrucción | 0,85 | 0,30–1,00, de a 0,05 |
| Semilla | 42 | 0–999.999 |

Más pasos pueden aumentar el tiempo de ejecución; una fuerza mayor permite más
libertad de generación dentro de la máscara. Ninguno de estos parámetros
convierte el resultado en una reconstrucción históricamente comprobada.

## Modelos utilizados

Los identificadores y revisiones exactos están declarados en
`restaurador/modelos.py`:

| Uso | Modelo | Revisión fijada |
| --- | --- | --- |
| Inpainting / reconstrucción | `stable-diffusion-v1-5/stable-diffusion-inpainting` | `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb` |
| Detección de objetos | `PekingU/rtdetr_r50vd` | `df939e661d8c52e80608d1ec566561aabd25a4e7` |
| Descripción de imagen | `Salesforce/blip-image-captioning-base` | `82a37760796d32b1411fe092ab5d4e227313294b` |

El modo básico no carga ni descarga estos modelos. En el modo de IA, los pesos
se descargan de Hugging Face la primera vez que se necesiten y luego se
mantienen en caché durante el proceso. Se necesita conexión a Internet para esa
primera descarga.

## Paquetes y dependencias

La fuente de verdad de las versiones permitidas es `pyproject.toml`. Los límites
son rangos de compatibilidad; no fijan una versión exacta.

| Paquete | Rango declarado | Función |
| --- | --- | --- |
| Python | `>=3.12,<3.13` | Runtime requerido por el proyecto. |
| `accelerate` | `>=1.1,<2` | Soporte para ejecución eficiente de modelos de difusión. |
| `diffusers` | `>=0.34,<1` | Pipeline de Stable Diffusion Inpainting. |
| `numpy` | `>=1.26,<3` | Conversión y operaciones numéricas sobre píxeles y máscaras. |
| `pillow` | `>=10,<13` | Lectura, conversión, redimensionamiento y guardado de imágenes. |
| `gradio` | `>=5,<6` | Interfaz web y servidor de la aplicación. |
| `torch` | `>=2.4,<3` | Tensores, inferencia y detección de CUDA. |
| `transformers` | `>=4.45,<5` | Pipelines de detección y descripción de imágenes. |
| `pytest` | `>=8,<9` | Dependencia de desarrollo para pruebas; no es necesaria para ejecutar la app. |

Los modelos completos requieren más memoria y tardan más que el modo básico.
La aplicación puede usar CPU, pero la inferencia será más lenta. El notebook de
Colab incluido exige una GPU compatible y CUDA disponible.

## Estructura del proyecto

```text
.
├── app.py
├── colab.ipynb
├── pyproject.toml
├── README.md
├── PIPELINE.md
├── COLAB.GITHUB.md
├── INFORME_MVP.md
├── images/
│   ├── foto_00.jpg
│   ├── foto_01.jpg
│   ├── foto_02.jpg
│   ├── foto_03.jpg
│   ├── foto_04.jpg
│   └── puente.jpg
└── restaurador/
    ├── __init__.py
    ├── imagenes.py
    ├── modelos.py
    └── procesamiento.py
```

`__pycache__/` puede aparecer después de ejecutar Python; contiene caché
generada y no es código fuente. No hace falta incluirla en el ZIP de Colab.

## Ejecutar en Google Colab desde un notebook nuevo

Este procedimiento parte de un notebook vacío y del código fuente del proyecto
empaquetado en un ZIP. También se puede abrir y ejecutar `colab.ipynb`, que ya
contiene las celdas equivalentes.

### 1. Preparar el ZIP del proyecto

Desde PowerShell, en la carpeta raíz del proyecto, ejecutar:

```powershell
Compress-Archive -Path app.py,restaurador\*.py,pyproject.toml,README.md -DestinationPath restauracion-colab.zip
```

El ZIP debe contener `app.py`, `pyproject.toml`, `README.md` y la carpeta
`restaurador/` con sus archivos Python. No incluir `.venv/`, `__pycache__/` ni
`.pytest_cache/`. La carpeta `images/` es opcional: la aplicación no la usa para
el selector actual.

### 2. Crear el notebook y seleccionar GPU

1. Abrir [Google Colab](https://colab.research.google.com/) y crear un notebook
   nuevo.
2. Ir a **Entorno de ejecución → Cambiar tipo de entorno de ejecución** y
   seleccionar una GPU.
3. Ejecutar una celda de carga de archivos. Subir exactamente el ZIP creado:

```python
from pathlib import Path
from zipfile import ZipFile
from google.colab import files

subidos = files.upload()
archivos_zip = [nombre for nombre in subidos if nombre.lower().endswith(".zip")]
if len(archivos_zip) != 1:
    raise ValueError("Subí exactamente un archivo ZIP del proyecto.")

carpeta_destino = Path("/content/restauracion_mvp").resolve()
carpeta_destino.mkdir(parents=True, exist_ok=True)

with ZipFile(archivos_zip[0]) as archivo_zip:
    for miembro in archivo_zip.infolist():
        ruta_destino = (carpeta_destino / miembro.filename).resolve()
        if not ruta_destino.is_relative_to(carpeta_destino):
            raise ValueError("El ZIP contiene una ruta de extracción no válida.")
    archivo_zip.extractall(carpeta_destino)

archivos_proyecto = list(carpeta_destino.rglob("pyproject.toml"))
if len(archivos_proyecto) != 1:
    raise FileNotFoundError("El ZIP debe contener un único pyproject.toml.")

PROJECT_DIR = archivos_proyecto[0].parent
%cd $PROJECT_DIR
```

La celda verifica que se suba un único ZIP y que sus rutas no intenten extraer
archivos fuera del directorio del proyecto.

### 3. Instalar dependencias

En una nueva celda, instalar los paquetes de inferencia e interfaz:

```python
%pip install "accelerate>=1.1,<2" "diffusers>=0.34,<1" "gradio>=5,<6" "numpy>=1.26,<3" "pillow>=10,<13" "transformers>=4.45,<5"
```

La celda omite `torch` deliberadamente: Colab normalmente proporciona una
versión de PyTorch preparada para su runtime CUDA. El notebook no ejecuta
`pip install .`; importa `app.py` desde el directorio extraído.

### 4. Verificar Python y GPU

Antes de arrancar la app, comprobar la versión de Python y la disponibilidad de
CUDA:

```python
import sys
import torch

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        f"El proyecto requiere Python 3.12; Colab está usando {sys.version.split()[0]}."
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA no está disponible. Seleccioná una GPU en el tipo de entorno de ejecución de Colab y volvé a ejecutar."
    )

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
```

Si la comprobación falla, revisar el tipo de runtime y la versión de Python
antes de continuar. El notebook incluido detiene la ejecución ante cualquiera
de esas condiciones.

### 5. Arrancar la aplicación

Ejecutar en otra celda:

```python
import app

app.demo.queue().launch(
    server_name="0.0.0.0",
    share=True,
    inbrowser=False,
    show_error=True,
)
```

Abrir el enlace `gradio.live` que aparece en la salida. La celda debe permanecer
ejecutándose mientras se use la aplicación. El servidor de Gradio crea un
enlace compartible y accesible desde Internet mientras el runtime siga activo;
compartirlo solo con personas de confianza y no subir imágenes sensibles.

### 6. Probar los dos modos

1. En la app, cargar una fotografía JPG/PNG.
2. Pintar en rojo el área a procesar; la vista previa muestra la máscara en rojo.
3. Para comprobar el flujo sin descargar modelos, mantener **Demostración local
   (sin IA)** y pulsar **Probar en local**.
4. Para usar los modelos, seleccionar **Pipeline completo de IA (GPU/CPU)**,
   escribir una instrucción, elegir pasos, fuerza y semilla, y pulsar
   **Reconstruir y analizar**.
5. Esperar la descarga inicial de los modelos y la inferencia. Comparar el antes,
   el después y el mapa; descargar el PNG si se desea.

La semilla ayuda a repetir una generación con los mismos parámetros y modelos,
pero no garantiza resultados idénticos entre distintas versiones de software o
hardware.

## Ejecución local opcional

Se requiere Python 3.12 y las dependencias indicadas arriba. Desde la raíz del
proyecto, instalar los paquetes y arrancar la interfaz:

```powershell
python -m pip install "accelerate>=1.1,<2" "diffusers>=0.34,<1" "gradio>=5,<6" "numpy>=1.26,<3" "pillow>=10,<13" "torch>=2.4,<3" "transformers>=4.45,<5"
python app.py
```

`app.py` lanza Gradio en `127.0.0.1`. Se puede usar el modo básico sin GPU ni
descarga de modelos. El pipeline de IA carga modelos y puede ejecutarse en CPU,
aunque con una demora considerable.

## Limitaciones conocidas

- El selector de ejemplos busca los archivos `danado_retrato_1860.jpg` y
  `danado_puente_1888.jpg` en `modelos_difusion/datos/`. Esa carpeta no forma
  parte de la estructura actual; por eso el selector queda vacío. Las imágenes
  de `images/` no se conectan automáticamente a ese selector.
- El modo básico solo suaviza con un filtro; no es una reparación generativa.
- Los modelos pueden inventar detalles y sus análisis no verifican la
  autenticidad de la reconstrucción.
- Colab puede cambiar sus versiones de runtime o disponibilidad de GPU. El
  notebook comprueba Python 3.12 y CUDA y muestra un error explícito si faltan.
- `pyproject.toml` configura pytest para buscar pruebas en `tests/`, pero el
  proyecto actual no incluye esa carpeta ni una suite de pruebas.
