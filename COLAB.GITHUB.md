# Ejecutar el MVP desde GitHub en Google Colab

Guía rápida para probar el MVP desde un repositorio público de GitHub, sin
descargar el proyecto ni preparar un entorno local. Está pensada para ejecutarse
en un notebook nuevo de Google Colab. Para ver la arquitectura y el alcance del
MVP, consultá el [`README.md`](./README.md); el flujo interno está en
[`PIPELINE.md`](./PIPELINE.md).

## Antes de empezar

- El repositorio de GitHub debe ser público y contener en su carpeta raíz
  `app.py`, `pyproject.toml` y la carpeta `restaurador/`.
- Se recomienda una GPU disponible en Colab para que el pipeline de IA termine
  en menos tiempo. La aplicación también puede ejecutarse en CPU, pero la
  generación y el análisis pueden tardar mucho.
- El proyecto requiere Python 3.12. La primera ejecución del modo IA necesita
  Internet para descargar los modelos de Hugging Face.
- El enlace compartido de Gradio será accesible por Internet mientras la celda
  siga ejecutándose. No cargues fotografías sensibles ni compartas el enlace
  públicamente.

## 1. Crear el notebook y elegir el runtime

1. Abrí [Google Colab](https://colab.research.google.com/) y creá un notebook
   nuevo.
2. Elegí **Entorno de ejecución → Cambiar tipo de entorno de ejecución**.
3. Seleccioná una GPU disponible y guardá el cambio.

La GPU no es necesaria para la demostración básica, pero se recomienda para
probar el pipeline de IA.

## 2. Clonar el repositorio

En una celda, reemplazá la URL de ejemplo por la URL HTTPS de tu repositorio
público. Debe ser la URL del repositorio, no la URL de un archivo o de una
carpeta dentro de GitHub.

```python
!git clone --depth 1 https://github.com/USUARIO/REPOSITORIO.git /content/restauracion-mvp
%cd /content/restauracion-mvp
```

Por ejemplo, una URL de repositorio tiene esta forma:
`https://github.com/organizacion/nombre-del-proyecto.git`.
La opción `--depth 1` descarga solo la versión más reciente de la rama
predeterminada. Si Colab indica que la carpeta ya existe, cambiá
`/content/restauracion-mvp` por otro nombre o reiniciá el entorno y volvé a
ejecutar la celda.

## 3. Comprobar el proyecto y la versión de Python

Ejecutá esta celda desde la carpeta clonada:

```python
from pathlib import Path
import sys

archivos_requeridos = [
    Path("app.py"),
    Path("pyproject.toml"),
    Path("restaurador") / "procesamiento.py",
]
faltantes = [str(ruta) for ruta in archivos_requeridos if not ruta.exists()]
if faltantes:
    raise FileNotFoundError(
        "El repositorio no tiene la estructura esperada. Faltan: "
        + ", ".join(faltantes)
    )

print("Python:", sys.version.split()[0])
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        "Este MVP requiere Python 3.12. "
        "Revisá la versión del runtime de Colab antes de continuar."
    )
```

## 4. Instalar las dependencias

Ejecutá:

```python
%pip install "accelerate>=1.1,<2" "diffusers>=0.34,<1" "gradio>=5,<6" "numpy>=1.26,<3" "pillow>=10,<13" "transformers>=4.45,<5"
```

No se instala `torch` en esta celda: Colab normalmente proporciona PyTorch
preparado para su runtime. Así se evita reemplazar accidentalmente la versión
de PyTorch configurada para CUDA.

## 5. Comprobar si Colab detecta la GPU

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print(
        "No se detectó GPU. La demostración local funciona en CPU; "
        "el pipeline completo de IA puede tardar mucho."
    )
```

Si esperabas usar GPU, volvé a la configuración del runtime, seleccioná una
GPU y comprobá que Colab la haya asignado antes de iniciar la app.

## 6. Iniciar la aplicación

Ejecutá esta celda:

```python
import app

app.demo.queue().launch(
    server_name="0.0.0.0",
    share=True,
    inbrowser=False,
    show_error=True,
)
```

La celda quedará ejecutándose. Abrí el enlace `https://....gradio.live` que
aparece en la salida para acceder a la aplicación. No cierres la celda mientras
estés usando el MVP.

## 7. Hacer una primera prueba

1. En la app, cargá una imagen JPG o PNG.
2. Pintá en rojo la región que querés procesar. La vista previa muestra la zona
   que se enviará al procesamiento.
3. Para probar rápidamente sin descargar modelos, dejá seleccionado
   **Demostración local (sin IA)** y pulsá **Probar en local**. Este modo aplica
   un filtro mediano y no genera contenido nuevo.
4. Para probar la generación, elegí **Pipeline completo de IA (GPU/CPU)**.
   Configurá la instrucción y los parámetros y pulsá **Reconstruir y analizar**.
5. La primera vez, esperá a que se descarguen los modelos. Después compará
   **Antes**, **Después** y el **Mapa de diferencias**. Podés descargar el PNG
   resultante.

La región generada por IA es una propuesta plausible, no una recuperación
verificada de lo que había en la fotografía. El porcentaje del mapa de
diferencias describe cambios de píxeles, no calidad o autenticidad.

## Problemas frecuentes

- **El repositorio no se clona:** verificá que la URL sea la del repositorio
  público y termine, si corresponde, en `.git`. Esta guía no configura acceso a
  repositorios privados.
- **Versión de Python distinta de 3.12:** el proyecto declara
  `>=3.12,<3.13`. No continúes con ese runtime; usá uno compatible.
- **No hay GPU:** podés probar el modo local en CPU. La inferencia del pipeline
  completo también tiene una ruta de CPU, pero puede ser muy lenta.
- **No aparece el selector de ejemplos:** si no hay archivos de ejemplo en la
  ruta esperada, cargá una imagen manualmente.
- **La celda de Gradio se detuvo:** el enlace deja de estar disponible al
  finalizar el runtime. Volvé a ejecutar la celda de inicio para generar otro.

## Detener la sesión

Cuando termines, detené la celda de inicio o seleccioná **Entorno de ejecución
→ Desconectar y eliminar el entorno de ejecución**. Esto cierra el enlace
compartido de Gradio.
