# Informe del MVP: Restauración fotográfica educativa

**Fecha:** 16 de noviembre de 2026  
**Propósito:** describir el MVP y sus límites para uso educativo. Este informe
se basa en el código y la configuración del proyecto; no es una evaluación
experimental ni una certificación histórica de las imágenes.

## 1. Descripción del problema y caso de uso

Las fotografías dañadas o con partes faltantes plantean una dificultad para
estudiar procesamiento digital de imágenes: una herramienta puede suavizar un
defecto, pero una técnica generativa también puede inventar contenido que no
está respaldado por la fotografía original. El MVP permite observar y comparar
ambas operaciones sin presentar la salida generativa como una recuperación
auténtica.

**Caso de uso:** una persona carga una fotografía, marca con el pincel la zona
que quiere intervenir y elige una de dos rutas:

- **Demostración local, sin IA:** aplica un filtro mediano de 5 × 5 y copia el
  resultado solo en la máscara. Suaviza detalles locales, pero no genera
  contenido nuevo.
- **Pipeline completo de IA:** Stable Diffusion Inpainting propone contenido
  para la zona marcada a partir de la imagen, la máscara y una instrucción. La
  aplicación presenta también análisis de objetos y una descripción de la
  imagen original y de la propuesta.

En los dos modos, los píxeles fuera de la máscara se conservan exactamente. La
aplicación compara el antes y el después, muestra un mapa de diferencias y
permite descargar el PNG resultante.

## 2. Persona usuaria

La persona usuaria objetivo es una **persona estudiante de procesamiento
digital de imágenes o una docente** que quiere experimentar con operaciones
clásicas y generativas, observar su efecto y entender sus límites. No se asume
que sea especialista en aprendizaje automático.

Necesidades principales:

- elegir de forma explícita qué región procesar;
- comparar una operación local clásica con una propuesta generativa;
- conocer qué hizo cada modo y con qué parámetros;
- distinguir cambios medibles de afirmaciones sobre autenticidad.

Esta persona es una hipótesis de diseño derivada del objetivo educativo del
proyecto; no representa resultados de entrevistas o pruebas con usuarios.

## 3. Arquitectura del sistema

```text
Persona usuaria
  │
  ▼
Gradio / app.py
  ├─ prepara y valida la imagen y la máscara
  ├─ actualiza la vista previa y el estado de la solicitud
  └─ ejecuta una ruta:
       ├─ Demostración local → filtro mediano 5 × 5
       └─ Pipeline IA → Stable Diffusion Inpainting
                         └─ RT-DETR + BLIP sobre original y propuesta
  │
  ▼
restaurador/imagenes.py
  ├─ normalización de imagen y máscara
  ├─ composición limitada a la máscara
  └─ mapa y porcentaje de diferencias
  │
  ▼
ResultadoPipeline → panel de comparación, explicación y descarga PNG
```

- [`app.py`](./app.py) implementa la interfaz Gradio, los controles, las
  validaciones, la invalidación de resultados obsoletos y la presentación.
- [`restaurador/imagenes.py`](./restaurador/imagenes.py) orienta la imagen con
  EXIF, la convierte a escala de grises, la limita a un máximo de 512 px por
  lado, crea y normaliza la máscara y calcula la comparación.
- [`restaurador/procesamiento.py`](./restaurador/procesamiento.py) implementa
  los dos modos y agrupa las salidas en `ResultadoPipeline`.
- [`restaurador/modelos.py`](./restaurador/modelos.py) carga bajo demanda y
  conserva en caché los modelos. Usa CUDA cuando PyTorch la detecta; si no,
  usa CPU.

El procesamiento se ejecuta en el entorno donde se inicia la app (equipo local
o runtime de Colab). En la primera ejecución del modo IA se descargan los pesos
desde Hugging Face; las inferencias posteriores se realizan con esos modelos
cargados en el proceso.

## 4. Stack tecnológico, modelos, licencias y hardware

### Stack declarado

Los siguientes son **rangos de compatibilidad** declarados en
[`pyproject.toml`](./pyproject.toml), no versiones exactas instaladas. El
proyecto no incluye `uv.lock`, por lo que no hay un registro reproducible de
las versiones resueltas.

| Componente | Rango declarado | Uso |
| --- | --- | --- |
| Python | `>=3.12,<3.13` | Runtime del proyecto. |
| Gradio | `>=5,<6` | Interfaz de usuario. |
| PyTorch | `>=2.4,<3` | Inferencia y detección de CUDA. |
| Diffusers | `>=0.34,<1` | Pipeline de inpainting. |
| Transformers | `>=4.45,<5` | Pipelines RT-DETR y BLIP. |
| Pillow | `>=10,<13` | Lectura, preparación y guardado de imágenes. |
| NumPy | `>=1.26,<3` | Operaciones sobre píxeles y máscaras. |
| Accelerate | `>=1.1,<2` | Soporte para inferencia con Diffusers. |

### Modelos

El código fija cada modelo a una revisión de Hugging Face. Los hashes son
identificadores de *commit* del repositorio del modelo, no números de versión
semántica.

| Función | Modelo y revisión fijada | Licencia declarada |
| --- | --- | --- |
| Propuesta de inpainting | [`stable-diffusion-v1-5/stable-diffusion-inpainting`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/tree/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb) — `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb` | CreativeML OpenRAIL-M. La ficha incluye restricciones de uso y advierte que no es un modelo factual para representar personas o eventos. |
| Detección de objetos | [`PekingU/rtdetr_r50vd`](https://huggingface.co/PekingU/rtdetr_r50vd/tree/df939e661d8c52e80608d1ec566561aabd25a4e7) — `df939e661d8c52e80608d1ec566561aabd25a4e7` | Apache-2.0. |
| Descripción de imagen | [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base/tree/82a37760796d32b1411fe092ab5d4e227313294b) — `82a37760796d32b1411fe092ab5d4e227313294b` | BSD-3-Clause en los metadatos de Hugging Face. La ficha también recomienda evaluar los riesgos y describe el lanzamiento en el contexto de investigación. |

Las licencias y las condiciones de uso corresponden a los repositorios de los
modelos, no conceden derechos sobre las fotografías que suba el usuario ni
resuelven por sí solas todas las obligaciones de redistribución o uso
comercial. Revisar las fichas y los textos de licencia antes de esos usos.

### Hardware probado y pendiente de confirmar

**No hay un registro verificable del hardware con el que se haya probado el
MVP**: el repositorio no incluye resultados de ejecución con modelo de GPU,
CPU, RAM o versiones efectivamente instaladas. El notebook de Colab exige una
GPU y comprueba que CUDA esté disponible, pero eso documenta un requisito del
flujo, no que se haya completado una prueba en un dispositivo determinado.

La aplicación tiene una ruta de CPU; se espera que sea más lenta. Para completar
este dato se debe ejecutar el proyecto y registrar la salida de:

```python
import platform
import torch

print("Plataforma:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 5. Instalación y ejecución con `uv` (Windows / PowerShell)

Desde PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

Abrir una nueva terminal y, desde la raíz del proyecto:

```powershell
Set-Location "C:\Users\DDulcich\Documents\Personal\IFTS24_PROGRAMACION\PDI_TP2_COLAB"
uv python install 3.12
uv sync --no-install-project --no-dev
uv run --no-sync python app.py
```

`uv sync` instala las dependencias declaradas y crea el entorno del proyecto;
`--no-install-project` evita intentar instalar como paquete un proyecto que se
ejecuta directamente desde sus archivos fuente. `--no-dev` omite las
dependencias de desarrollo. La app local abre Gradio en `127.0.0.1`. La primera
ejecución del modo IA necesita Internet para descargar los modelos; el modo
básico no los descarga.

Comprobación de PyTorch y CUDA:

```powershell
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

La disponibilidad de CUDA depende de la variante de PyTorch instalada, el
sistema operativo, el controlador y el hardware. La instalación estándar
resuelta desde PyPI puede instalar una compilación CPU-only en Windows. Para
usar una GPU local, elegir una compilación compatible en el selector oficial
de PyTorch y configurar el índice de PyTorch correspondiente en `uv`; la guía
oficial de [uv con PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
describe cómo fijar `torch` a un índice. El notebook de Colab tiene otro flujo:
usa el PyTorch que ofrece el runtime y exige CUDA.

## 6. Decisiones de diseño desde HAI

HAI (*Human-AI Interaction*) se aplica aquí como un principio de interacción
comprensible y controlable, no como una afirmación de que el sistema sea
autónomo:

- **Control humano de la región:** la persona dibuja la máscara; el sistema no
  decide por su cuenta qué parte de la fotografía debe modificarse.
- **Alternativa sin IA:** el modo básico permite explorar el flujo sin descargar
  modelos ni confundir el filtro clásico con una generación.
- **Parámetros visibles:** el modo IA expone instrucción, pasos, fuerza y
  semilla. La interfaz explica que más pasos pueden tardar más y que una fuerza
  mayor da más libertad de generación, pero ningún parámetro garantiza verdad.
- **Alcance de la acción legible:** la salida explica el modo utilizado, la
  proporción marcada, la operación y los parámetros relevantes; la comparación
  y el mapa ayudan a inspeccionar el cambio.
- **Incertidumbre explícita:** la reconstrucción se presenta como propuesta
  plausible, no como restauración verificada. Las detecciones y descripciones
  se etiquetan como análisis generales, no como verificación de identidad o
  historia.
- **Consistencia de la interfaz:** si cambia la imagen, la máscara o los
  parámetros después de procesar, se oculta el resultado anterior para no
  mostrarlo como si correspondiera a la solicitud nueva.

## 7. Qué es real y qué es inventado

| Elemento | Qué significa |
| --- | --- |
| Imagen fuera de la máscara | Los píxeles se copian de la entrada y se conservan exactamente. |
| Modo básico | Se aplica un filtro mediano 5 × 5 en la zona marcada; suaviza, pero no recupera información perdida. |
| Zona generada por difusión | Es una propuesta del modelo condicionada por imagen, máscara e instrucción. Puede ser plausible, pero el código no dispone de la foto original dañada ni de una fuente de verdad para comprobarla. |
| Detecciones RT-DETR y texto BLIP | Son inferencias automáticas con posibles errores; no demuestran que un objeto o detalle haya estado en la fotografía. |
| Mapa y porcentaje de diferencias | Describen cambios de píxeles. El porcentaje cuenta píxeles cuya diferencia media supera `10/255`; no mide calidad, fidelidad o autenticidad. |

En consecuencia, el MVP puede enseñar cómo se procesa una máscara y cómo un
modelo generativo completa una región, pero no reconstruir con certeza el
contenido histórico perdido.

## 8. Uso de agentes de IA

El producto **no incluye un agente conversacional ni delega decisiones a un
agente autónomo**. Sus modelos cumplen tareas acotadas: generación de imagen,
detección de objetos y descripción de imagen.

En el trabajo de desarrollo documentado en esta conversación:

- **Decisión de la persona responsable del proyecto:** mantener el propósito
  educativo y hacer que la explicación de la operación aparezca en el módulo
  que usa la persona.
- **Propuesta del agente de programación:** que la explicación del resultado
  detalle el modo ejecutado, el tamaño de la región marcada, los pasos del
  procedimiento, los parámetros de IA y la medida de diferencias; también
  diferenciar el mensaje de finalización del modo básico del mensaje del modo
  generativo.
- **Resultado:** se incorporó esa explicación dinámica en la interfaz y se
  aclaró que el resultado generativo es una propuesta. La selección de la zona,
  el modo y los parámetros continúa bajo control de la persona usuaria.

## 9. Limitaciones conocidas

- La imagen se convierte a escala de grises y se reduce a un máximo de 512 px
  por lado; se pierde información de color y detalle fino.
- La generación puede inventar contenido, tener sesgos o no representar bien
  personas, texto y escenas complejas.
- La fuerza de generación, la semilla y los pasos no garantizan resultados
  idénticos entre hardware o versiones diferentes.
- La inferencia en CPU puede ser muy lenta; el primer uso del pipeline IA
  requiere conexión para descargar pesos.
- BLIP produce descripciones en inglés. RT-DETR puede omitir objetos o
  detectar falsos positivos; la interfaz oculta puntajes menores que `0,50`.
- El mapa de diferencias no es una evaluación perceptual ni histórica.
- El selector de ejemplos busca archivos en `modelos_difusion/datos/`, carpeta
  que no está incluida en la estructura actual; las imágenes de `images/` deben
  cargarse manualmente.
- El repositorio declara `pytest`, pero no incluye una carpeta de pruebas ni
  resultados de una suite automatizada.
- El notebook de Colab crea un enlace público de Gradio mientras el runtime
  está activo. No se deben cargar fotografías sensibles ni compartir ese enlace
  sin controlar quién puede acceder.
- No se ha registrado una prueba en hardware específico ni una evaluación con
  usuarios; ambos son pendientes antes de afirmar rendimiento o usabilidad.

## Fuentes

- [`pyproject.toml`](./pyproject.toml), [`README.md`](./README.md),
  [`colab.ipynb`](./colab.ipynb) y el código de `restaurador/`.
- Fichas y metadatos oficiales de los modelos en Hugging Face, enlazados en la
  tabla de modelos.
- [Instalación de uv](https://docs.astral.sh/uv/getting-started/installation/),
  [gestión de proyectos](https://docs.astral.sh/uv/guides/projects/) y
  [configuración de PyTorch con uv](https://docs.astral.sh/uv/guides/integration/pytorch/).
- [Selector oficial de instalación de PyTorch](https://pytorch.org/get-started/locally/).
