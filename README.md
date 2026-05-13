# Clasificador Automático de Larvas mediante CNN

> Proyecto Final de Carrera — Bioingeniería

Sistema de visión artificial para la clasificación automática de larvas en tiempo real, basado en una Red Neuronal Convolucional (CNN). El modelo distingue larvas **aptas** de **no aptas** con un **92.93% de accuracy** y se puede desplegar en dispositivos de bajo costo como la Raspberry Pi mediante TensorFlow Lite.

---

## Tabla de contenidos

- [Motivación](#motivación)
- [Pipeline del sistema](#pipeline-del-sistema)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Dataset](#dataset)
- [Arquitectura del modelo](#arquitectura-del-modelo)
- [Resultados](#resultados)
- [Instalación y uso](#instalación-y-uso)
- [Deployment en Raspberry Pi](#deployment-en-raspberry-pi)
- [Autores](#autores)

---

## Motivación

En procesos biológicos que utilizan larvas de insectos (biorreactores, control de calidad entomológico, investigación biomédica), la selección manual de especímenes es lenta, subjetiva y propensa a errores. Este proyecto automatiza esa clasificación con un sistema de visión por computadora que analiza características morfológicas de las larvas a partir de imágenes digitales, eliminando la variabilidad humana y reduciendo el tiempo de procesamiento.

---

## Pipeline del sistema

```
Imagen adquirida
      │
      ▼
Análisis de espacio de color          ← notebook 01
(RGB → HSV: canales H y S óptimos)
      │
      ▼
Segmentación / enmascarado            ← notebook 03
(separación larva - fondo)
      │
      ▼
Entrenamiento CNN                     ← notebook 04
(clasificación: Apta / No Apta)
      │
      ▼
Modelo exportado (.h5 / .tflite)
      │
      ▼
Clasificación en tiempo real
(PC o Raspberry Pi)
```

---

## Estructura del repositorio

```
clasificador-larvas-cnn/
│
├── notebooks/
│   ├── 01_carga_datos_analisis_canales.ipynb   # Carga del dataset y análisis de canales RGB/HSV/YUV/LAB
│   ├── 02_analisis_fondo.ipynb                  # Análisis estadístico del fondo de imagen
│   ├── 03_segmentacion_cnn.ipynb                # Segmentación de larvas y preparación del dataset
│   └── 04_cnn_entrenamiento.ipynb               # Diseño, entrenamiento y evaluación de la CNN
│
├── models/
│   ├── model_acc_9293_pr_9091.h5                # Modelo entrenado en formato Keras (TF/Keras)
│   └── model_cnn_FINAL.tflite                   # Modelo optimizado para edge devices (TFLite)
│
├── sample_data/
│   ├── aptas/         (15 imágenes de ejemplo — clase positiva)
│   ├── no_aptas/      (15 imágenes de ejemplo — clase negativa)
│   └── labels.csv     # Metadatos del dataset completo (id, filename, label)
│
├── docs/
│   ├── Defensa_PFC_Luna_Simondi_Davila.pptx    # Presentación de defensa del PFC
│   └── Instructivo_Raspberry_Pi.docx            # Guía de deployment en Raspberry Pi
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

El dataset fue construido a partir de imágenes capturadas con un sistema de iluminación controlada diseñado para este proyecto.

| Clase     | Imágenes (originales) | Imágenes (enmascaradas) |
|-----------|-----------------------|--------------------------|
| Apta      | 479                   | 958 (aumentación ×2)     |
| No Apta   | 514                   | 1 027 (aumentación ×2)   |
| **Total** | **993**               | **1 985**                |

- Las imágenes originales están disponibles en `sample_data/` (30 muestras representativas).
- El dataset completo no se incluye en el repositorio por tamaño, pero puede reproducirse con los notebooks.

### Análisis de espacios de color

Se evaluaron cuatro espacios de color para la segmentación del fondo:

| Espacio | Resultado |
|---------|-----------|
| RGB     | Sensible a variaciones de iluminación |
| **HSV** | **Óptimo** — canales H (Hue) y S (Saturation) permiten separar larva del fondo con alta precisión |
| YUV     | Aceptable, menor robustez que HSV |
| LAB     | Buena separación cromática, mayor costo computacional |

---

## Arquitectura del modelo

La red convolucional fue diseñada e implementada con **TensorFlow/Keras** sobre imágenes segmentadas (larvas enmascaradas). El entrenamiento se realizó en **Google Colab**.

Características principales:
- **Entrada:** imágenes RGB redimensionadas
- **Capas convolucionales** con activación ReLU y MaxPooling
- **Dropout** para regularización
- **Capa densa de salida** con activación Sigmoid (clasificación binaria)
- **Optimizador:** Adam
- **Loss:** Binary Crossentropy

El modelo final fue exportado en dos formatos:
- `.h5` — para inferencia con TensorFlow/Keras en PC
- `.tflite` — para inferencia optimizada en Raspberry Pi (TensorFlow Lite)

---

## Resultados

| Métrica   | Valor      |
|-----------|------------|
| Accuracy  | **92.93%** |
| Precision | **90.91%** |

Los modelos entrenados están disponibles directamente en la carpeta `models/` y pueden ser usados sin necesidad de reentrenar.

---

## Instalación y uso

### Requisitos

- Python 3.8+
- Las dependencias están listadas en `requirements.txt`

### Instalación

```bash
git clone https://github.com/agustinluna760/clasificador-larvas-cnn.git
cd clasificador-larvas-cnn
pip install -r requirements.txt
```

### Ejecutar los notebooks

```bash
jupyter notebook
```

Se recomienda seguir los notebooks en orden:

1. `01_carga_datos_analisis_canales.ipynb` — exploración del dataset y análisis de canales
2. `02_analisis_fondo.ipynb` — análisis del fondo de imagen
3. `03_segmentacion_cnn.ipynb` — segmentación y preparación del dataset
4. `04_cnn_entrenamiento.ipynb` — entrenamiento y evaluación del modelo

> Los notebooks también pueden ejecutarse directamente en **Google Colab** subiendo los archivos de dataset correspondientes.

### Inferencia con el modelo entrenado

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
model = tf.keras.models.load_model("models/model_acc_9293_pr_9091.h5")

# Preparar imagen
img = Image.open("sample_data/aptas/larva_20250526_145522.jpg").resize((224, 224))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

# Predicción
pred = model.predict(img_array)
print("Apta" if pred[0][0] > 0.5 else "No Apta")
```

---

## Deployment en Raspberry Pi

El modelo `.tflite` está optimizado para correr en hardware de bajos recursos. Ver la guía completa en [`docs/Instructivo_Raspberry_Pi.docx`](docs/Instructivo_Raspberry_Pi.docx).

```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="models/model_cnn_FINAL.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocesar imagen y correr inferencia
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

---

## Autores

- **Agustín Luna** 
- **Simondi**
- **Dávila**

Proyecto Final de Carrera — Bioingeniería  
[agustinluna760@gmail.com](mailto:agustinluna760@gmail.com)

---

## Licencia

Este proyecto se distribuye con fines académicos. Si usás este trabajo como referencia, por favor citalo apropiadamente.
