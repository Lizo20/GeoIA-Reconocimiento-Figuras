# GeoIA — Reconocimiento de figuras geométricas

Proyecto académico de **Sistemas Expertos, Semana 8 (UNIMINUTO)**. Prototipo educativo de red neuronal convolucional (CNN) que reconoce **círculos, cuadrados, triángulos y rectángulos** en imágenes sintéticas en escala de grises de 64 × 64 píxeles.

## Tecnologías
- Python 3
- PyTorch para la red neuronal convolucional
- Pillow y NumPy para generar y procesar las imágenes
- scikit-learn y Matplotlib para evaluación y gráficos

## Arquitectura
Dos capas convolucionales con activación ReLU y MaxPool2d; una capa densa intermedia, Dropout(0.2) y cuatro puntuaciones de salida. Optimización Adam, pérdida CrossEntropyLoss y regularización weight_decay.

## Ejecutar localmente
```bash
pip install -r requirements.txt
python geoia.py --train --epocas 12 --por-clase 180
python geoia.py --predict resultados/imagen_ejemplo_circulo.png
```

La primera ejecución crea la carpeta `resultados/` con el modelo entrenado (`modelo_geoia.pt`), métricas y gráficos. El comando de predicción **requiere primero entrenar**, salvo que ya cuente con el archivo del modelo.

## Ejecutar en Google Colab
Abra [GeoIA_Colab.ipynb](GeoIA_Colab.ipynb) en Google Colab. Siga las celdas para instalar dependencias, obtener `geoia.py` y entrenar y evaluar el sistema.

## Evaluación
En la ejecución documentada, el clasificador identificó correctamente **180 de 180 imágenes sintéticas de prueba (100 %)**, tras 12 épocas, con particiones de 720 imágenes para entrenamiento, 180 para validación y 180 para prueba. Los datos de prueba proceden de una generación independiente respecto del entrenamiento.

**Limitación importante:** la exactitud en figuras sintéticas no demuestra ese mismo desempeño con fotografías, dibujos a mano u otras imágenes externas.

## Estructura
- `geoia.py`: generación de datos, CNN, entrenamiento, evaluación y predicción.
- `GeoIA_Colab.ipynb`: cuaderno para ejecutar en Google Colab.
- `requirements.txt`: bibliotecas necesarias.
- `GUION_VIDEO.txt`: propuesta de exposición.

**Apoyo de IA:** se utilizaron herramientas de inteligencia artificial como apoyo para la redacción, organización de la documentación y preparación del prototipo. Los resultados cuantitativos indicados corresponden a pruebas ejecutadas del proyecto.
