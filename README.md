# Reconocimiento de rasgos faciales en imágenes tomadas desde una cámara

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE.md` para más detalles.

## Introdución
El proyecto trata sobre detectar "intrusos" a través de la camara, y guardar una captura del intruso.
Luego, se analizarán los rasgos faciales del intruso y se generará una imagen con la cara de dicho intruso y una pequeña descripción con los rasgos detectados.

## Organización del repositorio
### Archivo `main.py`
El programa principal del proyecto, con todas las funcionalidades incluidas. Actualmente NO FUNCIONAL.

### Archivo `FacesDetection.py`
El programa principal con toda la parte de analisis facial. Actualmente FUNCIONAL.

### Carpeta functionsSeparated
En esta carpeta se encuentran las distintas funciones utilizadas en el programa principal por separado.

### Carpeta haarcascades
Son clasificadores ya entrenados e importados desde el respositorio de OpenCV. En esta carpeta se encuentran los clasificadores de **caras** y **ojos**.

### Carpeta images
Una serie de imagenes de prueba para las distintas funciones.

### Carpeta info
Archivos con infomación adicional sobre las librearías usadas.

### Carpeta captures
Las imágenes con caras detectadas por la camara.

### Carpeta intruders
Las imágenes generadas con la descripción de la cara.

## Instalaciones previas
Para poder ejecutar este programa, será necesario instalar algunas librerias:
- Python (La versión utilizada para este proyecto es la 3.12.6)

- OpenCV: una librería de computación visual para el procesamiento de imágenes en Python.
```
pip install opencv-python
```

- Numpy: una librería para crear vectores y matrices multidimensionales (arrays) en Python.
```
pip install numpy
```

- DeepFace: una librería para la detección de caras y el análisis facial en Python.
```
pip install deepface
```

- TensorFlow: una librería de Machine Learning desarrollada por Google, necesaria para usar DeepFace.
```
pip install tf-keras
```

- MediaPipe: es un marco de trabajo de código abierto desarrollado por Google para construir pipelines de aprendizaje automático multimodales. Es especialmente útil para tareas de visión por computadora, como el reconocimiento facial, el seguimiento de manos, la segmentación de imágenes, entre otros
```
pip install mediapipe
```

## Ejecución
Basta con ejecutar el archivo llamado `main.py`, ya sea a través del propio IDE o con el siguiente comando:
```
python main.py
```