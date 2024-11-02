# DeepFace on CCTV: reconocimiento de rasgos faciales en imágenes tomadas desde una cámara

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE.md` para más detalles.

## Introdución
El proyecto trata sobre detectar "intrusos" a través de la cámara, y guardar una captura del intruso.
Luego, se analizarán los rasgos faciales del intruso y se generará una imagen con la cara de dicho intruso y una pequeña descripción con los rasgos detectados.

## Organización del repositorio
### Archivo `main.py`
El programa principal del proyecto. Ejecuta primero el archivo `Camera_CCTV.py` y una vez termine, ejecutará el archivo `AnalyzeAllImages.py`.

### Archivo `Camera_CCTV.py`
El programa con toda la parte de capturar caras con la cámara. Abrirá la cámara y cada vez que detecte una cara, guardará una captura en la carpeta *captures* con su timestamp correspondiente.

### Archivo `AnalyzeAllImages.py`
El programa con toda la parte de análisis facial. Analizará todas las imágenes en la carpeta *captures* y creará un cartel de **INTRUDER** en la carpeta *intruders* con sus datos.

### Carpeta functionsSeparated
En esta carpeta se encuentran las distintas funciones utilizadas en el programa principal por separado.

### Carpeta haarcascades
Son clasificadores ya entrenados e importados desde el repositorio de OpenCV. En esta carpeta se encuentran los clasificadores de **caras** y **ojos**.

### Carpeta images
Una serie de imágenes de prueba para las distintas funciones.

### Carpeta info
Archivos con información adicional sobre las librerías usadas.

### Carpeta captures
Las imágenes con caras detectadas por la cámara.

### Carpeta intruders
Las imágenes generadas con la descripción de la cara.

## Instalaciones previas
Para poder ejecutar este programa, será necesario instalar algunas librerias:
- Python (La versión utilizada para este proyecto es la 3.12.6)

- OS: una librería que viene instalada con Python. Proporciona funcionalidad independiente del sistema operativo.

- Datetime: una librería que viene instalada con Python y sirve para manejar fechas en Python.

- Subprocess: una librería que viene instalada con Python. Es una herramienta que te permite ejecutar otros programas o comandos desde tu código Python.

- OpenCV: una librería de computación visual para el procesamiento de imágenes en Python.
```
pip install opencv-python
```

- Numpy: una librería para crear vectores y matrices multidimensionales (arrays) en Python.
IMPORTANTE: para este proyecto se ha usado la versión 1.26.4, ya que la librería MediaPipe necesita una versión de Numpy anterior a la 2.0.0 y la librería TensorFlow necesita una versión de Numpy entre la 2.1.0 (sin incluir) y la 1.26.0 (incluida). Para instalar Numpy-1.26.4, se ejecuta el siguiente comando:

```
pip install numpy==1.26.4
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