# MIT License
# Copyright (c) 2024 Ruth Torres Gallego
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import cv2
from deepface import DeepFace
import numpy as np
import mediapipe as mp

# Crear la carpeta 'intruders' si no existe
if not os.path.exists('./intruders'):
    os.makedirs('./intruders')

# Crear la carpeta 'captures' si no existe
if not os.path.exists('./captures'):
    os.makedirs('./captures')

# Propiedades para dibujar en una imagen (rectangulos, círculos, etc.)
color = (0, 255, 255)
grosor = 2

# Cargar los clasificadores Haarcascade para la detección de caras y ojos
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Analiza la cara de una persona y saca algunos rasgos (edad, género, raza, emoción)
def analyze_face(image):
    # Analizar la imagen usando RetinaFace para la detección de caras
    # analysis = DeepFace.analyze(image, detector_backend='retinaface', actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
    
    # Para un analisis más rápido pero menos preciso, no usar RetinaFace
    analysis = DeepFace.analyze(image, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
    
    # Extraer los resultados del análisis facial
    dicc = analysis[0]

    # Mostrar resultados
    # print("Resultados del análisis facial:", dicc)
    # print("Age:", dicc['age'])
    # print("Gender:", dicc.get('dominant_gender'))
    # print("Race:", dicc.get('dominant_race'))
    # print("Emotion:", dicc.get('dominant_emotion'))

    # Devolver los resultados del análisis facial
    return dicc

# Función para detectar la cara y obtener la región del pelo
# Función auxiliar de get_hair_color
def detect_face_and_hair(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Detectar caras
        results = face_detection.process(rgb_image)

        # Dibujar las detecciones en la imagen
        if results.detections:
            for detection in results.detections:
                # Dibujar la detección en la imagen
                # mp_drawing.draw_detection(image, detection)
                
                # Obtener las coordenadas de la caja delimitadora
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                ancho = int(bboxC.width * iw)
                alto = int(bboxC.height * ih)

                # Definir la región del pelo (por encima de la frente)
                hair_region_top = max(0, y - int(alto * 0.35))  # Ajustar según sea necesario
                hair_region_bottom = y-30 # Ajustar según sea necesario
                hair_region_left = x + 30
                hair_region_right = x + ancho - 30

                # Extraer la región del pelo como una nueva imagen
                hair_region = image[hair_region_top:hair_region_bottom, hair_region_left:hair_region_right]

    # Mostrar la imagen completa
    resized = cv2.resize(image, (0, 0), fx=0.9, fy=0.9)    
    cv2.imshow('Imagen completa', resized)
    
    # Mostrar la imagen con la región del pelo marcada
    cv2.imshow('Hair Region', hair_region)

    return hair_region

# Función para encontrar el color dominante
# Función auxiliar de get_hair_color
def get_dominant_color(image, k=4):
    # Convertir la imagen a un array de píxeles
    pixels = np.float32(image.reshape(-1, 3))
    
    # Usar K-means para encontrar los colores dominantes
    _, labels, palette = cv2.kmeans(pixels, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Contar la frecuencia de cada color
    _, counts = np.unique(labels, return_counts=True)
    
    # Encontrar el color más frecuente
    dominant_color = palette[np.argmax(counts)]
    
    return dominant_color

# Función para clasificar el color del pelo
# Función auxiliar de get_hair_color
def classify_hair_color(hsv):
    # Definir los rangos HSV para diferentes colores de pelo
    colors = {
        "castanio claro": [(10, 50, 50), (30, 255, 255)],
        "castanio oscuro": [(10, 50, 20), (30, 255, 100)],
        "castanio": [(10, 50, 50), (20, 255, 150)],
        "negro": [(0, 0, 0), (180, 255, 50)],
        "rubio": [(20, 50, 150), (40, 255, 255)],
        "pelirrojo": [(0, 50, 50), (10, 255, 255)],
        "canoso": [(0, 0, 50), (180, 50, 255)]
    }

    # Verificar en qué rango de color cae el valor HSV dado
    for color_name, (lower_bound, upper_bound) in colors.items():
        if all(lower_bound[i] <= hsv[i] <= upper_bound[i] for i in range(3)):
            return color_name

    return "otro color"

# Analiza el color del pelo de una persona
def get_hair_color(image):
    # Detectar la cara y obtener la región del pelo
    hair_region = detect_face_and_hair(image)

    if hair_region is not None:
        # Encontrar el color dominante en la región del pelo
        dominant_color = get_dominant_color(hair_region)
        
        # Quitar los decimales de los números
        bgr = [int(num) for num in dominant_color]

        # Convertir el color BGR a HSV
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]

        # Clasificar el color del pelo
        hair_color = classify_hair_color(hsv)

        # Mostrar el color dominante
        # print("Color de pelo (BGR):", hair_color, " ",bgr)

        # Mostrar una imagen con el color del pelo
        img = np.ones((300, 300, 3), np.uint8)*255
        img [:] = bgr  # Color RGB
        cv2.imshow('Imagen color de pelo', img)

        # Devolver el color del pelo
        color = hair_color # + " " + str(bgr)
        return color
    else:
        print("No se detectó ninguna cara en la imagen.")

# Función para detectar si la persona lleva gafas
def detect_glasses(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        """
        # Código para pintar círculos en los ojos
        for (x1, y1, ancho1, alto1) in eyes:
            radio = int((ancho1 + alto1)/4)
            cv2.circle(image, (x1 + x + radio, y1 + y + radio), radio, color, grosor)
        """
        # Ya que no es posible detectar gafas directamente con Haarcascades, 
        # ara detectar gafas habrá que verificar si hay al menos dos ojos por cara
        if len(eyes) >= 2:
            return "No lleva gafas"
    return "Lleva gafas"


# Obtenemos la ruta del directorio que contiene las imágenes
# (se hace de esta forma para que funcione en cualquier sistema operativo)
# Ruta del directorio actual
directorio_actual = os.getcwd()

# Ruta a la carpeta 'captures'
directorio = os.path.join(directorio_actual, 'captures')
# print(f"Ruta completa a 'captures': {directorio}")

# Obtener la lista de archivos en el directorio "captures"
archivos = os.listdir(directorio)

# Filtrar solo los archivos de imagen (con extensiones .jpg, .jpeg, .png)
extensiones_validas = ['.jpg', '.jpeg', '.png']
imagenes = [archivo for archivo in archivos if os.path.splitext(archivo)[1].lower() in extensiones_validas]

# Bucle para abrir y analizar cada imagen
for imagen in imagenes:
    nombre_archivo, _ = os.path.splitext(imagen)  # Obtener el nombre del archivo sin la extensión
    ruta_imagen = os.path.join(directorio, imagen)
    print(f"Analizando la imagen: {ruta_imagen}")  # Imprimir la ruta completa de la imagen
    img = cv2.imread(ruta_imagen)
    
    if img is not None:
        try:
            dicc = analyze_face(img)
            hair_color = get_hair_color(img)
            glasses_result = detect_glasses(img)

            # Redimensionar la imagen
            resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

            # Definir el tamaño del padding
            top = 150
            bottom = 300
            left = 100
            right = 100

            # Definir el color del padding (en este caso, blanco)
            color = [255, 255, 255]

            # Añadir el padding a la imagen
            padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            # Obtener las dimensiones de la imagen con padding
            (h, w) = padded_image.shape[:2]
            # Obtener el tamaño del texto "INTRUDER"
            (text_width, text_height), baseline = cv2.getTextSize("INTRUDER", cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            # Calcular la posición para centrar el texto "INTRUDER" horizontalmente
            x_position = (w - text_width) // 2

            # Añaadir la información de intrusos a la imagen
            cv2.putText(padded_image, f"INTRUDER", (x_position, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
            cv2.putText(padded_image, f"Age: {dicc['age']}", (100, h-230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(padded_image, f"Gender: {dicc.get('dominant_gender')}", (100, h-190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(padded_image, f"Race: {dicc.get('dominant_race')}", (100, h-150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(padded_image, f"Emotion: {dicc.get('dominant_emotion')}", (100, h-110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(padded_image, f"Hair Color: {hair_color}", (100, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(padded_image, f"Glasses: {glasses_result}", (100, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Guardar la imagen con la información de intrusos
            cv2.imwrite(f'./intruders/intruder_info_{nombre_archivo}.jpg', padded_image)
            cv2.imshow('Intruder', padded_image)
            cv2.waitKey(0)  # Esperar a que se presione una tecla para analizar la siguiente imagen
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error al analizar la imagen {nombre_archivo}: {e}")
    else:
        print(f"No se pudo abrir la imagen: {ruta_imagen}")
