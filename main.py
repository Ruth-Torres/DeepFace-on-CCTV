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

import cv2
import mediapipe as mp
import datetime
import os
import numpy as np
from deepface import DeepFace

# Crear la carpeta 'intruders' si no existe
if not os.path.exists('../intruders'):
    os.makedirs('../intruders')

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Configurar la captura de video
camara = cv2.VideoCapture(0)

# Propiedades para dibujar en una imagen (rectángulos, círculos, etc.)
color = (0, 255, 255)
grosor = 2

# Cargar los clasificadores Haarcascade para la detección de caras y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Función para analizar la cara de una persona y extraer algunos rasgos (edad, género, raza, emoción)
def analyze_face(image):
    analysis = DeepFace.analyze(image, detector_backend='retinaface', actions=['age', 'gender', 'race', 'emotion'])
    dicc = analysis[0]
    return dicc

# Función para detectar la cara y obtener la región del pelo
def detect_face_and_hair(image):
    img_byn = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    caras = face_cascade.detectMultiScale(img_byn)
    for (x, y, ancho, alto) in caras:
        cv2.rectangle(image, (x, y), (x + ancho, y + alto), color, grosor)
        hair_region = image[y-200:y, x:x+ancho]
    return hair_region

# Función para encontrar el color dominante
def get_dominant_color(image, k=4):
    pixels = np.float32(image.reshape(-1, 3))
    _, labels, palette = cv2.kmeans(pixels, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color = palette[np.argmax(counts)]
    return dominant_color

# Función para clasificar el color del pelo
def classify_hair_color(hsv):
    colors = {
        "castaño claro": [(10, 50, 50), (30, 255, 255)],
        "castaño oscuro": [(10, 50, 20), (30, 255, 100)],
        "castaño": [(10, 50, 50), (20, 255, 150)],
        "negro": [(0, 0, 0), (180, 255, 50)],
        "rubio": [(20, 50, 150), (40, 255, 255)],
        "pelirrojo": [(0, 50, 50), (10, 255, 255)],
        "canoso": [(0, 0, 50), (180, 50, 255)]
    }
    for color_name, (lower_bound, upper_bound) in colors.items():
        if all(lower_bound[i] <= hsv[i] <= upper_bound[i] for i in range(3)):
            return color_name
    return "otro color"

# Función para obtener el color del pelo
def get_hair_color(image):
    hair_region = detect_face_and_hair(image)
    if hair_region is not None:
        dominant_color = get_dominant_color(hair_region)
        bgr = [int(num) for num in dominant_color]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        hair_color = classify_hair_color(hsv)
        return hair_color
    return "No se detectó ninguna cara en la imagen."

# Función para detectar si la persona lleva gafas
def detect_glasses(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return "La persona no lleva gafas"
    return "La persona lleva gafas"

# Crear el detector de caras
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while camara.isOpened():
        ret, frame = camara.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_path = f'../intruders/intruder_{timestamp}.jpg'
                cv2.imwrite(image_path, frame)

                # Analizar la imagen guardada
                image = cv2.imread(image_path)
                if image is not None:
                    dicc = analyze_face(image)
                    hair_color = get_hair_color(image)
                    glasses_result = detect_glasses(image)

                    # Crear una imagen con la información de intrusos
                    info_image = np.zeros((500, 500, 3), np.uint8)
                    cv2.putText(info_image, f"Age: {dicc['age']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(info_image, f"Gender: {dicc.get('dominant_gender')}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(info_image, f"Race: {dicc.get('dominant_race')}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(info_image, f"Emotion: {dicc.get('dominant_emotion')}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(info_image, f"Hair Color: {hair_color}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(info_image, f"Glasses: {glasses_result}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Guardar la imagen con la información de intrusos
                    cv2.imwrite(f'../intruders/intruder_info_{timestamp}.jpg', info_image)

        cv2.imshow('Camara de Vigilancia', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camara.release()
cv2.destroyAllWindows()
