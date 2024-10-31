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
import numpy as np

color = (0, 255, 255)
grosor = 2

# Cargar el clasificador Haarcascade para la detección de caras
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')

# Función para detectar la cara y obtener la región del pelo
def detect_face_and_hair(image):
    img_byn = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    caras = face_cascade.detectMultiScale(img_byn)
    for (x, y, ancho, alto) in caras:
        # Dibujar un rectángulo alrededor de la cara
        cv2.rectangle(image,(x, y),(x + ancho, y + alto),color, grosor)
        
        # Definir la región del pelo (por encima de la frente)
        hair_region = image[y-200:y, x:x+ancho]  # Ajusta estos valores según sea necesario
        
    cv2.imshow('Imagen completa',image)
    
    return hair_region

# Función para encontrar el color dominante
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

def classify_hair_color(hsv):
    # Definir los rangos HSV para diferentes colores de pelo
    colors = {
        "castaño claro": [(10, 50, 50), (30, 255, 255)],
        "castaño oscuro": [(10, 50, 20), (30, 255, 100)],
        "castaño": [(10, 50, 50), (20, 255, 150)],
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

# Cargar la imagen
#image_path = '../images/persona.jpg'
image_path = '../images/galgadot.jpg'
image = cv2.imread(image_path)

# Detectar la cara y obtener la región del pelo
hair_region = detect_face_and_hair(image)

if hair_region is not None:
    # Encontrar el color dominante en la región del pelo
    dominant_color = get_dominant_color(hair_region)
    
    # Quitar los decimales de los números
    bgr = [int(num) for num in dominant_color]

    # Convertir el color BGR a HSV
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hair_color = classify_hair_color(hsv)

    # Mostrar el color dominante
    print("Color de pelo (BGR):", hair_color, " ",bgr)
    img = np.ones((300, 300, 3), np.uint8)*255
    img [:] = bgr  # Color RGB
    cv2.imshow('Imagen color', img)

    # Mostrar la imagen con la región del pelo marcada
    cv2.imshow('Hair Region', hair_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se detectó ninguna cara en la imagen.")
