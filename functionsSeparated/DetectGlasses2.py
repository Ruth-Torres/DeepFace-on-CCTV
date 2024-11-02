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

# Cargar los clasificadores Haarcascade para la detección de caras y gafas
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

color = (0, 255, 255)
grosor = 2

def detect_glasses(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        glasses = glasses_cascade.detectMultiScale(roi_gray)

        # Dibujar círculos en los ojos
        for (x1, y1, ancho1, alto1) in faces:
            radio = int((ancho1 + alto1)/4)
            cv2.circle(image, (x1 + x + radio, y1 + y + radio), radio, color, grosor)
        if len(glasses) >= 1:
            return "La persona lleva gafas"
    return "La persona no lleva gafas"

# Cargar la imagen
image_path = '../images/galgadot.jpg'  # Asegúrate de que esta ruta sea correcta
image = cv2.imread(image_path)

# Verificar si la imagen se ha cargado correctamente
if image is None:
    raise ValueError(f"Error al cargar la imagen desde la ruta: {image_path}")

# Detectar si la persona lleva gafas
result = detect_glasses(image)
print(result)

# Mostrar la imagen con las detecciones
cv2.imshow('Glasses Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
