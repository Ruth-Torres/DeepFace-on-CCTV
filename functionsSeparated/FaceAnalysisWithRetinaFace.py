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

from deepface import DeepFace
import cv2

# Cargar la imagen
image_path = '../images/galgadot.jpg'
image = cv2.imread(image_path)

# Verificar si la imagen se ha cargado correctamente
if image is None:
    raise ValueError(f"Error al cargar la imagen desde la ruta: {image_path}")

# Analizar la imagen usando RetinaFace para la detección de caras
analysis = DeepFace.analyze(image, detector_backend='retinaface', actions=['age', 'gender', 'race', 'emotion'])

# Mostrar resultados
dicc = analysis[0]
print("Resultados del análisis facial:", dicc)
print("Age:", dicc['age'])
print("Gender:", dicc.get('dominant_gender'))
print("Race:", dicc.get('dominant_race'))
print("Emotion:", dicc.get('dominant_emotion'))

# Mostrar la imagen
cv2.imshow('Face Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
