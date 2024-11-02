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

# Crear la carpeta 'captures' si no existe
if not os.path.exists('./captures'):
    os.makedirs('./captures')

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Configurar la captura de video
camara = cv2.VideoCapture(0)
if not camara.isOpened():
    print("No es posible abrir la cámara")
    exit()

# Crear el detector de caras
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while camara.isOpened():
        ret, frame = camara.read()
        if not ret:
            print("No es posible obtener la imagen")
            break

        # Convertir el frame a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar caras
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Dibujar las detecciones en el frame
                # mp_drawing.draw_detection(frame, detection)
                
                # Guardar la imagen con la cara detectada en la carpeta 'capturas'
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                cv2.imwrite(f'../captures/capture_{timestamp}.jpg', frame)

        # Mostrar el frame con las detecciones
        cv2.imshow('Cámara de Vigilancia', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la captura de video y cerrar todas las ventanas
camara.release()
cv2.destroyAllWindows()