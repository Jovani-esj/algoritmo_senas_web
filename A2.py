import cv2
import mediapipe as mp
import os
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Nuevo import para drawing_utils
mp_drawing = mp.solutions.drawing_utils

# Carpeta que contiene las imágenes en formato PNG
folder_input_name = input("Ingresa la letra del folder para tomar las imágenes: ")

input_folder = "fotos_sin_rastreo/" + folder_input_name  # Cambia esto a la ruta de tu carpeta con imágenes
output_folder = "data"  # Carpeta donde se guardarán las imágenes procesadas
current_letter = input('Letra (misma que anterior) para guardar resultado: ')  # Inicializar con la primera letra
capture_count = 0

# Asegúrate de que la carpeta de salida exista
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Obtener una lista de todos los archivos PNG en la carpeta
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# Procesar cada imagen PNG automáticamente
for image_file in image_files:
    # Cargar la imagen
    image_path = os.path.join(input_folder, image_file)
    frame = cv2.imread(image_path)

    # Convertir la imagen a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos
    results = hands.process(rgb_frame)

    # Comprobar si se detecta la mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibuja los puntos de la mano en la imagen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convierte los landmarks a un array NumPy
            landmarks_array = np.array(
                [[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in hand_landmarks.landmark[0:21]])

            # Define la región de la mano (ajusta según sea necesario)
            x, y, w, h = cv2.boundingRect(landmarks_array)

            # Ampliar la región de recorte
            margin = 70  # Puedes ajustar este valor para ampliar más o menos la región
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)

            hand_roi = frame[y:y + h, x:x + w]

            # Guarda la imagen en la carpeta correspondiente a la letra del abecedario
            folder_path = os.path.join(output_folder, current_letter)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            cv2.imwrite(f"{folder_path}/{current_letter}_{capture_count}.png", hand_roi)
            print(f"Imagen {capture_count} capturada para la letra {current_letter}")

            # Incrementar el contador sin cambiar de letra
            capture_count += 1

# Liberar los recursos de MediaPipe
hands.close()
