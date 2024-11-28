import cv2
import mediapipe as mp
import numpy as np
import joblib  # Para cargar el modelo guardado
import threading

# Cargar el modelo entrenado
clf = joblib.load('modelo_svm_letras_videos.pkl')  # Asegúrate de que el nombre coincida con el archivo guardado

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Import para drawing_utils
mp_drawing = mp.solutions.drawing_utils

# Configuración de OpenCV
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara predeterminada, puedes cambiar esto según tus necesidades

# Variables globales para comunicación entre hilos
frame = None
resultados_mano = None
predicted_letter = None
lock = threading.Lock()  # Lock para manejar el acceso a variables compartidas

def captura_video():
    global frame
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("No se puede abrir la cámara.")
            break

        with lock:
            frame = img.copy()


def procesar_mano():
    global frame, resultados_mano, predicted_letter

    while True:
        # Captura el frame actual bajo el bloqueo
        with lock:
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar manos
        resultados_mano = hands.process(rgb_frame)

        if resultados_mano.multi_hand_landmarks:
            for hand_landmarks in resultados_mano.multi_hand_landmarks:
                # Captura landmarks para predicción
                landmarks_array = np.array([[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in hand_landmarks.landmark[0:21]])
                if landmarks_array.size > 0:  # Verificar si landmarks_array no está vacío
                    try:
                        x, y, w, h = cv2.boundingRect(landmarks_array)
                        hand_roi = cv2.resize(frame[y:y+h, x:x+w], (128, 128), interpolation=cv2.INTER_AREA)
                        hand_roi_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                        hand_roi_flatten = hand_roi_gray.flatten()

                        # Realizar la predicción con el modelo SVM
                        predicted_letter = clf.predict([hand_roi_flatten])[0]

                    except Exception as e:
                        print(f"Error al procesar la mano: {e}")


# Iniciar los hilos para captura y procesamiento
hilo_captura = threading.Thread(target=captura_video, daemon=True)
hilo_procesamiento = threading.Thread(target=procesar_mano, daemon=True)

hilo_captura.start()
hilo_procesamiento.start()

while True:
    with lock:
        if frame is None:
            continue
        display_frame = frame.copy()

    # Dibujar los resultados si existen
    if resultados_mano and resultados_mano.multi_hand_landmarks:
        for hand_landmarks in resultados_mano.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar la letra predicha en la ventana
    if predicted_letter is not None:
        cv2.putText(display_frame, f'Letra: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Tracking", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
