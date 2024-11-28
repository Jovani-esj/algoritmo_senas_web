import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib  # Para guardar el modelo entrenado

# Inicializar variables
data_folder = "videos"  # Carpeta que contiene los videos
letters = "abcdlmnopqrstuvwxyz"  # Letras a procesar
X, y = [], []


# Función para preprocesar un cuadro (frame)
def preprocesar_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Reducción de ruido
    frame = cv2.resize(frame, (128, 128))  # Redimensionar
    frame = cv2.equalizeHist(frame)  # Mejorar contraste
    frame = frame / 255.0  # Normalizar
    return frame.flatten()


# Cargar videos y extraer cuadros
for letter in letters:
    video_path = f"{data_folder}/video_{letter}.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"No se pudo abrir el video para la letra: {letter}")
        continue

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar y agregar cada 10mo cuadro para reducir redundancia
        if frame_count % 50 == 0:
            processed_frame = preprocesar_frame(frame)
            X.append(processed_frame)
            y.append(letter)

        frame_count += 1

    cap.release()

# Convertir a matrices NumPy
X = np.array(X)
y = np.array(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo SVM
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(clf, 'modelo_svm_letras_videos.pkl')
print("Modelo entrenado y guardado exitosamente.")
