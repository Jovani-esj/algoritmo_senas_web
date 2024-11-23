import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib  # Para guardar el modelo entrenado

# Inicializar variables
data_folder = "data"  # Asegúrate de tener la carpeta con las imágenes
letters = "abc"#abcdefghijklmnopqrstuvwxyz
X, y = [], []

# Cargar imágenes de entrenamiento
for letter in letters:
    for i in range(500):
        img_path = f"{data_folder}/{letter}/{letter}_{i}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Ajusta el tamaño según sea necesario
            X.append(img.flatten())
            y.append(letter)

X = np.array(X)
y = np.array(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo SVM
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(clf, 'modelo_svm_letras.pkl')
print("Modelo entrenado y guardado exitosamente.")
