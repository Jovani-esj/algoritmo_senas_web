import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Para guardar el modelo entrenado

# Inicializar variables
data_folder = "data"  # Asegúrate de tener la carpeta con las imágenes
letters = "abcdlmnopqrstuvwxyz"  # Define las letras que estás usando
X, y = [], []

# Cargar imágenes de entrenamiento
for letter in letters:
    for i in range(20):  # Ajusta el número de imágenes según tu dataset
        img_path = f"{data_folder}/{letter}/{letter}_{i}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256))  # Ajusta el tamaño según sea necesario
            X.append(img.flatten())  # Aplanar la imagen para usarla en el modelo SVM
            y.append(letter)

X = np.array(X)
y = np.array(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo SVM
clf = svm.SVC(kernel='linear', C=1)  # Usa un kernel lineal para la SVM
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
joblib.dump(clf, 'modelo_svm_letras.pkl')
print("Modelo entrenado y guardado exitosamente.")
