import cv2
import os

# Nombre del video de entrada
video_name = input("Nombre del video a procesar: ") + ".mp4"
video_path = "videos/" + video_name

# Crear la carpeta para las capturas
carpeta_name = input("Carpeta Name: ")
output_folder = "fotos_sin_rastreo/" + carpeta_name
os.makedirs(output_folder, exist_ok=True)

# Leer el video
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Contador de fotogramas
frame_count = 0
saved_frames = 0
total_frames = 500  # Número de capturas deseadas
skip_frames = 3    # Número de frames a saltar entre capturas

# Obtener la cantidad total de frames del video
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Procesar el video
while saved_frames < total_frames and frame_count < total_video_frames:
    ret, frame = cap.read()

    # Si no hay más fotogramas, salir del bucle
    if not ret:
        print("El video tiene menos frames de los necesarios.")
        break

    # Guardar el fotograma actual si es el frame adecuado
    if frame_count % skip_frames == 0:
        frame_path = os.path.join(output_folder, f"b_{saved_frames}.png")
        cv2.imwrite(frame_path, frame)
        saved_frames += 1

        # Mostrar progreso
        print(f"Guardado: {frame_path}")

    # Incrementar el contador de frames
    frame_count += 1

# Liberar recursos
cap.release()
print(f"Proceso terminado. Se guardaron {saved_frames} capturas en la carpeta '{output_folder}'.")
