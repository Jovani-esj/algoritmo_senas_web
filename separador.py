import cv2
import os

# Nombre del video de entrada
video_name = "video_" + input("Nombre del video a procesar: ") + ".mp4"
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

# Pedir la cantidad de capturas deseadas
total_frames = int(input("Número de capturas deseadas: "))

# Obtener la cantidad total de frames del video
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames > total_video_frames:
    print(f"El número de capturas deseadas ({total_frames}) es mayor al número total de frames del video ({total_video_frames}).")
    total_frames = total_video_frames
    print(f"Se ajustará para capturar un frame por cada fotograma disponible ({total_video_frames}).")

# Calcular el intervalo entre capturas
frame_interval = total_video_frames // total_frames

# Procesar el video
frame_count = 0
saved_frames = 0

while saved_frames < total_frames:
    ret, frame = cap.read()

    # Si no hay más fotogramas, salir del bucle
    if not ret:
        print("Error: No se pudieron leer más fotogramas del video.")
        break

    # Guardar el fotograma actual si es el adecuado
    if frame_count % frame_interval == 0:
        frame_path = os.path.join(output_folder, carpeta_name + f"_{saved_frames}.png")
        cv2.imwrite(frame_path, frame)
        saved_frames += 1

        # Mostrar progreso
        print(f"Guardado: {frame_path}")

    # Incrementar el contador de frames
    frame_count += 1

# Liberar recursos
cap.release()
print(f"Proceso terminado. Se guardaron {saved_frames} capturas en la carpeta '{output_folder}'.")
