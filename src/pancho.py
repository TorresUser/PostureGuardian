# Prepara el dataset para entrenar el modelo. 
import cv2
import os
import json
from src import movenet 
import time


def make_frames(video_path, image_output, keypoint_output=None, fps=12):
    # Abre el video
    start_time = time.time()
    video = cv2.VideoCapture(video_path)
    video_name = video_path.split("/")[-1].split(".")[0]
    print(f"Convirtiendo {video_name}...")
    
    # Verifica si el video se abrió correctamente
    if not video.isOpened():
        print("No se pudo abrir el video.")
        return
    
    # Crea la carpeta de salida si no existe
    os.makedirs(image_output, exist_ok=True)
    
    # Inicializa el contador de fotogramas
    count = 0
    
    # Lee el video y guarda los fotogramas en la carpeta de salida
    while True:
        # Lee el siguiente fotograma
        ret, frame = video.read()
        # Si no hay más fotogramas, termina el bucle
        if not ret:
            break
        if keypoint_output:
            # Procesa el fotograma para obtener los keypoints
            keypoints = movenet.predict_movenet_for_image(frame, process_image=False)
            
            # Guarda los keypoints en un archivo JSON
            output_path = os.path.join(keypoint_output, f"keypoints_{video_name}_f{count}.json")
            with open(output_path, "w") as f:
                json.dump(keypoints, f)

        # Guarda el fotograma en un archivo de imagen
        output_path = os.path.join(image_output, f"{video_name}_f{count}.jpg")
        cv2.imwrite(output_path, frame)
        
        # Incrementa el contador de fotogramas
        count += 1
        print(f"Convirtiendo fotograma {count} del video {video}...")
        video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES) + fps))
    
    # Cierra el video
    video.release()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Se han convertido {count} fotogramas. Los archivos se han guardado en {image_output}")
    print(f"Tiempo de ejecución: {execution_time} segundos")
