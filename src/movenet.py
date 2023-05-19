# Recibe path de imagen y devuelve los keypoints "procesados"* de la persona detectada en la imagen junto con la imagen con los keypoints dibujados. 

#* "procesados" significa que se eliminó el score de cada keypoint.

import os
import json
import tensorflow as tf
import numpy as np
from .helper import draw_prediction_on_image
from matplotlib import pyplot as plt

# Input size for the model 
input_size = 256

def predict_movenet_for_image(image, output_path:str=None, process_image:bool=True):
    '''Recibe una imagen o un path de imagen y devuelve los keypoints procesados de la persona detectada en la imagen junto con la imagen con el overlay si process_image=True. Guarda el overlay en output_path si se especifica.'''

    if type(image) == str:
        image = tf.io.read_file(image)
        image = tf.image.decode_jpeg(image)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run model inference.

    interpreter = tf.lite.Interpreter(model_path="./src/models/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite")

    interpreter.allocate_tensors()
    """Runs detection on an input image.

    Args:
        input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    array = np.array(keypoints_with_scores)

    # Obtener un array con las positions que necesitas
    positions = array[0, :, :, :2]
    positions = positions[0]

    # Guardar los datos procesados en un nuevo archivo
    keypoints = {'keypoints': positions.tolist()}
    # print(etiquetar_keypoints(keypoints_with_scores))

    # Procesar la imagen si es necesario
    if process_image:
        # Visualize the predictions with image.
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(
            display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

        # Guardar la imagen procesada si es necesario
        if output_path is not None:
            plt.imsave(output_path, output_overlay)

        # Devolver los keypoints y la imagen procesada si es necesario guardarla
        return keypoints, output_overlay
    
    # Devolver los keypoints si no es necesario procesar la imagen
    return keypoints