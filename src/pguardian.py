# Recibe los keypoints procesados de la persona detectada en la imagen y devuelve :ok: si la pose es correcta, :warning: si la pose es incorrecta y :x: si no se detecta la pose. PONELE . Además devuelve el score de la pose detectada.

import tensorflow as tf
import numpy as np
import json

# Load the saved model

# Define a function to preprocess the data and make predictions
def predict(data):
    model = tf.keras.models.load_model('modelo.h5')
    # Extract the keypoints from the JSON file

    keypoints = data['keypoints']

    # Reshape the keypoints array to match the model's input shape
    keypoints = np.array(keypoints).reshape(1, 17, 2)

    # Normalize the keypoints
    print(keypoints)
    keypoints_norm = (keypoints - np.mean(keypoints)) / np.std(keypoints)
    print(keypoints_norm)

    # Make predictions using the model
    y_pred = model.predict(keypoints_norm)

    # Convert the predicted probabilities to a binary label
    if y_pred > 0.5:
        label = 'arriba'
        score = y_pred
    else:
        label = 'abajo'
        score = 1 - y_pred

    # convert the numpy array to a int with one decimal place but no rounding
    score = int(score * 1000) / 10    
    return label, score


# Example usage
data = {'keypoints': [[0.3846595883369446, 0.46221697330474854], [0.37319162487983704, 0.4746403992176056], [0.37329304218292236, 0.45039796829223633], [0.3830275535583496, 0.4931066334247589], [0.3829796612262726, 0.4338470995426178], [0.43629926443099976, 0.49987372756004333], [0.4378158450126648, 0.41646891832351685], [0.38333672285079956, 0.5633391737937927], [0.3816681206226349, 0.35007932782173157], [0.29026979207992554, 0.5700106024742126], [0.29460954666137695, 0.3569708466529846], [0.6084905862808228, 0.4925762414932251], [0.6069090366363525, 0.4224661588668823], [0.7345713376998901, 0.49883294105529785], [0.7394946217536926, 0.42075279355049133], [0.8468168377876282, 0.5078426003456116], [0.851485013961792, 0.4189421236515045]]}
label,score = predict(data)
print(f'Tiene los brazos {label}. Con una probabilidad de {score}%')
