from src import movenet
from src import pguardian

import cv2

def analize_image(image, process_image=False):
    keypoints = movenet.predict_movenet_for_image(image, process_image=process_image)
    prediction, score = pguardian.predict(keypoints)

    return prediction, score

def analize_video(video, process_image=False):
    print("Analizando video...")
    if type(video) == str:
        video = cv2.VideoCapture(video)
         
    predictions = []
    scores = []
    frame_num = 0
    while True:
        frame_num += 1
        ret, frame = video.read()
        if not ret:
            break
        prediction, score = analize_image(frame, process_image)
        predictions.append(prediction)
        scores.append(score)
        video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES) + 6))
    video.release()
    
    errors = 0
    # haz promedio de predicciones y scores
    for prediction in predictions:
        if prediction == "mala_pose":
            errors += 1

    error = errors / frame_num * 100
    prom_score = sum(scores) / len(scores)
        

    return error, prom_score

