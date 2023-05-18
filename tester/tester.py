# Ejecuta la IA en todos los inputs en los directorios "cat_1" y "cat_2" y guarda un nuevo test en el directorio "tests".
# Por cada test se guardan los inputs y outputs en 2 directorios, en outputs se guardan las imágenes con el overlay del BD y un json con data de los keypoints, el label de la imagen y la predicción de la IA.
# También se guardan datos de cada test en un json "data" general y uno "result" específico de cada test.

import os
import json
import time
from src import movenet
from src import pguardian


start_time = time.time()
tests = os.listdir("./tester/tests")
current_test = "./tester/tests/test" + str(len(tests) + 1).zfill(4)
os.makedirs(current_test)

os.makedirs(current_test + "/inputs")
os.makedirs(current_test + "/outputs")

categories = ["arriba", "abajo"]
correct_predictions = 0

num_of_images = {
    "total":0
}
for category in categories:
    num_of_images[category] = 0


for category in categories:
    os.makedirs(current_test + "/inputs/" + category)
    os.makedirs(current_test + "/outputs/" + category)
    images = os.listdir("./tester/" + category)
    num_of_images["total"] += len(images)
    num_of_images[category] = len(images)

    for image in images:
        keypoints, overlay = movenet.predict_movenet_for_image("./tester/" + category + "/" + image, current_test + "/outputs/" + category + "/" + image, process_image=True)
        prediction, score = pguardian.predict(keypoints)

        if prediction == category:
            correct_predictions += 1
        
        data = {
            "image": image,
            "keypoints": keypoints,
            "label": category,
            "prediction": prediction,
            "score": score
        }
        with open(current_test + "/outputs/"+ category + "/" + image.split(".")[0] + ".json", "w") as outfile:
            json.dump(data, outfile)

        # Moveme la imagen a inputs
        os.rename("./tester/" + category + "/" + image, current_test + "/inputs/" + category + "/" +  image)

end_time = time.time()
execution_time = end_time - start_time

accuracy = correct_predictions / num_of_images["total"] * 100
test = {
    "accuracy": accuracy,
    "total_images": num_of_images["total"]
}
for category in categories:
    test[f"{category}_images"] = num_of_images[category]

test["execution_time"] = execution_time
test["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
test["pancho"] = "🌭"

with open(current_test + "/result.json", "w") as outfile:
    json.dump(test, outfile)

with open("./tester/data.json", "r+") as outfile:
    data = json.load(outfile)
    type(data)
    data["test" + str(len(tests) + 1).zfill(4)] = test
    # Guarda el json con los datos de todos los tests pisando el anterior
    outfile.seek(0)
    json.dump(data, outfile, indent=4)




