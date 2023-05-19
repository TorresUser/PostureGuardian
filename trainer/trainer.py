import os
import json
import numpy as np
import dotenv
import time
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


start_time = time.time()
# Versiona el modelo segun .env
version = int(dotenv.get_key('.env', 'LAST_MODEL_VERSION')) + 1 
version = str(version)
dotenv.set_key('.env', 'LAST_MODEL_VERSION', version)

datasets = os.listdir("./trainer/datasets")
current_model = "model_v" + str(version)
current_train = "train_" + str(len(datasets) + 1).zfill(4)
current_dataset = "./trainer/datasets/dataset_" + str(len(datasets) + 1).zfill(4) + "_model_v" + str(version)

# Definir la ruta de las carpetas de datos
cat_1_dir = './trainer/dataset/buena_pose/'
cat_2_dir = './trainer/dataset/mala_pose/'

# Obtener los archivos json de cada carpeta
cat_1_files = os.listdir(cat_1_dir)
cat_2_files = os.listdir(cat_2_dir)

# Cargar los datos de cada archivo json y guardarlos en listas
cat_1_data = []
for file in cat_1_files:
    with open(os.path.join(cat_1_dir, file), 'r') as f:
        data = json.load(f)
        data = np.array(data['keypoints'])
        data = (data - np.mean(data)) / np.std(data) # Normalizar los datos
        cat_1_data.append(data)

cat_2_data = []
for file in cat_2_files:
    with open(os.path.join(cat_2_dir, file), 'r') as f:
        data = json.load(f)
        data = np.array(data['keypoints'])
        data = (data - np.mean(data)) / np.std(data) # Normalizar los datos
        cat_2_data.append(data)

# Crear las etiquetas para cada conjunto de datos
cat_1_labels = np.ones(len(cat_1_data))
cat_2_labels = np.zeros(len(cat_2_data))

# Unir los datos y las etiquetas en un solo conjunto
X = np.array(cat_1_data + cat_2_data)
y = np.concatenate([cat_1_labels, cat_2_labels])


# Dividir el conjunto en datos de entrenamiento y datos de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
model = Sequential([
    Flatten(input_shape=(X.shape[1], X.shape[2])),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo entrenado

model.save(f'./src/models/model-v{version}.h5')


# Evaluar el modelo con los datos de prueba
loss, accuracy = model.evaluate(X_test, y_test)


# Guardame el accuracy en % con dos decimales sin redondear

accuracy = f'{accuracy*100:.2f}'
accuracy = float(accuracy)

loss = f'{loss:.2f}'
loss = float(loss)

print(f'Accuracy: {accuracy}%')
end_time = time.time()
training_time = end_time - start_time
print(f'Training time: {training_time}s')

shutil.make_archive(f'{current_dataset}', 'zip', './trainer/dataset')

data = {
    'model': f'model-v{version}.h5',
    "train": current_train,
    "dataset": {
        "total": len(cat_1_data) + len(cat_2_data),
        "buena_pose": len(cat_1_data),
        "mala_pose": len(cat_2_data)
    },
    'accuracy': accuracy,
    'loss': loss,
    'version': version,
    "training_time": training_time,
    "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "dataset_zip" : f'{current_dataset}.zip'
}
print(data)
# Guardar en models.json todos el historial de los modelos entrenados

with open('./trainer/models.json', 'r+') as f:
    models = json.load(f)
    models[current_train] = data
    f.seek(0)
    json.dump(models, f, indent=4)


