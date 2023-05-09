from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization

#-------------------------[ Rutas y tamaños de datos ]--------------------------------
DATA_PATH = os.path.join('mp_data')
# definir las acciones a detectar
actions = np.array(['hola', 'como estas', 'vos', 'bien', 'mal', 'mas o menos', 'nos vemos', 'gracias', 'muchas gracias',
                    'de nada', 'encantado'])
# cantidad de videos por valor de dato
no_sequences = 30
# duracion de videos en cuadros frames
sequence_length = 30


# crear un diccionario para mapear las etiquetas de las acciones a números
label_map = {label:num for num, label in enumerate(actions)}

# crear dos listas vacías para almacenar las secuencias de datos y las etiquetas
sequences, labels = [], []
# recorrer todas las acciones
for action in actions:
    # recorrer todas las secuencias de datos para cada acción
    for sequence in range(no_sequences):
        # crear una lista vacía para almacenar los datos de cada fotograma de la secuencia
        window = []
        # recorrer todos los fotogramas de la secuencia
        for frame_num in range(sequence_length):
            # cargar los datos del fotograma correspondiente
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            # agregar los datos del fotograma a la lista
            window.append(res)
        # agregar la lista de datos de la secuencia a la lista de secuencias
        sequences.append(window)
        # agregar la etiqueta correspondiente a la lista de etiquetas
        labels.append(label_map[action])

# print(np.array(sequences).shape)

# convertir las listas de secuencias y etiquetas en arrays de numpy
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# dividir los datos en conjuntos de entrenamiento y prueba
# 5% de los datos se usaran para el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

#-----------------------------[ Construir el modelo de la red neuronal ]--------------------
"""
la capa LSTM se espera que reciba una secuencia de 30 fotogramas (como se especifica en el
parámetro input_shape=(30,1662)), donde cada fotograma se representa mediante un vector de
características de tamaño 1662.

En la implementación de este modelo, el vector de características de cada fotograma se
construye mediante la concatenación de los vectores de características de la pose, la cara
y las manos, que tienen tamaños de 132, 1404 y 63x2=126, respectivamente, lo que da un total de 1662.
"""
# definir la ruta donde se guardará el registro del entrenamiento
log_dir = os.path.join('Logs')
# definir el callback de TensorBoard
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(actions.shape[0], activation='softmax'))
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
# model.add(Dropout(0.2))     # 20% de las neuronas seran desactivadas
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(Dropout(0.2))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(actions.shape[0], activation='softmax'))
#
#--------------------[ Compilar el modelo ]-----------------------
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#------------------------[ Entrenar el modelo ]-----------------
model.fit(X_train, y_train, epochs=300, callbacks=[tb_callback])

#-------------------[ Imprimir un resumen del modelo ]------------------
model.summary()
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#----------------------[ Guardar el modelo ]---------------------
model.save('action.h5')
