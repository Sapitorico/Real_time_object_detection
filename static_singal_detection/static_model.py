#!/usr/bin/python
from tensorflow.keras import backend as K #Si hay una sesion de keras, lo cerramos para tener todo limpio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import os

K.clear_session()  #Limpiamos todo


class CNN_Model:

    def __init__(self, DATASET_DIR=None, height=None, width=None, sequence_length=None, size_pool=None):
        self.DATASET_DIR = DATASET_DIR
        self.height = height
        self.width = width
        self.sequence_length = sequence_length
        self.size_pool = size_pool

    def Build_model(self, clasess, kernels_layer1,size_kernel1, kernels_layer2, size_kernel2, kernels_layer3, size_kernel3, size_pool):
        CNN = Sequential()  #Red neuronal secuencial
        #Agregamos filtros con el fin de volver nuestra imagen muy profunda pero pequeña
        CNN.add(Conv2D(kernels_layer1, size_kernel1, padding = 'same', input_shape=(self.width, self.width, 3), activation = 'relu'))
        CNN.add(MaxPooling2D(pool_size=size_pool)) #Despues de la primera capa vamos a tener una capa de max pooling y asignamos el tamaño
        CNN.add(Conv2D(kernels_layer2, size_kernel2, padding = 'same', activation='relu')) #Agregamos nueva capa
        CNN.add(MaxPooling2D(pool_size=size_pool))
        CNN.add(Conv2D(kernels_layer3, size_kernel3, padding = 'same', activation='relu')) #Agregamos nueva capa
        CNN.add(MaxPooling2D(pool_size=size_pool))
        #Ahora vamos a convertir esa imagen profunda a una plana, para tener 1 dimension con toda la info
        CNN.add(Flatten())  #Aplanamos la imagen
        CNN.add(Dense(640,activation='relu'))  #la cantidad de nueronas depende de la cantidad de clases 256 funcionan bien con 2 clases
        CNN.add(Dropout(0.5)) #Apagamos el 50% de las neuronas en la funcion anterior para no sobreajustar la red
        CNN.add(Dense(clasess, activation='softmax'))  #es la que nos dice la probabilidad de que sea alguna de las clases
        return CNN

    def Create_train_set(self):
        train_set = image_dataset_from_directory(
            self.DATASET_DIR,
            label_mode="categorical",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.height, self.width),
            batch_size=self.sequence_length,
        )
        return train_set

    def Create_validation_set(self):
        val_set = image_dataset_from_directory(
            self.DATASET_DIR,
            label_mode="categorical",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.height, self.width),
            batch_size=self.sequence_length
        )
        return val_set

    def Fit_model(self, CNN, lr, train_set, val_set, steps, val_steps, iterations, ):
        # Agregamos parametros para optimizar el modelo
        #Durante el entrenamiento tenga una autoevalucion, que se optimice con Adam, y la metrica sera accuracy
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        optimizer = Adam(learning_rate= lr)
        CNN.compile(loss = 'categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
        #Entrenaremos nuestra red
        CNN.fit(train_set, steps_per_epoch=steps, epochs= iterations,
                validation_data=val_set, validation_steps=val_steps,
                callbacks=[tb_callback])
        CNN.summary()
        return CNN

    def Save_model(self, CNN):
        #Guardamos el modelo
        CNN.save('Modelo.h5')
        CNN.save_weights('pesos.h5')
