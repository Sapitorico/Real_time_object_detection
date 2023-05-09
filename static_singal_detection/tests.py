
# import numpy as np
# def Get_resized_image(imgHand):
#     if imgHand is not None:
#         try:
#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             if imgHand.shape[1] <= imgWhite.shape[0] and imgHand.shape[0] <= imgWhite.shape[0]:
#                 x_start = int((imgWhite.shape[1] - imgHand.shape[1]) / 2)
#                 y_start = int((imgWhite.shape[0] - imgHand.shape[0]) / 2)
#                 imgWhite[y_start:y_start + imgHand.shape[0], x_start:x_start + imgHand.shape[1]] = imgHand
#             else:
#                 # imgHand = cv2.resize(imgHand, (imgWhite.shape[1] - x_offset, imgWhite.shape[0] - y_offset)) * 255
#                 imgHand = cv2.resize(imgHand, (imgWhite.shape[1], imgWhite.shape[0]))
#                 x_start = int((imgWhite.shape[1] - imgHand.shape[1]) / 2)
#                 y_start = int((imgWhite.shape[0] - imgHand.shape[0]) / 2)
#                 imgWhite[y_start:y_start + imgHand.shape[0], x_start:x_start + imgHand.shape[1]] = imgHand
#             cv2.imshow("imgWhite", imgWhite)
#             if key == ord('q'):
#                 for action in actions:
#                     for sequence in range(save_frequency):
#                         image_path = os.path.join(DATA_PATH, action, f'{sequence}.png')
#                         cv2.imwrite(image_path, imgWhite)
#         except ValueError as e:
#             print("Error: ", e)

# iterations = 100  #Numero de iteraciones (epocas)
# height , width = 200, 200
# sequence_length = 1 #Numero de imagenes que vamos a enviar batch_size

# steps = 5000/sequence_length  #Numero de veces que se va a procesar la informacion en cada iteracion, son 1000 imagenes
# validation_steps = 5000/sequence_length #Despues de cada iteracion, validamos lo anterior 1000 imagenes
#
# kernels_layer1 = 32
# kernels_layer2 = 64     #Numero de filtros que vamos a aplicar en cada convolucion
# kernels_layer3 = 128
#
# size_kernel1 = (4,4)
# size_kernel2 = (3,3)
# size_kernel3 = (2,2)   #Tamaños de los filtros
#
# size_pool = (2,2)  #Tamaño del filtro en max pooling
# clases = 5  #por el momento 5 clase de palabras
#
# lr = 0.0005  #ajustes de la red neuronal para acercarse a una solucion optima

# from tensorflow.keras.preprocessing.image import ImageDataGenerator #Nos ayuda a preprocesar las imagenes que le entreguemos al modelo
# from tensorflow.keras.models import Sequential  #Nos permite hacer redes neuronales secuenciales
# from tensorflow.keras.layers import Convolution2D, MaxPooling2D  #Capas para hacer las convoluciones
# from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
# import tensorflow.keras.optimizers