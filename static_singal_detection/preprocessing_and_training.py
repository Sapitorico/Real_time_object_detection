#!/usr/bin/python
from jax._src.interpreters.partial_eval import Val

from static_model import CNN_Model

DATASET_DIR = "mp_dataset"
HEIGHT , WIDTH = 200, 200
SEQUENCE_LENGTH = 1
SIZE_POOL = (2,2)
CLASSES = 5

ITERATIONS = 100
STEPS = 4040 // SEQUENCE_LENGTH
VALIDATION_STEPS = 1010 // SEQUENCE_LENGTH

KERNELS_LAYER1 = 32
KERNELS_LAYER2 = 64
KERNELS_LAYER3 = 128

SIZE_KERNEL1 = (4,4)
SIZE_KERNEL2 = (3,3)
SIZE_KERNEL3 = (2,2)

LR = 0.0005

model = CNN_Model(DATASET_DIR, HEIGHT, WIDTH, SEQUENCE_LENGTH, SIZE_POOL)
train_set = model.Create_train_set()
val_set = model.Create_validation_set()
# class_names = train_set.class_names
# print(class_names)
CNN = model.Build_model(CLASSES,KERNELS_LAYER1, SIZE_KERNEL1, KERNELS_LAYER2, SIZE_KERNEL2, KERNELS_LAYER3, SIZE_KERNEL3, SIZE_POOL)
CNN = model.Fit_model(CNN, LR, train_set, val_set, STEPS, VALIDATION_STEPS, ITERATIONS)
model.Save_model(CNN)