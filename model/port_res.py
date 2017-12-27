# -*- coding: utf-8 -*-
#from keras.applications.inception_v3 import InceptionV3
from keras.models import Model,load_model
from keras.layers import Dense,GlobalAveragePooling2D,Flatten, Input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


input_tensor = Input(shape=(200,200,1))

model = ResNet50(include_top=True,weights=None,input_tensor =input_tensor,classes=34)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
train_datagen = image.ImageDataGenerator(rescale=1./255)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset42',
        target_size=(200, 200),
        color_mode='grayscale',
        batch_size=84)
#colormode
validation_generator = test_datagen.flow_from_directory(
        'data/train',
        target_size=(200, 200),
        color_mode='grayscale',
        batch_size=84)

model.fit_generator(
        train_generator,
        steps_per_epoch=17,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=4)
model.save('weight3.h5')
#        ,validation_data=validation_generator, validation_steps=30
