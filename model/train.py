# -*- coding: utf-8 -*-
from keras.models import Model,load_model
from keras.layers import Dense,MaxPooling2D,Flatten, Input,Conv2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import numpy as np

classes=34
input_tensor = Input(shape=(100,200,1))
img_input = input_tensor
# Block 1
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
#x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
#x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
#x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
#x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
#x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

x = Flatten(name='flatten')(x)
#x = Dense(512, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(classes, activation='softmax', name='predictions')(x)

model = Model(img_input, x, name='pss')

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
train_datagen = image.ImageDataGenerator(rescale=1./255,zoom_range=0.2)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset40',
        target_size=(100,200),
        color_mode='grayscale',
        batch_size=17)


model.fit_generator(
        train_generator,
        steps_per_epoch=200,
        epochs=10)
model.save('ps.h5')
