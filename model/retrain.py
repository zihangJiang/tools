# -*- coding: utf-8 -*-

from keras.models import Model,load_model
from keras.preprocessing import image
import numpy as np

model=load_model('pss0.h5')

train_datagen = image.ImageDataGenerator(rescale=1./255,zoom_range=0.2)


train_generator = train_datagen.flow_from_directory(
        'dataset40',
        target_size=(100,200),
        color_mode='grayscale',
        batch_size=17)


model.fit_generator(
        train_generator,
        steps_per_epoch=200,
        epochs=3)
model.save('pss0.h5')
