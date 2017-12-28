"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os

import keras
import keras.preprocessing.image

from keras_retinanet.models import ResNet50RetinaNet
from keras_retinanet.preprocessing.pascal_voc import CostomIterator
import keras_retinanet

import tensorflow as tf

model_path='ps.h5'
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_model(weights='imagenet'):
    image = keras.layers.Input((None, None, 3))
    return ResNet50RetinaNet(image, num_classes=31, weights=weights)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for Pascal VOC object detection.')
    parser.add_argument('--voc_path', help='Path to Pascal VOC directory (ie. /tmp/VOCdevkit/VOC2007).',default='./dataset')
    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).', default='imagenet')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).',default='0')

    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    
    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())
    
    # create the model
    print('Creating model, this may take a second...')
    model = create_model(weights=args.weights)

    model.load_weights(model_path, by_name=True)
    # compile model (note: set loss to None since loss is added inside layer)
    model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(sigma=3),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    # print model summary
    print(model.summary())

    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
    )

    # create a generator for training data
    train_generator = CostomIterator(
        args.voc_path,
        'trainval',
        train_image_data_generator,
        image_min_side=500,
        image_max_side=600
    )



    # start training
    batch_size = 1
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator.image_names) // batch_size,
        epochs=1,
        verbose=1,
    )

    # store final result too
    model.save('ps.h5')
