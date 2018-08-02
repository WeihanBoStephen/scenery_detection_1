#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import numpy as np
import os
#%%
img_height = 100
img_width = 100
img_depth = 3
image_bytes = img_width * img_depth * img_height
# %%

IMAGE_SIZE = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200


# %%
def get_inputs(data_dir, is_train, batch_size, shuffle):  # get batch of data and label
    
    filenames = [os.path.join(data_dir, 'testing.tfrecords')]
    print('Testing stage.')
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    
    print(filenames)
    filename_queue = tf.train.string_input_producer(filenames)

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    image, label = read_data(filename_queue, is_train)
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(image,
                                                           height, width)
    label = tf.cast(label, tf.int3t)
    image = image_std(resized_image)

    image.set_shape([height, width, 3])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    return generator_batch(image, label, min_queue_examples, batch_size, shuffle)


# %%



def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'training.tfrecords')]

    filename_queue = tf.train.string_input_producer(filenames)

    image, label = read_data(filename_queue, 'train')
    label = tf.cast(label, tf.int32)
    image = image_arg(image)
    image = image_std(image)
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    return generator_batch(image, label, min_queue_examples, batch_size, shuffle=True)

    # %%


def read_data(filename_queue, is_train):  # make file queue to read data

    reader = tf.TFRecordReader()
    key2, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    if is_train == 'train':
        image = tf.reshape(image, [img_width, img_height, img_depth])
    elif is_train == 'test':
        image = tf.reshape(image, [img_width, img_height, img_depth])
        print('read testing data.')
    else:
        image = tf.reshape(image, [100, 50, 3])
        print('read test eyes or lips images.')
    image = tf.cast(image, tf.float32)
    label = tf.cast(img_features['label'], tf.int32)

    return image, label


    # %%


def image_arg(image):
    # data argumentation
    image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])  # random crop the image
    image = tf.image.random_brightness(image, max_delta=63)  # random changing the brightness
    image = tf.image.random_flip_left_right(image)  # random flip the image
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)  # random increase the contrast
    return image


# %%

def image_std(image):  # standardize the whole image
    image = tf.image.per_image_standardization(image)
    return image


# %%

def generator_batch(image, label, min_queue_examples, batch_size, shuffle):
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=16,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=16,
            capacity=min_queue_examples + 3 * batch_size)

    return image_batch, tf.reshape(label_batch, [batch_size])

























