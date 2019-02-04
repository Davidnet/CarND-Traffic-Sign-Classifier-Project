# Excellent read: https://cs230-stanford.github.io/tensorflow-input-data.html
import pickle
import dicto as do

import numpy as np
import tensorflow as tf

def parse_function(image, label):

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(label, tf.int32)

    return image, label


def train_preprocess(image, label):

    if label in [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]:
            image = tf.image.random_flip_left_right(image)

    if label in [1, 5, 12, 15, 17]:
        image = tf.image.random_flip_up_down(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label



def input_fn(images, labels, params, training):
    
    ds = tf.data.Dataset.from_tensor_slices((
        images,
        labels,
    ))

    ds = ds.map(parse_function, num_parallel_calls=4)
    
    if training:
        ds = ds.map(train_preprocess, num_parallel_calls=4)
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat( 
        buffer_size = params.buffer_size,
        count = params.epochs,
        ))
        
    #ds  ds.map(lambda x, y: ({"images": x}, y))

    # ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)))
        
    ds = ds.batch(params.batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=2)

    
    return ds

# ds = input_fn(train['features'], train['labels'], params, training=True)



if __name__ == "__main__":
    with open("/data/train.p", "rb") as fd:
        train = pickle.load(fd)

    with open("/data/test.p", "rb") as fd:
        test = pickle.load(fd)

    params = do.Dicto(
        buffer_size = 34799,
        batch_size = 16,
        epochs = 400
    )
    train_input_fn = lambda : input_fn(train['features'].astype(np.float32), train['labels'].astype(np.int32), params, training=True)
    eval_input_fn = lambda : input_fn(test['features'].astype(np.float32), test['labels'].astype(np.int32), params, training=False)
