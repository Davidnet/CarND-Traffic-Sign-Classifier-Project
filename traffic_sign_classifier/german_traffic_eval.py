import pickle
import dicto as do

import numpy as np
import tensorflow as tf

from fire import Fire

from german_traffic_densenet import model_fn
from german_traffic_dataset import input_fn

@do.fire_options("configs.yml")
def main(train_params, model_dir):

    params = do.Dicto(train_params)

    with open("/data/valid.p", "rb") as fd:
        valid = pickle.load(fd)

    
    validate_input_fn = lambda : input_fn(valid['features'].astype(np.float32), valid['labels'].astype(np.int32), params, training=False)

    classifier = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    print(classifier.evaluate(input_fn=validate_input_fn))

if __name__ == "__main__":
    Fire(main)

