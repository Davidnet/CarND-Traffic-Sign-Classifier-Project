import pickle
import numpy as np
import tensorflow as tf 

from fire import Fire
import dicto as do
# from dicto import fire_options

from german_traffic_dataset import input_fn 
from german_traffic_densenet import DenseNet, model_fn

@do.fire_options("configs.yml")
def main(train_params, model_dir):
    params = do.Dicto(train_params)
    # params = do.Dicto(
    #     buffer_size = 34799,
    #     batch_size = 16,
    #     epochs = 400
    # )

    with open("/data/train.p", "rb") as fd:
        train = pickle.load(fd)
        
    train_input_fn = lambda : input_fn(train['features'], train['labels'].astype(np.int32), params, training=True)

    with open("/data/test.p", "rb") as fd:
        test = pickle.load(fd)

    eval_input_fn = lambda : input_fn(test['features'], test['labels'].astype(np.int32), params, training=False)
   
    classifier = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == "__main__":
    Fire(main)

