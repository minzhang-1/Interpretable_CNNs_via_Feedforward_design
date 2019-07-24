# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import DeepFool
from cleverhans.loss import CrossEntropy
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper
from keras.datasets import cifar10
from cleverhans.utils_tf import train, model_eval
import keras
from keras import backend
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import os
from lenet_model import cifar_model
from cifar_ff_model import cifar_ff_model
from cleverhans.utils import pair_visual, grid_visual
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

FLAGS = flags.FLAGS


def cifar_tutorial(train_start=0, train_end=50000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001, train_dir="train_dir",
                   filename="cifar.ckpt", load_model=True,
                   testing=False, label_smoothing=0.1, method='FGSM'):
    """
    Cifar  tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param train_dir: Directory storing the saved model
    :param filename: Filename to save model under
    :param load_model: True for load, False for not load
    :param testing: if true, test error is calculated
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :return: an AccuracyReport object
    """
    keras.layers.core.K.set_learning_phase(0)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    # viz_enabled=True
    targeted=False

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    num_classes=10
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    y_train_ori = y_train
    y_test_ori = y_test
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print ('y_train.shape',y_train.shape)


    # Obtain Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    print('img_rows: {}, img_cols: {}, nchannels: {}'.format(img_rows, img_cols, nchannels))
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    if train_dir=='cifar_ff_model':
        model=cifar_ff_model()
    elif train_dir=='cifar_BP_model':
        model = cifar_model(img_rows=img_rows, img_cols=img_cols,
                      channels=nchannels, nb_filters=64,
                      nb_classes=nb_classes)
    preds = model(x)
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
        report.clean_train_clean_eval = acc
#        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': train_dir,
        'filename': filename
    }

    rng = np.random.RandomState([2017, 8, 30])
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    ckpt = tf.train.get_checkpoint_state(train_dir)
    print(train_dir, ckpt)
    ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
    wrap = KerasModelWrapper(model)

    if load_model and ckpt_path:
        saver = tf.train.Saver()
        print(ckpt_path)
        saver.restore(sess, ckpt_path)
        print("Model loaded from: {}".format(ckpt_path))
        evaluate()
    else:
        print("Model was not loaded, training from scratch.")
        loss = CrossEntropy(wrap, smoothing=label_smoothing)
        train(sess, loss, x, y, x_train, y_train, evaluate=evaluate,
              args=train_params, save=True, rng=rng)
        print('Training done!')

    # Calculate training error
    print('testing param:', testing)
    if testing:
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, x_train, y_train, args=eval_params)
        report.train_clean_train_clean_eval = acc

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    # fgsm = FastGradientMethod(wrap, sess=sess)
    if method=='FGSM':
        clw=FastGradientMethod(wrap, sess=sess)
    elif method=='BIM':
        clw=BasicIterativeMethod(wrap, sess=sess)
    elif method=='DeepFool':
        clw=DeepFool(wrap, sess=sess)
    else:
        raise NotImplementedError
    print('method chosen: ', method)
    clw_params = {}
    adv_x = clw.generate(x, **clw_params)
    with sess.as_default():
        feed_dict={x:x_test[:1000], y:y_test[:1000]}
        store_data=adv_x.eval(feed_dict=feed_dict)
        print('store_data: {}'.format(store_data.shape))
        save_name='{}/cifar_{}_data.pkl'.format(train_dir, method)
        with open(save_name,'wb') as fw:
            pickle.dump(store_data, fw, protocol=2)
            print('data stored in {}'.format(save_name))


    # Consider the attack to be constant
    adv_x = tf.stop_gradient(adv_x)

    preds_adv = model(adv_x)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc

    # Calculating train error
    if testing:
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, x_train,
                         y_train, args=eval_par)
        report.train_clean_train_adv_eval = acc


    return report


def main(argv=None):
    cifar_tutorial(nb_epochs=FLAGS.nb_epochs,
                   batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   train_dir=FLAGS.train_dir,
                   filename=FLAGS.filename,
                   load_model=FLAGS.load_model,
                   method=FLAGS.method)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 40, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('train_dir', 'cifar_ff_model',
                        'Directory where to save model.')
    flags.DEFINE_string('filename', 'FF_init_model.ckpt', 'Checkpoint filename.')
    flags.DEFINE_boolean('load_model', True, 'Load saved model or train.')
    flags.DEFINE_string('method', 'FGSM', 'Adversarial attack method')
    tf.app.run()