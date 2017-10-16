from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from keras.models import Model, Sequential
from keras.datasets import mnist
import numpy as np
from cleverhans.attacks import FastGradientMethod
import tensorflow as tf
import keras
from cleverhans.utils import cnn_model
from tensorflow.python.platform import flags
from cleverhans.utils_tf import model_train, model_eval
from keras.utils import np_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

sess = tf.Session()
keras.backend.set_session(sess)

input_img = Input(shape=(1, 28, 28))
y = tf.placeholder(tf.float32, shape=(None, 10))

encoder = Sequential()
x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x2 = MaxPooling2D((2, 2), padding='same')(x1)
x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
x4 = MaxPooling2D((2, 2), padding='same')(x3)
x5 = Conv2D(8, (3, 3), activation='relu', padding='same')(x4)
encoded = MaxPooling2D((2, 2), padding='same')(x5)
ae = Model(input_img, encoded)
encoder.add(ae)
encoder.add(Flatten())
encoder.add(Dense(10))
encoder.summary()

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x6 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(8, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(16, (3, 3), activation='relu')(x9)
x11 = UpSampling2D((2, 2))(x10)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x11)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))  # adapt this if using `channels_first` image data format

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#x_train, y_train, x_test, y_test = data_mnist()

assert y_train.shape[1] == 10.
label_smooth = .1
y_train = y_train.clip(label_smooth / 9., 1. - label_smooth)

autoencoder.fit(x_train, x_train,
                epochs=25,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

model = cnn_model()
predictions_2 = model(input_img)
fgsm = FastGradientMethod(model, sess=sess)  # new object
fgsm_params = {'eps': 0.3}  # parameters
adv_x = fgsm.generate(input_img, **fgsm_params)
predictions_2_adv = encoder(adv_x)

def evaluate_2():
    # Accuracy of adversarially trained model on legitimate test inputs
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = model_eval(sess, input_img, y, predictions_2, x_test, y_test,
                          args=eval_params)
    print('Test accuracy on legitimate test examples: %0.4f' % accuracy)

    # Accuracy of the adversarially trained model on adversarial examples
    accuracy_adv = model_eval(sess, input_img, y, predictions_2_adv, x_test,
                              y_test, args=eval_params)
    print('Test accuracy on adversarial examples: %0.4f' % accuracy_adv)

train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }

# Perform and evaluate adversarial training
model_train(sess, input_img, y, predictions_2, x_train, y_train,
            predictions_adv=predictions_2_adv, evaluate=evaluate_2,
            args=train_params)




