import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import math
import json
import sys

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.layers import Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model, Sequential
from keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import *
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils import dataset as dataset
import argparse

import time
from datetime import timedelta


def build_dataset(data_directory, img_width):
    X, y, tags = dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    label = np_utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes


def build_model(SHAPE, nb_classes, bn_axis, seed=None):
    # We can't use ResNet50 directly, as it might cause a negative dimension
    # error.
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    # Step 1
    x = Conv2D(32, 3, 3, kernel_initializer='glorot_uniform',
               padding='same', activation='relu')(input_layer)
    # Step 2 - Pooling
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Step 1
    x = Conv2D(48, 3, 3, kernel_initializer='glorot_uniform', padding='same',
               activation='relu')(x)
    # Step 2 - Pooling
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    # Step 1
    x = Conv2D(64, 3, 3, kernel_initializer='glorot_uniform', padding='same',
               activation='relu')(x)
    # Step 2 - Pooling
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Step 1
    x = Conv2D(96, 3, 3, kernel_initializer='glorot_uniform', padding='same',
               activation='relu')(x)
    # Step 2 - Pooling
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    # Step 3 - Flattening
    x = Flatten()(x)

    # Step 4 - Full connection

    x = Dense(256, activation='relu')(x)
    # Dropout
    x = Dropout(0.5)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(input_layer, x)

    return model


def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=48)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=3)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    # parser.add_argument('-o', '--optimizer',
    #                     help='choose the optimizer (rmsprop, adagrad, adadelta, adam, adamax, nadam)', default="adam")
    parser.add_argument('-o', '--output',
                        help='a result file', type=str, default="hasilnya.txt")
    args = parser.parse_args()
    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    epochs = args.epochs
    batch_size = args.batch_size
    SHAPE = (img_width, img_height, channel)
    bn_axis = 3 if K.image_data_format() == 'tf' else 1

    data_directory = args.input
    period_name = data_directory.split('/')
    seq_len = '20'#args.input.split('_')[-2]

    print("loading dataset")
    X_train, Y_train, nb_classes = build_dataset(
        "{}/train".format(data_directory), args.dimension)
    X_test, Y_test, nb_classes = build_dataset(
        "{}/test".format(data_directory), args.dimension)
    print("number of classes : {}".format(nb_classes))

    # Load model
    # It can be used to reconstruct the model identically.
    # reconstructed_model = keras.models.load_model("my_h5_model.h5")
    # make checkpoints
    model = build_model(SHAPE, nb_classes, bn_axis)
    #path_model = 'checkpoints2/DCNN_128_20_50_Taiwan50/model-0050.h5'
    #model = keras.models.load_model(path_model)

    adam = keras.optimizers.Adam(learning_rate=1.0e-4)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    path_checkpoint = 'checkpoints2/DCNN_' + f'{batch_size}_{seq_len}' + '_' + str(args.dimension) + '_' + period_name[1] + "/model-epochs-{epoch:04d}.h5"
    path_dir_checkpoint = os.path.dirname(path_checkpoint)
    # Create a callback that saves the model's structure by five epochs
    callback_cp = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint, verbose=1, period=5)
    callback_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8, restore_best_weights=True, mode='min', verbose=1)
    callback_csv = tf.keras.callbacks.CSVLogger(f'training_log2/DCNN_{batch_size}_{seq_len}_{args.dimension}_{period_name[1]}.csv')

    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback_cp, callback_stop, callback_csv])

    # Save Model or creates a HDF5 file
    model.save('saved_model2/{}epochs_{}batch_cnn_model_{}.h5'.format(
        epochs, batch_size, data_directory.split("/")[1]), overwrite=True)
    # del model  # deletes the existing model
    predicted = model.predict(X_test)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    if tp == 0:
        tp = 1
    if tn == 0:
        tn = 1
    if fp == 0:
        fp = 1
    if fn == 0:
        fn = 1
    TPR = float(tp)/(float(tp)+float(fn))
    FPR = float(fp)/(float(fp)+float(tn))
    accuracy = round((float(tp) + float(tn))/(float(tp) +
                                              float(fp) + float(fn) + float(tn)), 3)
    specitivity = round(float(tn)/(float(tn) + float(fp)), 3)
    sensitivity = round(float(tp)/(float(tp) + float(fn)), 3)
    mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
        (float(tp)+float(fp))
        * (float(tp)+float(fn))
        * (float(tn)+float(fp))
        * (float(tn)+float(fn))
    ), 3)

    f_output = open(args.output, 'a')
    f_output.write('=======\n')
    f_output.write('{}epochs_{}batch_cnn_{}\n'.format(
        epochs, batch_size, period_name[1]))
    f_output.write('TN: {}\n'.format(tn))
    f_output.write('FN: {}\n'.format(fn))
    f_output.write('TP: {}\n'.format(tp))
    f_output.write('FP: {}\n'.format(fp))
    f_output.write('TPR: {}\n'.format(TPR))
    f_output.write('FPR: {}\n'.format(FPR))
    f_output.write('accuracy: {}\n'.format(accuracy))
    f_output.write('specitivity: {}\n'.format(specitivity))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write('=======\n')
    f_output.close()
    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()
