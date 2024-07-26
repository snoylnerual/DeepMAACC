from keras.callbacks import EarlyStopping
from keras.datasets import fashion_mnist, mnist, cifar10, cifar100
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, BatchNormalization, Resizing, MaxPooling2D, Dropout, Input, Activation, ZeroPadding2D
#from keras_cv.models import ResNetBackbone
from keras.models import load_model, Model
from keras import Sequential, layers
from keras.applications import VGG16  # , ResNet50
from argparse import ArgumentParser
from keras.regularizers import l2
from keras.utils import np_utils
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os


class Network:
    def __init__(self, model_type, Dataset):
        self._model_type = model_type
        self._dataset_name = Dataset.get_dataset_name()
        self._model_file_name = model_type + '-' + self._dataset_name + '.h5'
        self._x_train = Dataset.get_x_train()
        self._y_train = Dataset.get_y_train()
        self._x_test = Dataset.get_x_test()
        self._y_test = Dataset.get_y_test()
        self._model = None

    def train(self):
        if self._model_type == 'fcnn':
            self._model = self.fcnn(self._x_train, self._y_train, self._x_test, self._y_test)
        elif self._model_type == 'lenet5':
            self._model = self.lenet5(self._x_train, self._y_train, self._x_test, self._y_test)
        elif self._model_type == 'resnet18':
            self._model = self.ResNet18(self._x_train, self._y_train, self._x_test, self._y_test)
        elif self._model_type == 'alexnet':
            self._model = self.alexnet(self._x_train, self._y_train, self._x_test, self._y_test)
        elif self._model_type == 'vggnet16':
            self._model = self.vggnet16(self._x_train, self._y_train, self._x_test, self._y_test)
        else:
            raise ValueError('Invalid model type')

    def retrieve_model(self):
        self._model = load_model('examples/' + self._dataset_name + '/' + self._model_file_name)

    def get_model(self):
        return self._model

    def fcnn(self, x_train, y_train, x_test, y_test):
        nb_classes = 10
        model = Sequential()

        model.add(Flatten(input_shape=(32, 32, 3)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=0)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        model.save('examples/' + self._dataset_name + '/fcnn-' + self._dataset_name + '.h5')
        print("Model saved")
        # TODO change print to logs
        return model

    def lenet5(self, x_train, y_train, x_test, y_test):
        nb_classes = 10
        model = Sequential()

        model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        model.save('examples/' + self._dataset_name + '/lenet5-' + self._dataset_name + '.h5')
        print("Model saved")
        # TODO change print to logs
        return model

    # def resnet18(self, x_train, y_train, x_test, y_test):
    #     nb_classes = 10
    #     model = Sequential()
    #
    #     model.add(Input(shape=(32, 32, 3)))
    #     model.add(Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Add())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Add())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(128, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(BatchNormalization())
    #     model.add(Add())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(256, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Add())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(BatchNormalization())
    #     model.add(Add())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(512, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Add())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(BatchNormalization())
    #     model.add(Add())
    #     model.add(Activation('relu'))
    #
    #     model.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #
    #     #model.add(Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu'))
    #     #model.add(BatchNormalization())
    #     model.add(Add())
    #     model.add(Activation('relu'))
    #
    #     model.add(AveragePooling2D())  # pool_size=(2, 2)))
    #     model.add(Flatten())
    #     model.add(Dense(nb_classes, activation='softmax'))
    #
    #
    #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    #     early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
    #     model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])
    #
    #     model.save('examples/' + self._dataset_name + '/resnet18-' + self._dataset_name + '.h5')
    #     print("Model saved")
    #     # TODO change print to logs
    #     return model

    def conv2d_bn(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=l2(weight_decay)
                       )(x)
        layer = BatchNormalization()(layer)
        return layer

    def conv2d_bn_relu(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = self.conv2d_bn(x, filters, kernel_size, weight_decay, strides)
        layer = Activation('relu')(layer)
        return layer

    def ResidualBlock(self, x, filters, kernel_size, weight_decay, downsample=True):
        if downsample:
            # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
            residual_x = self.conv2d_bn(x, filters, kernel_size=1, strides=2)
            stride = 2
        else:
            residual_x = x
            stride = 1
        residual = self.conv2d_bn_relu(x,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  weight_decay=weight_decay,
                                  strides=stride,
                                  )
        residual = self.conv2d_bn(residual,
                             filters=filters,
                             kernel_size=kernel_size,
                             weight_decay=weight_decay,
                             strides=1,
                             )
        out = layers.add([residual_x, residual])
        out = Activation('relu')(out)
        return out

    def ResNet18(self, x_train, y_train, x_test, y_test):
        #https://github.com/jerett/Keras-CIFAR10/blob/master/classifiers/ResNet.py
        nb_classes = 10
        weight_decay = 1e-4
        input = Input(shape=(32, 32, 3))
        x = input
        x = Resizing(224,224)(x)
        x = self.conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
        # # conv 2
        x = self.ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        x = self.ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        # # conv 3
        x = self.ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
        x = self.ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        # # conv 4
        x = self.ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
        x = self.ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        # # conv 5
        x = self.ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
        x = self.ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
        x = Flatten()(x)
        x = Dense(nb_classes, activation='softmax')(x)
        model = Model(input, x, name='ResNet18')

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        model.save('examples/' + self._dataset_name + '/resnet18-' + self._dataset_name + '.h5')
        print("Model saved")
        # TODO change print to logs
        return model

    def resnet18_2(self, x_train, y_train, x_test, y_test):
        nb_classes = 10
        input = Input(shape=(32,32,3))
        c1 = input
        c1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',  use_bias=False, kernel_regularizer=l2(1e-4))(c1)
        c1 = BatchNormalization()(c1)
        c1 = Activation('relu')(c1)

        # # conv 2
        residual_x = c1 #x
        c2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(c1)
        c2 = BatchNormalization()(c2)
        c2 = Activation('relu')(c2)
        c2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(c2)
        c2 = BatchNormalization()(c2)
        c2 = layers.add([residual_x, c2])
        c2 = Activation('relu')(c2)

        residual_x = c2
        c22 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(c2)
        c22 = BatchNormalization()(c22)
        c22 = Activation('relu')(c22)
        c22 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(c22)
        c22 = BatchNormalization()(c22)
        c22 = layers.add([residual_x, c22])
        c22 = Activation('relu')(c22)

        # # conv 3
        c3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(c22)
        x = BatchNormalization()(c3)
        residual_x = c3
        c31 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(c3)
        c31 = BatchNormalization()(c31)
        c31 = Activation('relu')(c31)
        c31 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(c31)
        c31 = BatchNormalization()(c31)
        c31 = layers.add([residual_x, c31])
        x = Activation('relu')(c31)

        residual_x = x
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = layers.add([residual_x, x])
        x = Activation('relu')(x)

        # # conv 4

        x = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        residual_x = x
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = layers.add([residual_x, x])
        x = Activation('relu')(x)

        residual_x = x
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = layers.add([residual_x, x])
        x = Activation('relu')(x)

        # # conv 5

        x = Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        residual_x = x
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = layers.add([residual_x, x])
        x = Activation('relu')(x)

        residual_x = x
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = layers.add([residual_x, x])
        x = Activation('relu')(x)

        x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
        x = Flatten()(x)
        x = Dense(nb_classes, activation='softmax')(x)
        model = Model(input, x, name='ResNet18')

        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #
        # early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        # model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])
        #
        # model.save('examples/' + self._dataset_name + '/resnet18-' + self._dataset_name + '.h5')
        # print("Model saved")

        return model

    # def resnet18_2(self, x_train, y_train, x_test, y_test):
    #     nb_classes = 10
    #     model = ResNetBackbone.from_preset('resnet18')
    #     early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
    #     model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])
    #
    #     model.save('examples/' + self._dataset_name + '/resnet18-' + self._dataset_name + '.h5')
    #     print("Model saved")
    #     return model


    def alexnet(self, x_train, y_train, x_test, y_test):
        # from https://medium.datadriveninvestor.com/alexnet-implementation-using-keras-7c10d1bb6715
        # Instantiate an empty model
        nb_classes = 10
        model = Sequential()
        model.add(Input(shape=(32, 32, 3)))
        model.add(ZeroPadding2D((5, 5)))
        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(32, 32, 3), kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='valid'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='valid'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(10))
        model.add(Activation('softmax'))


        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ["accuracy"])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        model.save('examples/' + self._dataset_name + '/alexnet-' + self._dataset_name + '.h5')
        print("Model saved")
        return model

    def vggnet16(self, x_train, y_train, x_test, y_test):
        nb_classes = 10

        model = Sequential()

        model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=nb_classes, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])

        model.save('examples/' + self._dataset_name + '/vggnet16-' + self._dataset_name + '.h5')
        print("Model saved")
        return model

class Dataset:
# mnist, fmnist, kmnist, emnist, cifar10, cifar100

    def __init__(self, dataset_name):
        # mnist/data/mnist_train_inputs.npy
        # mnist/data/mnist_train_outputs.npy
        # mnist/data/mnist_test_inputs.npy
        # mnist/data/mnist_test_outputs.npy
        self._dataset_name = dataset_name
        if self.check_file(dataset_name):
            print("Files exist")
            # TODO change print to logs
            self._x_train = np.load('examples/' + dataset_name + '/data/' + dataset_name + '_train_inputs.npy')
            self._y_train = np.load('examples/' + dataset_name + '/data/' + dataset_name + '_train_outputs.npy')
            self._x_test = np.load('examples/' + dataset_name + '/data/' + dataset_name + '_test_inputs.npy')
            self._y_test = np.load('examples/' + dataset_name + '/data/' + dataset_name + '_test_outputs.npy')
        else:
            print("Files don't exist")
            # TODO change print to logs
            self._x_train, self._y_train, self._x_test, self._y_test = self.dataset(dataset_name)

    def get_dataset_name(self):
        return self._dataset_name
    def get_x_train(self):
        return self._x_train
    def get_y_train(self):
        return self._y_train
    def get_x_test(self):
        return self._x_test
    def get_y_test(self):
        return self._y_test

    def dataset(self,dataset_name):
        if dataset_name == 'mnist':
            print("Mnist data retrieval")
            # TODO change print to logs
            nb_classes = 10
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
            x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
            x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = np_utils.to_categorical(y_train, nb_classes), np_utils.to_categorical(y_test, nb_classes)
        elif dataset_name == 'fmnist':
            print("Fmnist data retrieval")
            # TODO change print to logs
            nb_classes = 10
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
            x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
            x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = np_utils.to_categorical(y_train, nb_classes), np_utils.to_categorical(y_test, nb_classes)
        elif dataset_name == 'kmnist':
            nb_classes = 10
            x_train = np.load('examples/kmnist/data/kmnist_train_inputs.npz')['arr_0']
            y_train = np.load('examples/kmnist/data/kmnist_train_outputs.npz')['arr_0']
            x_test = np.load('examples/kmnist/data/kmnist_test_inputs.npz')['arr_0']
            y_test = np.load('examples/kmnist/data/kmnist_test_outputs.npz')['arr_0']
            x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
            x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
            x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = np_utils.to_categorical(y_train, nb_classes), np_utils.to_categorical(y_test, nb_classes)
        elif dataset_name == 'emnist':
            nb_classes = 27
            data = sio.loadmat('examples/emnist/data/emnist-letters')['dataset']
            x_train = data['train'][0, 0]['images'][0, 0]
            y_train = data['train'][0, 0]['labels'][0, 0]
            x_test = data['test'][0, 0]['images'][0, 0]
            y_test = data['test'][0, 0]['labels'][0, 0]
            x_train = x_train.reshape(124800,28,28)
            x_test = x_test.reshape(20800,28,28)
            x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
            x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
            x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)

        elif dataset_name == 'cifar10':
            print("Cifar10 data retrieval")
            nb_classes = 10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = np_utils.to_categorical(y_train, nb_classes), np_utils.to_categorical(y_test, nb_classes)

        elif dataset_name == 'cifar100':
            print("Cifar100 data retrieval")
            nb_classes = 100
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
            x_train, x_test = x_train / 255., x_test / 255.
            y_train, y_test = np_utils.to_categorical(y_train, nb_classes), np_utils.to_categorical(y_test, nb_classes)
        else:
            x_train = y_train = x_test = y_test = np.array([])

        np.save('examples/' + dataset_name + '/data/' + dataset_name + '_train_inputs', x_train)
        np.save('examples/' + dataset_name + '/data/' + dataset_name + '_train_outputs', y_train)
        np.save('examples/' + dataset_name + '/data/' + dataset_name + '_test_inputs', x_test)
        np.save('examples/' + dataset_name + '/data/' + dataset_name + '_test_outputs', y_test)
        return x_train, y_train, x_test, y_test
        pass

    def check_file(self, file_name):
        tx = 'examples/' + file_name + '/data/' + file_name + '_train_inputs.npy'
        ty = 'examples/' + file_name + '/data/' + file_name + '_train_outputs.npy'
        ttx = 'examples/' + file_name + '/data/' + file_name + '_test_inputs.npy'
        tty = 'examples/' + file_name + '/data/' + file_name + '_test_outputs.npy'
        return os.path.isfile(tx) and os.path.isfile(ty) and os.path.isfile(ttx) and os.path.isfile(tty)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        dest='model_type',
                        help='Potential models: \'fcnn\', \'lenet5\', \'resnet18\', \'alexnet\', \'vggnet16\'',
                        required=True)
    parser.add_argument('-d',
                        '--dataset',
                        dest='dataset',
                        help='Potential values: \'mnist\', \'fmnist\', \'kmnist\', \'emnist\', \'cifar10\', \'cifar100\'',
                        required=True)
    args = parser.parse_args()

    data = Dataset(args.dataset)
    network = Network(args.model_type, data)
    network.train()



