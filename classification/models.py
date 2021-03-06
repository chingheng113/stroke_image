from keras.engine import Model
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Input, Activation, BatchNormalization, Dropout, Convolution3D
import keras.optimizers
from keras.regularizers import l2


def get_VoxCNN(config):
    model = Sequential()
    # 1st Volumetric Convolutional block
    model.add(Convolution3D(8, (3, 3, 3), activation='relu', padding='same', input_shape=config['input_shape']))
    model.add(Convolution3D(8, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 2nd Volumetric Convolutional block
    model.add(Convolution3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 3rd Volumetric Convolutional block
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 4th Volumetric Convolutional block
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    # 1th fully connected layer with batchnorm and dropout for regularization
    model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.7))
    # 2th fully connected layer
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # Output with softmax nonlinearity for classification
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.rmsprop(lr=1e-5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def get_simple_VoxCNN(config):
    # VoxCNN
    input_lay = Input(shape=config['input_shape'])
    conv_1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.005))(input_lay)
    act_1 = Activation('relu')(conv_1)
    conv_2 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.005))(act_1)
    act_2 = Activation('relu')(conv_2)
    maxp_2 = MaxPooling3D(pool_size=(2, 2, 2))(act_2)
    conv_3 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.005))(maxp_2)
    act_3 = Activation('relu')(conv_3)
    conv_4 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.005))(act_3)
    act_4 = Activation('relu')(conv_4)
    maxp_4 = MaxPooling3D(pool_size=(2, 2, 2))(act_4)
    flat = Flatten()(maxp_4)
    den_5 = Dense(units=128, kernel_regularizer=l2(0.005))(flat)
    act_5 = Activation('relu')(den_5)
    # nor_5 = BatchNormalization()(act_5)
    drop_5 = Dropout(0.5)(act_5)
    den_6 = Dense(units=64, kernel_regularizer=l2(0.005))(drop_5)
    act_6 = Activation('relu')(den_6)
    # nor_6 = BatchNormalization()(act_6)
    drop_6 = Dropout(0.5)(act_6)
    output = Dense(units=config['n_classes'], activation='softmax')(drop_6)
    model = Model(inputs=input_lay, outputs=output, name='simple_VoxCNN')
    opt = keras.optimizers.adam(lr=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def get_simple_AlexNet(config):
    # AlexNet
    # con dimension = floor(((n-f+2p)/s)+1)
    # pooling dimension  = floor(((n-f)/s)+1)

    input_lay = Input(shape=config['input_shape'])
    conv_1 = Conv3D(filters=32, kernel_size=(11, 11, 11), strides=(4, 4, 4), padding='same')(input_lay)
    # nor_1 = BatchNormalization()(conv_1)
    act_1 = Activation('relu')(conv_1)
    maxp_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(act_1)

    conv_2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(maxp_1)
    # nor_2 = BatchNormalization()(conv_2)
    act_2 = Activation('relu')(conv_2)
    maxp_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(act_2)

    flat_6 = Flatten()(maxp_2)
    den_6 = Dense(units=512)(flat_6)
    # nor_6 = BatchNormalization()(den_6)
    act_6 = Activation('relu')(den_6)
    drop_6 = Dropout(0.6)(act_6)

    den_7 = Dense(units=256)(drop_6)
    # nor_7 = BatchNormalization()(den_7)
    act_7 = Activation('relu')(den_7)
    drop_7 = Dropout(0.6)(act_7)

    den_8 = Dense(units=128)(drop_7)
    # nor_8 = BatchNormalization()(den_8)
    act_8 = Activation('relu')(den_8)
    drop_8 = Dropout(0.6)(act_8)

    output = Dense(units=config['n_classes'], activation='softmax')(drop_8)
    model = Model(inputs=input_lay, outputs=output, name='simple_AlexNet')

    opt = keras.optimizers.adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model