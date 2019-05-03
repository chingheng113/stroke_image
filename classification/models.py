from keras.engine import Model
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Input, Activation, BatchNormalization, Dropout
import keras.optimizers


def get_AlexNet(config):
    # AlexNet
    # con dimension = floor(((n-f+2p)/s)+1)
    # pooling dimension  = floor(((n-f)/s)+1)
    pad = 'valid'
    input_lay = Input(shape=config['input_shape'])
    conv_1 = Conv3D(filters=96, kernel_size=(2, 11, 11), strides=(1, 4, 4), padding=pad)(input_lay)
    act_1 = Activation('tanh')(conv_1)
    nor_1 = BatchNormalization()(act_1)
    maxp_1 = MaxPooling3D(pool_size=(2, 2, 2))(nor_1)

    conv_2 = Conv3D(filters=256, kernel_size=(2, 11, 11), strides=(1, 1, 1), padding=pad)(maxp_1)
    act_2 = Activation('tanh')(conv_2)
    nor_2 = BatchNormalization()(act_2)
    maxp_2 = MaxPooling3D(pool_size=(2, 2, 2))(nor_2)

    conv_3 = Conv3D(filters=384, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding=pad)(maxp_2)
    act_3 = Activation('tanh')(conv_3)
    nor_3 = BatchNormalization()(act_3)

    conv_4 = Conv3D(filters=384, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding=pad)(nor_3)
    act_4 = Activation('tanh')(conv_4)
    nor_4 = BatchNormalization()(act_4)

    # conv_5 = Conv3D(filters=256, kernel_size=(2, 3, 3), strides=(1, 1, 1), padding=pad)(nor_4)
    # act_5 = Activation('tanh')(conv_5)
    # nor_5 = BatchNormalization()(act_5)
    # maxp_5 = MaxPooling3D(pool_size=(2, 2, 2))(nor_5)

    flat_6 = Flatten()(nor_4)
    den_6 = Dense(units=512)(flat_6)
    act_6 = Activation('tanh')(den_6)
    nor_6 = BatchNormalization()(act_6)
    drop_6 = Dropout(0.4)(nor_6)

    den_7 = Dense(units=256)(drop_6)
    act_7 = Activation('tanh')(den_7)
    nor_7 = BatchNormalization()(act_7)
    drop_7 = Dropout(0.4)(nor_7)

    den_8 = Dense(units=128)(drop_7)
    act_8 = Activation('tanh')(den_8)
    nor_8 = BatchNormalization()(act_8)
    drop_8 = Dropout(0.4)(nor_8)

    output = Dense(units=config['n_classes'], activation='softmax')(drop_8)
    model = Model(inputs=input_lay, outputs=output, name='AlexNet')

    opt = keras.optimizers.adam(lr=0.00001)
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