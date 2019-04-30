from keras.engine import Model
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Input, Activation, BatchNormalization, Dropout
import keras.optimizers


def get_AlexNet(config):
    # AlexNet
    # con dimension = floor(((n-f+2p)/s)+1)
    # pooling dimension  = floor(((n-f)/s)+1)

    input_lay = Input(shape=config['input_shape'])
    conv_1 = Conv3D(filters=96, kernel_size=(11, 11, 11), strides=(4, 4, 4), padding='same')(input_lay)
    nor_1 = BatchNormalization()(conv_1)
    act_1 = Activation('relu')(nor_1)
    maxp_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(act_1)

    conv_2 = Conv3D(filters=256, kernel_size=(11, 11, 11), strides=(1, 1, 1), padding='same')(maxp_1)
    nor_2 = BatchNormalization()(conv_2)
    act_2 = Activation('relu')(nor_2)
    maxp_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(act_2)

    conv_3 = Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(maxp_2)
    nor_3 = BatchNormalization()(conv_3)
    act_3 = Activation('relu')(nor_3)

    conv_4 = Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act_3)
    nor_4 = BatchNormalization()(conv_4)
    act_4 = Activation('relu')(nor_4)

    conv_5 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act_4)
    nor_5 = BatchNormalization()(conv_5)
    act_5 = Activation('relu')(nor_5)
    maxp_5 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(act_5)

    flat_6 = Flatten()(maxp_5)
    den_6 = Dense(units=4096)(flat_6)
    nor_6 = BatchNormalization()(den_6)
    act_6 = Activation('relu')(nor_6)
    drop_6 = Dropout(0.5)(act_6)

    den_7 = Dense(units=2048)(drop_6)
    nor_7 = BatchNormalization()(den_7)
    act_7 = Activation('relu')(nor_7)
    drop_7 = Dropout(0.4)(act_7)

    den_8 = Dense(units=1000)(drop_7)
    nor_8 = BatchNormalization()(den_8)
    act_8 = Activation('relu')(nor_8)
    drop_8 = Dropout(0.4)(act_8)

    output = Dense(units=config['n_classes'], activation='softmax')(drop_8)
    model = Model(inputs=input_lay, outputs=output, name='AlexNet')

    opt = keras.optimizers.adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def get_simple_AlexNet(config):
    # AlexNet
    # con dimension = floor(((n-f+2p)/s)+1)
    # pooling dimension  = floor(((n-f)/s)+1)

    input_lay = Input(shape=config['input_shape'])
    conv_1 = Conv3D(filters=96, kernel_size=(11, 11, 11), strides=(4, 4, 4), padding='same')(input_lay)
    nor_1 = BatchNormalization()(conv_1)
    act_1 = Activation('relu')(nor_1)
    maxp_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(act_1)

    conv_2 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(maxp_1)
    nor_2 = BatchNormalization()(conv_2)
    act_2 = Activation('relu')(nor_2)
    maxp_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(act_2)

    flat_6 = Flatten()(maxp_2)
    den_6 = Dense(units=1024)(flat_6)
    nor_6 = BatchNormalization()(den_6)
    act_6 = Activation('relu')(nor_6)
    drop_6 = Dropout(0.6)(act_6)

    den_7 = Dense(units=512)(drop_6)
    nor_7 = BatchNormalization()(den_7)
    act_7 = Activation('relu')(nor_7)
    drop_7 = Dropout(0.6)(act_7)

    den_8 = Dense(units=256)(drop_7)
    nor_8 = BatchNormalization()(den_8)
    act_8 = Activation('relu')(nor_8)
    drop_8 = Dropout(0.6)(act_8)

    output = Dense(units=config['n_classes'], activation='softmax')(drop_8)
    model = Model(inputs=input_lay, outputs=output, name='simple_AlexNet')

    opt = keras.optimizers.adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model