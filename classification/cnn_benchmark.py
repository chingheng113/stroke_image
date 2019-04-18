import sys, os
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
from data import data_util
from classification import generator
from keras.engine import Model
import numpy as np
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Input, Activation, BatchNormalization, Dropout
from keras import backend
import keras.optimizers

backend.set_image_data_format('channels_first')
config = dict()
config['which_machine'] = 'mri'
# config['image_shape'] = (128, 128, 28) # CHANNEL, WIDTH, HEIGHT, DEPTH
config['image_shape'] = (256, 256, 20) # CHANNEL, WIDTH, HEIGHT, DEPTH
config['n_classes'] = 2
if config['which_machine'] == 'ct':
    config['all_sequences'] = ['ct']
else:
    # MRI
    config["all_sequences"] = ['dwi', 'flair']
config['n_channels'] = len(config["all_sequences"])
config['input_shape'] = tuple([config['n_channels']] + list(config['image_shape']))
config['batch_size'] = 10
config["n_epochs"] = 300

if __name__ == '__main__':
    read_file_path = data_util.write_data_to_file(config)
    data_file = data_util.open_data_file(read_file_path)
    train_generator, validation_generator, n_train_steps, n_validation_steps = generator.get_training_and_validation_generators(data_file, config)

    # AlexNet
    # con dimension = floor(((n-f+2p)/s)+1)
    # pooling dimension  = floor(((n-f)/s)+1)

    input_lay = Input(shape=config['input_shape'])
    conv_1 = Conv3D(filters=96, kernel_size=(11, 11, 3), strides=(4, 4, 4), padding='same')(input_lay)
    nor_1 = BatchNormalization()(conv_1)
    act_1 = Activation('relu')(nor_1)
    maxp_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(act_1)

    conv_2 = Conv3D(filters=256, kernel_size=(11, 11, 3), strides=(1, 1, 1), padding='same')(maxp_1)
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
    drop_6 = Dropout(0.4)(act_6)

    den_7 = Dense(units=4096)(drop_6)
    nor_7 = BatchNormalization()(den_7)
    act_7 = Activation('relu')(nor_7)
    drop_7 = Dropout(0.4)(act_7)

    den_8 = Dense(units=1000)(drop_7)
    nor_8 = BatchNormalization()(den_8)
    act_8 = Activation('relu')(nor_8)
    drop_8 = Dropout(0.4)(act_8)

    output = Dense(units=config['n_classes'], activation='softmax')(drop_8)
    model = Model(inputs=input_lay, outputs=output)

    opt = keras.optimizers.adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    # no callback yet
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=n_train_steps,
                                  epochs=config["n_epochs"],
                                  validation_data=validation_generator,
                                  validation_steps=n_validation_steps,
                                  verbose=1
                                  )
    print('Training done..')

    X_data = data_util.open_data_file(read_file_path).root.data[:]
    y_data_o = data_util.open_data_file(read_file_path).root.label[:]
    y_data = keras.utils.to_categorical(y_data_o, num_classes=config['n_classes'])
    # score = model.predict(X_data)
    loss, acc = model.evaluate(X_data, y_data, verbose=0)
    print(acc)
