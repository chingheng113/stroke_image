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
config['which_machine'] = 'ct'
config['image_shape'] = (128, 128, 28) # CHANNEL, WIDTH, HEIGHT, DEPTH
config['n_classes'] = 2
if config['which_machine'] == 'ct':
    config['all_modalities'] = ['ct']
else:
    config["all_modalities"] = ['t1', 't1ce', 'flair', 't2']
config['n_channels'] = len(config["all_modalities"])
config['input_shape'] = tuple([config['n_channels']] + list(config['image_shape']))
config['batch_size'] = 10
config["n_epochs"] = 500

if __name__ == '__main__':
    read_file_path = data_util.write_data_to_file(config)
    data_file = data_util.open_data_file(read_file_path)
    train_generator, validation_generator, n_train_steps, n_validation_steps = generator.get_training_and_validation_generators(data_file, config)

    # AlexNet
    input_lay = Input(shape=config['input_shape'])
    conv_1 = Conv3D(filters=96, kernel_size=(11, 11, 11), strides=(4, 4, 4), padding='valid')(input_lay)
    nor_1 = BatchNormalization()(conv_1)
    act_1 = Activation('relu')(nor_1)
    maxp_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='valid')(act_1)

    conv_2 = Conv3D(filters=256, kernel_size=(11, 11, 11), strides=(1, 1, 1), padding='valid')(maxp_1)
    nor_2 = BatchNormalization()(conv_2)
    act_2 = Activation('relu')(nor_2)
    maxp_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='valid')(act_2)

    conv_3 = Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid')(maxp_2)
    nor_3 = BatchNormalization()(conv_3)
    act_3 = Activation('relu')(nor_3)

    conv_4 = Conv3D(filters=384, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid')(act_3)
    nor_4 = BatchNormalization()(conv_4)
    act_4 = Activation('relu')(nor_4)

    conv_5 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid')(act_4)
    nor_5 = BatchNormalization()(conv_5)
    act_5 = Activation('relu')(nor_5)
    maxp_5 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='valid')(act_5)


    flat_1 = Flatten()(maxp_5)
    den_1 = Dense(units=4096, activation='relu')(flat_1)
    nor_4 = BatchNormalization()(den_1)
    den_2 = Dense(units=1000, activation='relu')(nor_4)
    nor_5 = BatchNormalization()(den_2)
    output = Dense(units=config['n_classes'], activation='softmax')(nor_5)
    model = Model(inputs=input_lay, outputs=output)

    opt = keras.optimizers.Adadelta()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # print(metricsodel.summary())
    # no callback yet
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=n_train_steps,
                                  epochs=config["n_epochs"],
                                  validation_data=validation_generator,
                                  validation_steps=n_validation_steps,
                                  verbose=1
                                  )
    print('Training done...')

    X_data = data_util.open_data_file(read_file_path).root.data[0:20]
    # y_data_o = data_util.open_data_file(read_file_path).root.label[0:20]
    # n_classes = np.unique(y_data_o).shape[0]
    # y_data = keras.utils.to_categorical(y_data_o, num_classes=n_classes)
    score = model.predict(X_data)

    print(score)
