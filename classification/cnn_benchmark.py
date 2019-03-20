from keras.engine import Input, Model
import numpy as np
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Input, Activation, BatchNormalization, Dropout
from keras import backend
import keras
import glob, os
from data import data_util

backend.set_image_data_format('channels_first')
config = dict()
config['which_machine'] = 'ct'
config['image_shape'] = (64, 64, 64)
if config['which_machine'] == 'ct':
    config['all_modalities'] = ['ct']
else:
    config["all_modalities"] = ['t1', 't1ce', 'flair', 't2']
config['n_channels'] = len(config["all_modalities"])
config['input_shape'] = tuple([config['n_channels']] + list(config['image_shape'])) # CHANNEL, WIDTH, HEIGHT, DEPTH

if __name__ == '__main__':
    read_file_path = data_util.write_data_to_file(config)
    X_data = data_util.open_data_file(read_file_path).root.data[0:8]


    y_data_o = data_util.open_data_file(read_file_path).root.label[0:8]
    n_classes = np.unique(y_data_o).shape[0]
    y_data = keras.utils.to_categorical(y_data_o, num_classes=n_classes)

    input_lay = Input(shape=config['input_shape'])
    conv_1 = Conv3D(filters=5, kernel_size=(3, 3, 3), strides=(1, 1, 1))(input_lay)
    # nor_1 = BatchNormalization(axis=1)(conv_1)
    maxp_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_1)

    conv_2 = Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1))(maxp_1)
    maxp_2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_2)

    conv_3 = Conv3D(filters=11, kernel_size=(3, 3, 3), strides=(1, 1, 1))(maxp_2)
    maxp_3 = MaxPooling3D(pool_size=(2, 2, 2))(conv_3)


    flat_1 = Flatten()(maxp_3)
    den_1 = Dense(units=1000, activation='relu')(flat_1)
    drop_1 = Dropout(rate=0.2)(den_1)
    den_2 = Dense(units=50, activation='relu')(drop_1)
    output = Dense(units=n_classes, activation='softmax')(den_2)
    model = Model(inputs=input_lay, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    history = model.fit(x=X_data,
                        y=y_data,
                        batch_size=3,
                        epochs=30,
                        verbose=1
                        )
    score = model.predict(X_data)
    print(score)