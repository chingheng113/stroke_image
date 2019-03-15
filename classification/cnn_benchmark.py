from keras.engine import Input, Model
import SimpleITK as sitk
from keras.layers import Dense, Conv3D, MaxPooling3D, Flatten, Input, Activation, BatchNormalization, Dropout
from keras import backend
import keras
import glob, os
from data import data_util


if __name__ == '__main__':
    img_paths = data_util.get_img_paths('ct')
    sitk_t1 = sitk.ReadImage(img_paths[0])
    t1 = sitk.GetArrayFromImage(sitk_t1)
    # I want overfitting
    x_train_o = t1.T
    x_train = x_train_o.reshape(-1, 1, 64, 64, 64)
    x_test = x_train

    y_training = keras.utils.to_categorical([0], num_classes=2)
    y_test = y_training

    # CHANNEL, WIDTH, HEIGHT, DEPTH
    backend.set_image_data_format('channels_first')
    input_lay = Input(shape=(1, 64, 64, 64))
    conv_1 = Conv3D(filters=3, kernel_size=(3, 3, 3), strides=(1, 1, 1))(input_lay)
    nor_1 = BatchNormalization(axis=1)(conv_1)
    maxp_1 = MaxPooling3D(pool_size=(2, 2, 2))(nor_1)
    flat_1 = Flatten()(maxp_1)
    den_1 = Dense(units=1000, activation='relu')(flat_1)
    drop_1 = Dropout(rate=0.2)(den_1)
    den_2 = Dense(units=50, activation='relu')(drop_1)
    output = Dense(units=2, activation='softmax')(den_2)
    model = Model(inputs=input_lay, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x=x_train,
                        y=y_training,
                        batch_size=1,
                        epochs=30,
                        verbose=1
                        )
    score = model.predict(x_test)
    print(score)