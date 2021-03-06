import sys, os
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
from data import data_util
from classification import generator, models
from keras import backend
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import keras.optimizers

backend.set_image_data_format('channels_first')
backend.set_image_dim_ordering('th')

config = dict()
config['which_machine'] = 'mri' # Need to rewrite 'write_ct_image_label_to_file' for training and testing
config['image_shape'] = (96, 256, 256) # CHANNEL, DEPTH, WIDTH, HEIGHT
config['n_classes'] = 2
if config['which_machine'] == 'ct':
    config['all_sequences'] = ['ct']
else:
    config["all_sequences"] = ['dwi']
config['n_channels'] = len(config["all_sequences"])
config['input_shape'] = tuple([config['n_channels']] + list(config['image_shape']))
config['batch_size'] = 5
config['n_epochs'] = 200

if __name__ == '__main__':
    read_training_file_path = os.path.join('..', 'data', config['which_machine'], config['which_machine']+'_data_training.h5')
    read_test_file_path = os.path.join('..', 'data', config['which_machine'], config['which_machine']+'_data_testing.h5')
    if os.path.isfile(read_training_file_path) & os.path.isfile(read_test_file_path):
        print('Using exist data')
    else:
        read_training_file_path = data_util.write_data_to_file(config, 'training')
        read_test_file_path = data_util.write_data_to_file(config, 'testing')
    # train data
    training_data_file = data_util.open_data_file(read_training_file_path)
    # test data
    X_test_data = data_util.open_data_file(read_test_file_path).root.data[:]
    y_test_data_o = data_util.open_data_file(read_test_file_path).root.label[:]
    y_test_data = y_test_data_o
    # y_test_data = keras.utils.to_categorical(y_test_data_o, num_classes=config['n_classes'])

    # Training
    train_generator, validation_generator, n_train_steps, n_validation_steps = generator.get_training_and_validation_generators(training_data_file, config)
    model = models.get_VoxCNN(config)
    print(model.summary())
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=n_train_steps,
                                  epochs=config["n_epochs"],
                                  validation_data=validation_generator,
                                  validation_steps=n_validation_steps,
                                  callbacks=[ReduceLROnPlateau(factor=0.5, patience=20, verbose=1),
                                             EarlyStopping(verbose=1, patience=100),
                                             ModelCheckpoint(os.path.join('..', 'results', model.name + '.h5'),
                                                             save_best_only=True, verbose=1)],
                                  verbose=1
                                  )
    data_util.save_history(model.name, history)
    # data_util.save_model(model.name, model)
    print('Training done..')
    # Testing
    loss, acc = model.evaluate(X_test_data, y_test_data, verbose=0)
    print(acc)
