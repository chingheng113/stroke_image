import sys, os
sys.path.append(os.path.abspath(os.path.join('/data/linc9/stroke_image/')))
from data import data_util
from classification import generator, models
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
config['batch_size'] = 30
config["n_epochs"] = 1

if __name__ == '__main__':
    read_file_path = data_util.write_data_to_file(config)
    data_file = data_util.open_data_file(read_file_path)
    train_generator, validation_generator, n_train_steps, n_validation_steps = generator.get_training_and_validation_generators(data_file, config)

    model = models.get_AlexNet(config)
    print(model.summary())
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=n_train_steps,
                                  epochs=config["n_epochs"],
                                  validation_data=validation_generator,
                                  validation_steps=n_validation_steps,
                                  verbose=1
                                  )
    data_util.save_history(model.name, history)
    print('Training done..')

    X_data = data_util.open_data_file(read_file_path).root.data[:]
    y_data_o = data_util.open_data_file(read_file_path).root.label[:]
    y_data = keras.utils.to_categorical(y_data_o, num_classes=config['n_classes'])
    # score = model.predict(X_data)
    loss, acc = model.evaluate(X_data, y_data, verbose=0)
    print(acc)
