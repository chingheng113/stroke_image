import pickle
import matplotlib.pyplot as plt


def plot_training_acc(model_name):
    with open(model_name+'_trainHistory.pickle', 'rb') as f:
        history = pickle.load(f)
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def plot_training_loss(model_name):
    with open(model_name+'_trainHistory.pickle', 'rb') as f:
        history = pickle.load(f)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    model_name = 'AlexNet'
    plot_training_acc(model_name)
    plot_training_loss(model_name)