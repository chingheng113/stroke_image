import pickle
import matplotlib.pyplot as plt


def plot_training_acc():
    with open('trainHistoryDict', 'rb') as f:
        history = pickle.load(f)
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def plot_training_loss():
    with open('trainHistoryDict', 'rb') as f:
        history = pickle.load(f)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    plot_training_acc()
    plot_training_loss()