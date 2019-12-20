import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras import regularizers
from keras.engine.topology import Layer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import itertools

song_labels = ["Blues","Classical","Country","Disco","Hip hop","Jazz","Metal","Pop","Reggae","Rock"]

###################################################################################################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def metric(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))

def cnn(num_genres=10, input_shape=(64,173,1)):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(4, 4),
                     activation='relu', #kernel_regularizer=regularizers.l2(0.04),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu'
                    , kernel_regularizer=regularizers.l2(0.04)
                    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (2, 2), activation='relu'
       # , kernel_regularizer=regularizers.l2(0.04)
        ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.04)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.04)))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=[metric])
    return(model)
    
class model(object):

    def __init__(self, ann_model):
        self.model = ann_model()

    def train_model(self, train_x, train_y,
                val_x=None, val_y=None,
                small_batch_size=200, max_iteration=10, print_interval=1,
                test_x=None, test_y=None):

        m = len(train_x)

        for it in range(max_iteration):

            # split training data into even batches
            batch_idx = np.random.permutation(m)
            train_x = train_x[batch_idx]
            train_y = train_y[batch_idx]

            num_batches = int(m / small_batch_size)
            for batch in range(num_batches):

                x_batch = train_x[ batch*small_batch_size : (batch+1)*small_batch_size]
                y_batch = train_y[ batch*small_batch_size : (batch+1)*small_batch_size]
                print("starting batch\t", batch, "\t Epoch:\t", it)
                self.model.train_on_batch(x_batch, y_batch)

            if it % print_interval == 0:
                validation_accuracy = self.model.evaluate(val_x, val_y)
                training_accuracy = self.model.evaluate(train_x, train_y)
                testing_accuracy = self.model.evaluate(test_x, test_y)
                # print of test error used only after development of the model
                print("\nTraining accuracy: %f\t Validation accuracy: %f\t Testing Accuracy: %f" %
                      (training_accuracy[1], validation_accuracy[1], testing_accuracy[1]))
                print("\nTraining loss: %f    \t Validation loss: %f    \t Testing Loss: %f \n" %
                      (training_accuracy[0], validation_accuracy[0], testing_accuracy[0]))
                print( )

            if (validation_accuracy[1] > .81):
                print("Saving confusion data...")
                model_name = "model" + str(100*validation_accuracy[1]) + str(100*testing_accuracy[1]) + ".h5"
                self.model.save(model_name) 
                pred = self.model.predict_classes(test_x, verbose=1)
                cnf_matrix = confusion_matrix(np.argmax(test_y, axis=1), pred)
                np.set_printoptions(precision=2)
                plt.figure()
                plot_confusion_matrix(cnf_matrix, classes=song_labels, normalize=True, title='Normalized confusion matrix')
                print(precision_recall_fscore_support(np.argmax(test_y, axis=1),pred, average='macro')) 
                plt.savefig(str(batch))
                
train_data = np.load('F:/ML/project/melspects/x_tr.npy')
train_label = np.load('F:/ML/project/melspects/y_tr.npy')
val_data = np.load('F:/ML/project/melspects/x_cv.npy')
val_label = np.load('F:/ML/project/melspects/y_cv.npy')
test_data = np.load('F:/ML/project/melspects/x_te.npy')
test_label = np.load('F:/ML/project/melspects/y_te.npy')

tr_idx = np.random.permutation(len(train_data))
te_idx = np.random.permutation(len(test_data))
val_idx = np.random.permutation(len(val_data))
train_data = train_data[tr_idx]
train_label = train_label[tr_idx]
val_data = val_data[val_idx]
val_label = val_label[val_idx]
test_data = test_data[te_idx] 
test_label = test_label[te_idx]
train_data = train_data[:,:,:,np.newaxis]
test_data = test_data[:,:,:,np.newaxis]
val_data = val_data[:,:,:,np.newaxis]

train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)
val_label = np_utils.to_categorical(val_label)

ann = model(cnn)
ann.train_model(train_data, train_label, val_x=val_data, val_y=val_label, test_x=test_data, test_y=test_label)
