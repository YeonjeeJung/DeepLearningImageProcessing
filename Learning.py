import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.losses import mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras import optimizers
import random

def load_data(frames, labels):
    if len(frames) != len(labels):
        print("Size not matched")
        return

    ret_frames = []
    ret_labels = []

    for i, frame_one_label in enumerate(frames):
        thislabel = [0 for _ in range(len(labels))]
        thislabel[i] = 1
        thislabel = np.array(thislabel)
        for thisframe in frame_one_label:
            ret_frames.append(thisframe)
            ret_labels.append(thislabel)

        #thislabel = [0 for _ in range(len(labels))]
        #thislabel[i] = 1
        #labels[i] = thislabel
        #Dataset.append((frames[i],thislabel))

    ret_frames = np.array(ret_frames)
    ret_labels = np.array(ret_labels)
    """
    for i, (frame, label) in enumerate(zip(frames,labels)):
        thislabel = [0 for _ in range(len(labels))]
        thislabel[i] = 1
        thislabel = [thislabel for _ in range(len(frame))]
        thislabel = np.array(thislabel)
        Dataset.append((frame,thislabel))
    """

    return (ret_frames, ret_labels)

def my_model(len_output):
    model = Sequential()
    model.add(Conv2D(32,(3,3)))
    model.add(Conv2D(32,(3,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    #model.add(Conv2D(1,(3,3)))
    model.add(Conv2D(32,(3,3)))
    model.add(Conv2D(32,(3,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(1,(3,3)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(len_output, activation='softmax'))

    #sgd = optimizers.SGD(lr=0.0001)
    #model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
    return model

def train(frames, labels, epoches=5, lr=0.0001):
    #print("len frames, labels",str(len(frames)), str(len(labels)))

    model = my_model(len(labels))
    #model = VGG16(weights=None, pooling='max', classes=len(labels))
    #model = Xception(weights=None, pooling='max', classes=len(labels))
    #model = MobileNet(weights=None, pooling='max', classes=len(labels))

    sgd = optimizers.Adam(lr=lr)
    model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    (ret_frames, ret_labels) = load_data(frames,labels)

    #print("len retframes, labels ",str(len(ret_frames)),str(len(ret_labels)))

    model.fit(x=ret_frames, y=ret_labels, epochs=epoches, shuffle=True, batch_size=16)
    """
    for epoch in range(epoches):
        print(str(epoch)+"th epoch")
        for i in random.sample(range(len(ret_frames)), len(ret_frames)):
            inputframe = ret_frames[i].reshape(1, *ret_frames[i].shape)
            inputlabel = ret_labels[i].reshape(1, *ret_labels[i].shape)
            model.fit(x=inputframe, y=inputlabel, verbose=0)
    """
    return model
