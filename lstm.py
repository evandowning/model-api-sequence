# Based on Matthew Landen's code
# Ported for using on malware API call sequences

import sys
import os
import cPickle as pkl
import numpy as np
import math

from sklearn import metrics
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Input, Dense, LSTM
from keras.preprocessing import sequence as seq
from keras import callbacks as cb
from keras import utils as ksUtil

# Creates multiple generators of the data to use on Keras
# We do this because we can have very large datasets we can't
# fit entirely into memory.
def sequence_generator(folder, sample, labels, labelMap, foldIDs, batchSize):
    xSet = list()
    ySet = list()

    for e,i in enumerate(foldIDs):
        x = list()
        y = list()

        # Read in sample's sequence and convert it to hot encoding
        path = os.path.join(folder,sample[i]+'.pkl')
        with open(path, 'rb') as fr:
            x = pkl.load(fr)

        # Here we put each api into its own array of size one.
        # This sounds silly, but it's how Keras works for a dataset like api call sequences
        # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        xSet.append([list([seq]) for seq in x])

        ySet.append(list([labelMap.index(labels[sample[i]])]))

        # Batch size reached, yield data
        if (e+1) % batchSize == 0:
            # Here we convert our lists into Numpy arrays because
            # Keras requires it as input for its fit_generator()
            x = np.array(xSet)
            y = np.array(ySet)

            xSet = list()
            ySet = list()

            yield (x, y)

    # Yield remaining set
    if len(xSet) > 0:
        yield (np.array(xSet), np.array(ySet))

# Builds LSTM model
def build_LSTM_model(trainData, trainBatches, testData, testBatches, maxLen, class_count):
    hidden_layers = 100
    early_stop = cb.EarlyStopping(monitor='acc', min_delta = 0.0001, patience = 3)

    model = Sequential()

    model.add(
        LSTM(
            hidden_layers,
            input_shape=(maxLen, 1),
            return_sequences=False
            )
        )

    model.add(Dense(class_count, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['acc'])

    hist = model.fit_generator(
        trainData,
        steps_per_epoch = trainBatches,
        epochs = 1,
        validation_data = testData,
        validation_steps = testBatches,
        callbacks = [early_stop])

    return model, hist

#TODO
# Trains and tests LSTM over samples
def train_lstm(folder, sample, labels, labelMap):
    # Number of folds in cross validation
    nFolds = 10

    # Batch size (# of samples to have LSTM train at once)
    batchSize = 11

    # Sequence lengths (should be all the same. they should be padded)
    maxLen = 0
    path = os.path.join(folder,sample[0]+'.pkl')
    with open(path, 'rb') as fr:
        x = pkl.load(fr)
        maxLen = len(x)

    # For holding results of trained networks
    predictions = dict()
    ground_truths = dict()
    yHatVecs = dict()
    yTruthVecs = dict()

    # Get folds for cross validation
    folds = KFold(n_splits=nFolds, shuffle=True)
    foldCount = 0

    # Train and test LSTM over each fold
    for trainFold, testFold in folds.split(sample):
        foldCount += 1
        print 'Training Fold {0}/{1}'.format(foldCount,nFolds)

        # Put features into format LSTM can ingest
        trainData = sequence_generator(folder, sample, labels, labelMap, trainFold, batchSize)
        testData = sequence_generator(folder, sample, labels, labelMap, testFold, batchSize)

        # Calculate number of batches
        train_num_batches = math.ceil(float(len(trainFold))/batchSize)
        test_num_batches = math.ceil(float(len(testFold))/batchSize)

        #TODO
        # Train LSTM model
        lstm,hist = build_LSTM_model(trainData, train_num_batches, testData, test_num_batches, maxLen, len(labelMap))
        print hist.history
        print ''
        continue

        # Test LSTM
        fold = self.generate_file(componentType, foldCount)
        testNames = fold.names
        yTest = fold.y

        yHat = lstm.predict(fold.x)

        for i in range(len(testNames)):
            classLabel = yTest[i].tolist().index(yTest[i].max())
            prediction = yHat[i].tolist().index(yHat[i].max())
            eventChainIdx = testNames[i]

            predictions[eventChainIdx] = prediction
            ground_truths[eventChainIdx] = classLabel
            yHatVecs[eventChainIdx] = yHat[i].tolist()
            yTruthVecs[eventChainIdx] = yTest[i].tolist()

    return predictions, ground_truths, yHatVecs, yTruthVecs

def usage():
    print 'usage: python lstm.py features/labels features/'
    sys.exit(2)

def _main():
    if len(sys.argv) != 3:
        usage()

    label_fn = sys.argv[1]
    folder = sys.argv[2]

    # Get all samples in features folder
    sample = list()

    labels = dict()
    labelMap = set()
    # Extract labels for samples
    with open(label_fn, 'r') as fr:
        for line in fr:
            line = line.strip('\n')
            s = line.split(' ')[0]
            l = line.split(' ')[1]

            sample.append(s)
            labels[s] = l
            labelMap.add(l)

    # Train LSTM
    train_lstm(folder, sample, labels, list(labelMap))

if __name__ == '__main__':
    _main()
