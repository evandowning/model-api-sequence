# Based on Matthew Landen's code
# Ported for using on malware API call sequences

import sys
import os

from sklearn import metrics
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Input, Dense, LSTM
from keras.preprocessing import sequence as seq
from keras import callbacks as cb
from keras import utils as ksUtil

#TODO
def train_lstm(samples):
    nFolds = 10
    foldCount = 0

    predictions = None
    ground_truths = None
    yHatVecs = None

    # Create folds in dataset
    folds = KFold(n_splits=nFolds, shuffle=True)

    # Train each fold
    for train, test in folds.split(samples):
        foldCount += 1
        print "Training for fold " + str(foldCount) + "/" + str(nFolds)
        continue

        xTrain = xSeq[train]
        yTrain = ySeq[train]
        xTest = xSeq[test]
        yTest = ySeq[test]

        # Train the LSTM on N - 1 folds
        lstm = build_LSTM_model(xTrain, yTrain)

        testNames = names[test]

        #test on remaining fold
        yHat = lstm.predict(xTest)
        for i in range(len(testNames)):
            classLabel = yTest[i].tolist().index(yTest[i].max())
            prediction = yHat[i].tolist().index(yHat[i].max())
            eventChainIdx = int(testNames[i])

            predictions[eventChainIdx] = prediction
            ground_truths[eventChainIdx] = classLabel
            yHatVecs[eventChainIdx] = yHat[i]

    return predictions, ground_truths, yHatVecs

#TODO
def build_LSTM_model(xSeqSet, ySeqSet):
    hidden_layers = 100#int((self.maxSeqLen + self.CLASS_COUNT) / 2) ** 2    
    early_stop = cb.EarlyStopping(monitor='acc', min_delta = 0.0001, patience = 3)

    model = Sequential()
    model.add(
        LSTM(hidden_layers, input_shape=(
                self.maxSeqLen, len(xSeqSet[0][0])),
                return_sequences=False))

    model.add(Dense(len(ySeqSet[0]), activation='softmax'))
    # model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['acc'])
    # print(model.summary())
    model.fit(
        xSeqSet,
        ySeqSet,
        epochs = 1,
        batch_size = 200,
        callbacks = [early_stop])
    return model

def usage():
    print 'usage: python lstm.py features/labels features/'
    sys.exit(2)

def _main():
    if len(sys.argv) != 3:
        usage()

    labels_fn = sys.argv[1]
    folder = sys.argv[2]

    #TODO
    # Train LSTM
    train_lstm(samples)

if __name__ == '__main__':
    _main()
