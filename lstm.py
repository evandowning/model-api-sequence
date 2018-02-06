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

def find_longest_event_chain():
    maxLen = -1
    for app in appSet:
        for cp in app.callpaths:
            length = len(cp.event_chain.events)
            if len(cp.event_chain.events[0].actions) > 0:
                length += 1
            if length > maxLen:
                maxLen = length

    return maxLen    

#TODO - sena
def get_sequences(folder, samples):
    # Pick one file
    s = samples[1]

    # Read in API call sequence and map each unique API call to a number

    # Print that number to screen

def get_longest(folder, samples):
    rv = 0

    # Iterate through each file
    for s in samples:
        count = 0

        # Read file
        path = os.path.join(folder,s)
        with open(path,'r') as fr:
            for line in fr:
                count += 1

        # If this is the max count of lines, store it in rv
        if count > rv:
            rv = count

    return rv

#TODO
def extract_features(folder, samples, labels):
    # Get longest API sequence
    longest = get_longest(folder,samples)
    print longest

    # TODO - sena
    # Convert API call sequences into numbers instead of strings AND
    # the malware family labels into numbers instead of strings
    get_sequences(folder,samples)
    sys.exit(1)

# Extracts labels from file
def get_labels(fn):
    rv = dict()

    with open(fn,'r') as fr:
        for line in fr:
            line = line.strip('\n')

            h = line.split('\t')[0]
            label = line.split('\t')[-1]

            if label not in rv:
                rv[label] = list()
            rv[label].append(h)

    return rv

def usage():
    print 'usage: python lstm.py api-sequence-folder/ hash.labels features-folder/'
    sys.exit(2)

def _main():
    if len(sys.argv) != 4:
        usage()

    folder = sys.argv[1]
    labels_fn = sys.argv[2]
    features = sys.argv[3]

    # Final set of samples
    samples = list()
    # Final set of labels
    final = list()

    # Read in labels
    labels = get_labels(labels_fn)
    for k in sorted(labels, key=lambda k: len(labels[k]), reverse=True):
        # Only consider families with >= 10 samples
        if len(labels[k]) < 10:
            break
        else:
            final.append(k)
            samples.extend(labels[k])

    # Extract features of files in data folder
    extract_features(folder, samples, labels)

    # Train LSTM
    train_lstm(samples)

if __name__ == '__main__':
    _main()
