import sys
import os
import shutil
import cPickle as pkl
import numpy as np
import math
from collections import Counter
from multiprocessing import Pool

from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, Embedding, Conv1D, MaxPooling1D
from keras import optimizers
from keras import callbacks as cb

# Trains on per sample (i.e., file)
def sequence_generator(folder, sample, foldIDs, batchSize):
    # We want to loop infinitely because we're training our data on multiple epochs in build_LSTM_model()
    while 1:
        xSet = np.array([])
        ySet = np.array([])

        num = 0;
        for i in foldIDs:
            x = np.array([])
            y = np.array([])

            # Extract sample's name and number of sequences
            fn = sample[i][0]
            numSeq = sample[i][1]

            # Read in sample's sequences
            path = os.path.join(folder,fn+'.pkl')
            with open(path, 'rb') as fr:
                for e in enumerate(range(numSeq)):
                    t = pkl.load(fr)
                    x = t[0]
                    y = t[1]

                    if len(xSet) == 0:
                        xSet = x
                        ySet = [y]
                    else:
                        xSet = np.vstack([xSet,x])
                        ySet = np.vstack([ySet,[y]])

                    # Increase count of number of sample features extracted
                    num += 1

                    # Batch size reached, yield data
                    if num % batchSize == 0:
                        # Here we convert our lists into Numpy arrays because
                        # Keras requires it as input for its fit_generator()
                        rv_x = xSet
                        rv_y = ySet

                        xSet = np.array([])
                        ySet = np.array([])

                        num = 0

                        yield (rv_x, rv_y)

        # Yield remaining set
        if len(xSet) > 0:
            yield (xSet, ySet)

# Builds LSTM model
def build_LSTM_model(trainData, trainBatches, testData, testBatches, windowSize, class_count, numCalls, batch_size):
    # TODO - What is this?
    # Specify number of units
    # https://stackoverflow.com/questions/37901047/what-is-num-units-in-tensorflow-basiclstmcell#39440218
    num_units = 128

    embedding_size = 256

    # https://keras.io/callbacks/#earlystopping
    early_stop = cb.EarlyStopping(monitor='sparse_categorical_accuracy', min_delta = 0.0001, patience = 3)

    model = Sequential()

    # We need to add an embedding layer because LSTM (at this moment) that the API call indices (numbers)
    # are of some mathematical significance. E.g., system call 2 is "closer" to system calls 3 and 4.
    # But system call numbers have nothing to do with their semantic meaning and relation to other
    # system calls. So we transform it using an embedding layer so the LSTM can figure these relationships
    # out for itself.
    # https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

    # https://stackoverflow.com/questions/40695452/stateful-lstm-with-embedding-layer-shapes-dont-match
    api_count = numCalls+1  # +1 because 0 is our padding number
    model.add(Embedding(input_dim=api_count, output_dim=256, input_length=windowSize))

    # https://keras.io/layers/recurrent/#lstm
    model.add(LSTM(num_units,input_shape=(windowSize, api_count),return_sequences=False))

    #TODO - If I want to add more layers
    # https://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras

    # https://keras.io/layers/core/#dense
    model.add(Dense(128))
    # https://keras.io/activations/
    model.add(Activation('relu'))

    # https://keras.io/layers/core/#dropout
    model.add(Dropout(0.5))

    model.add(Dense(class_count, name='logits'))
    model.add(Activation('softmax'))

    # Which optimizer to use
    # https://keras.io/optimizers/
    opt = optimizers.RMSprop(lr=0.01,decay=0.001)

    # https://keras.io/models/model/#compile
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        # Metrics to print
        # We use sparse_categorical_accuracy as opposed to categorical_accuracy
        # because: https://stackoverflow.com/questions/44477489/keras-difference-between-categorical-accuracy-and-sparse-categorical-accuracy
        # I.e., since we don't use hot-encoding, we use sparse_categorical_accuracy
        metrics=['sparse_categorical_accuracy'])

    # https://keras.io/models/model/#fit_generator
    hist = model.fit_generator(
        # Data to train
        trainData,
        # Use multiprocessing because python Threading isn't really
        # threading: https://docs.python.org/2/glossary.html#term-global-interpreter-lock
        use_multiprocessing = True,
        # Number of steps per epoch (this is how we train our large
        # number of samples dataset without running out of memory)
        steps_per_epoch = trainBatches,
        # Number of epochs
        epochs = 100,
        # Validation data (will not be trained on)
        validation_data = testData,
        validation_steps = testBatches,
        # Do not shuffle batches.
        shuffle = False,
        # List of callbacks to be called while training.
        callbacks = [early_stop])

    return model, hist

# Trains and tests LSTM over samples
def train_lstm(folder, fileMap, model_folder, class_count, windowSize, numCalls):
    batchSize = 3000

    # Get folds for cross validation
    nFolds = 10
    folds = KFold(n_splits=nFolds, shuffle=True)
    foldCount = 0

    # We're folding on each file (which contains multiple sequences to train/test)
    # We do this for memory consumption reasons (i.e., it's hard to hold an array of 2 million lists within memory
    # to keep track of each fold)
    numSamples = len(fileMap.keys())
    sample = [(k,v) for k,v in fileMap.iteritems()]

    # Train and test LSTM over each fold
    for trainFold, testFold in folds.split(range(numSamples)):
        foldCount += 1
        print '==========================================================='
        print 'Training Fold {0}/{1}'.format(foldCount,nFolds)

        # Put features into format LSTM can ingest
        trainData = sequence_generator(folder, sample, trainFold, batchSize)
        testData = sequence_generator(folder, sample, testFold, batchSize)

        # Calculate number of batches
        numTrainSeq = 0
        for i in trainFold:
            numTrainSeq += sample[i][1]
        numTestSeq = 0
        for i in testFold:
            numTestSeq += sample[i][1]

        train_num_batches = math.ceil(float(numTrainSeq)/batchSize)
        test_num_batches = math.ceil(float(numTestSeq)/batchSize)

        print 'Number of training sequences: {0}'.format(numTrainSeq)
        print 'Number of testing sequences: {0}'.format(numTestSeq)
        print 'Training batches: {0}'.format(train_num_batches)
        print 'Testing batches: {0}'.format(test_num_batches)

        # Train LSTM model
        lstm,hist = build_LSTM_model(trainData, train_num_batches, testData, test_num_batches, windowSize, class_count, numCalls, batchSize)
        # Print accuracy histories over the folds
#       print ''
#       print hist.history

        # Save trained model
        # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        # Convert model to JSON format to be stored
        modelJSON = lstm.to_json()
        # Store model in model_folder
        fn = os.path.join(model_folder,'fold{0}-model.json'.format(foldCount))
        with open(fn,'w') as fw:
            fw.write(modelJSON)
        # Store weights for model
        fn = os.path.join(model_folder,'fold{0}-weight.h5'.format(foldCount))
        lstm.save_weights(fn)

        # Save train/test fold to be used by eval.py
        fn = os.path.join(model_folder,'fold{0}-train.pkl'.format(foldCount))
        with open(fn,'wb') as fw:
            # Write the number of data we'll have in this file
            pkl.dump(train_num_batches,fw)

            # Loop through generator
            i = 0
            for g in trainData:
                # If we've reached the end of the generator
                if i == train_num_batches:
                    break

                # Write data to file
                pkl.dump(g,fw)
                i += 1

                sys.stdout.write('Saving training fold data: {0}/{1}\r'.format(i,train_num_batches))
                sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

        fn = os.path.join(model_folder,'fold{0}-test.pkl'.format(foldCount))
        with open(fn,'wb') as fw:
            # Write the number of data we'll have in this file
            pkl.dump(test_num_batches,fw)

            # Loop through generator
            i = 0
            for g in testData:
                # If we've reached the end of the generator
                if i == test_num_batches:
                    break

                # Write data to file
                pkl.dump(g,fw)
                i += 1

                sys.stdout.write('Saving testing fold data: {0}/{1}\r'.format(i,test_num_batches))
                sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

        #TODO - only do one fold for now
        break

def usage():
    print 'usage: python lstm.py features/ models/'
    sys.exit(2)

def _main():
    if len(sys.argv) != 3:
        usage()

    feature_folder = sys.argv[1]
    model_folder = sys.argv[2]

    # Remove model folder if it already exists
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
        os.mkdir(model_folder)
    else:
        os.mkdir(model_folder)

    windowSize = 0
    finalExtracted = 0
    labelCount = Counter()
    fileMap = dict()

    # Count number of unique api calls
    with open('api.txt','r') as fr:
        apis = fr.read()
    apis = apis.split('\n')
    numCalls = len(apis)

    # Extract metadata
    metafn = os.path.join(feature_folder,'metadata.pkl')
    with open(metafn,'rb') as fr:
        # Window Size
        windowSize = pkl.load(fr)
        # Number of samples per label
        labelCount = pkl.load(fr)
        # Number of samples per data file (so we can determine folds properly)
        fileMap = pkl.load(fr)

    print 'Window size: {0}'.format(windowSize)

    print 'total samples: {0}'.format(sum(fileMap.values()))
    print 'total samples: {0}'.format(sum(labelCount.values()))

    # Print class labels and counts
    print 'Total Dataset:'
    for k,v in labelCount.most_common():
        sys.stdout.write('Class: {0: <10} Count: {1: <10} ({2:.2f}% of dataset)\n'.format(k,v,100*float(v)/sum(labelCount.values())))
    print ''

    # Train LSTM
    train_lstm(feature_folder, fileMap, model_folder, numCalls, windowSize, numCalls)

if __name__ == '__main__':
    _main()
