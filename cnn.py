#!/usr/bin/env python3

import sys
import os
import pickle as pkl
import numpy as np
import math
from collections import Counter
from multiprocessing import Pool

from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, GlobalMaxPooling1D, Input, Embedding, Multiply, Dropout, MaxPooling1D, Flatten
from keras.models import Model
from keras import optimizers
from keras import callbacks as cb

# Trains on per sample (i.e., file)
def sequence_generator(folder, sample, foldIDs, batchSize, task, convert):
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

                    # If this should be binary classification, convert labels > 0 to 1
                    if task == 'binary_classification':
                        if y > 0:
                            y = 1
                    elif task == 'multi_classification':
                        y = convert.index(y)

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
    # Specify number of units
    # https://stackoverflow.com/questions/37901047/what-is-num-units-in-tensorflow-basiclstmcell#39440218
    num_units = 128

    embedding_size = 256

    # https://keras.io/callbacks/#earlystopping
    early_stop = cb.EarlyStopping(monitor='sparse_categorical_accuracy', min_delta = 0.0001, patience = 3)

    # We need to add an embedding layer because LSTM (at this moment) that the API call indices (numbers)
    # are of some mathematical significance. E.g., system call 2 is "closer" to system calls 3 and 4.
    # But system call numbers have nothing to do with their semantic meaning and relation to other
    # system calls. So we transform it using an embedding layer so the LSTM can figure these relationships
    # out for itself.
    # https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

    # https://stackoverflow.com/questions/40695452/stateful-lstm-with-embedding-layer-shapes-dont-match
    api_count = numCalls+1  # +1 because 0 is our padding number
    inp = Input( shape=(windowSize,))
    emb = Embedding(input_dim=api_count, output_dim=256, input_length=windowSize)(inp)

    # https://keras.io/layers/recurrent/#lstm
#   model.add(LSTM(num_units,input_shape=(windowSize, api_count),return_sequences=False))
    #TODO - GPU stuffs
#   model.add(CuDNNLSTM(num_units,input_shape=(windowSize, api_count),return_sequences=False))

    # From  malconv paper
    filt = Conv1D( filters=64, kernel_size=3, strides=1, use_bias=True, activation='relu', padding='valid' )(emb)
    attn = Conv1D( filters=64, kernel_size=3, strides=1, use_bias=True, activation='sigmoid', padding='valid')(emb)
    gated = Multiply()([filt,attn])
    drop = Dropout(0.5)(gated)
    feat = GlobalMaxPooling1D()( drop )
    dense = Dense(128, activation='relu')(feat)
    outp = Dense(class_count, activation='sigmoid')(dense)
    model = Model( inp, outp )

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
        # threading: https://docs.python.org/3/glossary.html#term-global-interpreter-lock
        use_multiprocessing = True,
        # Number of steps per epoch (this is how we train our large
        # number of samples dataset without running out of memory)
        steps_per_epoch = trainBatches,
        #TODO
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
def train_lstm(folder, fileMap, model_folder, class_count, windowSize, numCalls, save_model, save_data, task, convert):
    #TODO
    batchSize = 512

    # Get folds for cross validation
    nFolds = 10
    folds = KFold(n_splits=nFolds, shuffle=True)
    foldCount = 0

    # We're folding on each file (which contains multiple sequences to train/test)
    # We do this for memory consumption reasons (i.e., it's hard to hold an array of 2 million lists within memory
    # to keep track of each fold)
    numSamples = len(list(fileMap.keys()))
    sample = [(k,v) for k,v in fileMap.items()]

    # Train and test LSTM over each fold
    for trainFold, testFold in folds.split(list(range(numSamples))):
        foldCount += 1
        sys.stdout.write('===========================================================\n')
        sys.stdout.write('Training Fold {0}/{1}\n'.format(foldCount,nFolds))

        # Put features into format LSTM can ingest
        trainData = sequence_generator(folder, sample, trainFold, batchSize, task, convert)
        testData = sequence_generator(folder, sample, testFold, batchSize, task, convert)

        # Calculate number of batches
        numTrainSeq = 0
        for i in trainFold:
            numTrainSeq += sample[i][1]
        numTestSeq = 0
        for i in testFold:
            numTestSeq += sample[i][1]

        train_num_batches = math.ceil(float(numTrainSeq)/batchSize)
        test_num_batches = math.ceil(float(numTestSeq)/batchSize)

        sys.stdout.write('Number of training sequences: {0}\n'.format(numTrainSeq))
        sys.stdout.write('Number of testing sequences: {0}\n'.format(numTestSeq))
        sys.stdout.write('Training batches: {0}\n'.format(train_num_batches))
        sys.stdout.write('Testing batches: {0}\n'.format(test_num_batches))

        # Train LSTM model
        lstm,hist = build_LSTM_model(trainData, train_num_batches, testData, test_num_batches, windowSize, class_count, numCalls, batchSize)

        # Save trained model
        if save_model:
            sys.stdout.write('Saving model...')
            sys.stdout.flush()

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

            sys.stdout.write('done\n')

        # Save train/test fold to be used by eval.py
        if save_data:
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

        #NOTE: only do one fold for time
        break

def usage():
    sys.stderr.write('usage: python lstm.py cuckoo-headless/extract_raw/api.txt features/ models/ save_model[True|False] save_data[True|False] {binary_classification | multi_classification | regression} convert_class.txt\n\n')
    sys.stderr.write('\tout_class.txt: file which stores trained class values in case they change from the original stored\n')
    sys.exit(2)

def _main():
    if len(sys.argv) != 8:
        usage()

    api_fn = sys.argv[1]
    feature_folder = sys.argv[2]
    model_folder = sys.argv[3]
    save_model = eval(sys.argv[4])
    save_data = eval(sys.argv[5])
    task = sys.argv[6]
    convert_fn = sys.argv[7]

    # Test task parameter
    if task not in ['binary_classification', 'multi_classification', 'regression']:
        sys.stdout.write(('Error. "{0}" parameter is not "binary_classification" or "multi_classification" or "regression"\n'.format(task)))
        usage()

    # Error if model folder already exists
    if os.path.exists(model_folder):
        sys.stderr.write(('Error. Model folder "{0}" already exists.\n'.format(model_folder)))
        sys.exit(1)

    os.mkdir(model_folder)

    windowSize = 0
    finalExtracted = 0
    labelCount = Counter()
    fileMap = dict()

    # Count number of unique api calls
    with open(api_fn,'r') as fr:
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

    sys.stdout.write('Window size: {0}\n'.format(windowSize))

    sys.stdout.write('total samples: {0}\n'.format(sum(fileMap.values())))
    sys.stdout.write('total samples: {0}\n'.format(sum(labelCount.values())))

    # Print class labels and counts
    sys.stdout.write('Total Dataset:\n')
    for k,v in labelCount.most_common():
        sys.stdout.write('Class: {0: <10} Count: {1: <10} ({2:.2f}% of dataset)\n'.format(k,v,100*float(v)/sum(labelCount.values())))
    sys.stdout.write('\n')

    # Store class conversion
    if task == 'binary_classification':
        convert = [k for k,v in labelCount.most_common()]
        with open(convert_fn,'w') as fw:
            # NOTE: format: Trained Actual
            for i in convert:
                if i > 0:
                    fw.write('{0} {1}\n'.format(1,i))
                else:
                    fw.write('{0} {1}\n'.format(0,i))
    elif task == 'multi_classification':
        convert = sorted([k for k,v in labelCount.most_common()])
        with open(convert_fn,'w') as fw:
            # NOTE: format: Trained Actual
            for i in range(len(convert)):
                fw.write('{0} {1}\n'.format(i,convert[i]))

    # Train LSTM
    if task == 'binary_classification':
        train_lstm(feature_folder, fileMap, model_folder, 2, windowSize, numCalls, save_model, save_data, task, [])
    elif task == 'multi_classification':
        train_lstm(feature_folder, fileMap, model_folder, len(labelCount.keys())+1, windowSize, numCalls, save_model, save_data, task, convert)
    elif task == 'regression':
        train_lstm(feature_folder, fileMap, model_folder, numCalls, windowSize, numCalls, save_model, save_data, task, [])

if __name__ == '__main__':
    _main()
