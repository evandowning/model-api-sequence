import sys
import os
import shutil
import cPickle as pkl
import numpy as np
import math
from collections import Counter
from multiprocessing import Pool

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, Embedding, Conv1D, MaxPooling1D
from keras import optimizers
from keras import callbacks as cb

# Determines how many sequences we'll be giving to the LSTM
def numSequences(folder, sample, foldID, windowSize):
    rv = 0

    # Read in sample's sequence and convert it a sequence integers
    path = os.path.join(folder,sample[foldID]+'.pkl')
    with open(path, 'rb') as fr:
        x = pkl.load(fr)
        rv = len(x)-windowSize+1

    return rv

def numSequences_wrapper(args):
    return numSequences(*args)

# Creates multiple generators of the data to use on Keras
# We do this because we can have very large datasets we can't
# fit entirely into memory.
def sequence_generator(folder, sample, labels, labelMap, foldIDs, batchSize, windowSize):
    # We want to loop infinitely because we're training our data on multiple epochs in build_LSTM_model()
    while 1:
        xSet = np.array([])
        ySet = np.array([])

        num = 0;
        for i in foldIDs:
            x = np.array([])
            y = np.array([])

            # Read in sample's sequence and convert it a sequence integers
            path = os.path.join(folder,sample[i]+'.pkl')
            with open(path, 'rb') as fr:
                x = np.array(pkl.load(fr))

            # Pad sequence to window size if necessary
            remainder = windowSize - len(x)
            if remainder > 0:
                x = np.append(x,[0]*remainder)

            # From: https://stackoverflow.com/questions/15722324/sliding-window-in-numpy/42258242#42258242
            # Construct sliding windows
            index = np.arange(windowSize)[None,:] + np.arange(len(x)-windowSize+1)[:,None]

            # Insert sliding windowed inputs
            for w in x[index]:
                # If this is the first element
                if len(xSet) == 0:
                    xSet = w
                    ySet = [labelMap.index(labels[sample[i]])]
                # Else, keep the shape
                else:
                    xSet = np.vstack([xSet,w])
                    ySet = np.vstack([ySet,[labelMap.index(labels[sample[i]])]])

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
def build_LSTM_model(trainData, trainBatches, testData, testBatches, windowSize, class_count, numCalls):
    # TODO - What is this?
    # Specify number of units
    # https://stackoverflow.com/questions/37901047/what-is-num-units-in-tensorflow-basiclstmcell#39440218
    num_units = 128

    # https://keras.io/callbacks/#earlystopping
    early_stop = cb.EarlyStopping(monitor='sparse_categorical_accuracy', min_delta = 0.0001, patience = 3)

    model = Sequential()

    # We need to add an embedding layer because LSTM (at this moment) that the API call indices (numbers)
    # are of some mathematical significance. E.g., system call 2 is "closer" to system calls 3 and 4.
    # But system call numbers have nothing to do with their semantic meaning and relation to other
    # system calls. So we transform it using an embedding layer so the LSTM can figure these relationships
    # out for itself.
    # https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

    api_count = numCalls+1  # +1 because 0 is our padding number
    model.add(Embedding(input_dim=api_count, output_dim=256, input_length=windowSize))

    # https://keras.io/layers/recurrent/#lstm
    model.add(
        LSTM(
            num_units,
            input_shape=(windowSize, api_count),
            return_sequences=False
            )
        )

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
        # Shuffle order of batches at beginning of each epoch
        shuffle = True,
        # List of callbacks to be called while training.
        callbacks = [early_stop])

    return model, hist

# Trains and tests LSTM over samples
def train_lstm(folder, sample, labels, labelMap, model_folder, windowSize, numCalls):
    # Number of folds in cross validation
    nFolds = 10

    # Batch size (# of samples to have LSTM train at a time)
    # It's okay if this does not evenly divide your entire sample set
    batchSize = 500

    # Sequence lengths (should be all the same. they should be padded)
    path = os.path.join(folder,sample[0]+'.pkl')
    with open(path, 'rb') as fr:
        x = pkl.load(fr)

    # Get folds for cross validation
    folds = KFold(n_splits=nFolds, shuffle=True)
    foldCount = 0

    # Train and test LSTM over each fold
    for trainFold, testFold in folds.split(sample):
        foldCount += 1
        print '==========================================================='
        print 'Training Fold {0}/{1}'.format(foldCount,nFolds)

        # Put features into format LSTM can ingest
        trainData = sequence_generator(folder, sample, labels, labelMap, trainFold, batchSize, windowSize)
        testData = sequence_generator(folder, sample, labels, labelMap, testFold, batchSize, windowSize)

        # Count number of sequences we'll be passing to the LSTM
        numTrain = 0
        # Create argument pools
        args = [(folder,sample,i,windowSize) for i in trainFold]
        pool = Pool(10)
        results = pool.imap_unordered(numSequences_wrapper, args)
        for e,r in enumerate(results):
            numTrain += r
            sys.stdout.write('Counting sequences: {0}/{1}\r'.format(e+1,len(trainFold)))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

        # Count number of sequences we'll be passing to the LSTM
        numTest = 0
        # Create argument pools
        args = [(folder,sample,i,windowSize) for i in testFold]
        pool = Pool(10)
        results = pool.imap_unordered(numSequences_wrapper, args)
        for e,r in enumerate(results):
            numTest += r
            sys.stdout.write('Counting sequences: {0}/{1}\r'.format(e+1,len(testFold)))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

        # Calculate number of batches
        train_num_batches = math.ceil(float(numTrain)/batchSize)
        test_num_batches = math.ceil(float(numTest)/batchSize)

        print 'Number of training sequences: {0}'.format(numTrain)
        print 'Number of testing sequences: {0}'.format(numTest)
        print 'Training batches: {0}'.format(train_num_batches)
        print 'Testing batches: {0}'.format(test_num_batches)

        # Train LSTM model
        lstm,hist = build_LSTM_model(trainData, train_num_batches, testData, test_num_batches, windowSize, len(labelMap), numCalls)
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

        # Run predictions over test data one last time to get final results
        # https://keras.io/models/model/#predict_generator
        p = lstm.predict_generator(testData, steps=test_num_batches, use_multiprocessing=True)

        # Extract predicted classes for each sample in testData
        # https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
        predictClasses = p.argmax(axis=-1)

        # NOTE: Sloppy way of doing this, but it'll work for now
        # Extract true labels for training and testing data
        trueClassesTest = list()
        trueClassesTrain = list()
        for e,d in enumerate(testData):
            x = d[0]
            y = d[1]

            # Retrieve label for this input (x)
            for l in y:
                trueClassesTest.append(l[0])

            # If we've reached the end of our testing data, break
            if e == (test_num_batches - 1):
                break
        for e,d in enumerate(trainData):
            x = d[0]
            y = d[1]

            # Retrieve label for this input (x)
            for l in y:
                trueClassesTrain.append(l[0])

            # If we've reached the end of our training data, break
            if e == (train_num_batches - 1):
                break

        # Print AUC
        # NOTE: there are issues with doing this currently:
        # https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class#39703870
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#multiclass-settings
        # https://github.com/scikit-learn/scikit-learn/issues/3298
#       auc = roc_auc_score(trueClasses,predictClasses)
#       print 'AUC: {0}'.format(auc)

        # Print counts of each label for fold
        c = Counter(trueClassesTrain)
        print ''
        print 'Fold Indices/Counts (train dataset):'
        for e,l in enumerate(sorted(c.keys())):
            sys.stdout.write('Index: {0: <10} Class: {1: <20} Count: {2: <10} ({3:.4f}% of fold dataset)\n'.format(e,l,c[l],100*float(c[l])/sum(c.values())))

        # Print counts of each label for fold
        c = Counter(trueClassesTest)
        print ''
        print 'Fold Indices/Counts (test dataset):'
        for e,l in enumerate(sorted(c.keys())):
            sys.stdout.write('Index: {0: <10} Class: {1: <20} Count: {2: <10} ({3:.4f}% of fold dataset)\n'.format(e,l,c[l],100*float(c[l])/sum(c.values())))


        # Print confusion matrix
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        cf = confusion_matrix(trueClassesTest,predictClasses)
        print ''
        print 'Confusion Matrix (test dataset): (x-axis: Actual, y-axis: Predicted)'
        for x in cf:
            for y in x:
                sys.stdout.write('{0} '.format(y))
            sys.stdout.write('\n')
        print ''

        # Print TP/FP/FN/TN rates
        # A nice visual for determining these for the multi-class case:
        # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
        # https://stackoverflow.com/a/43331484
        # Convert confusion matrix to floats so we can have decimals
        cf = cf.astype(np.float32)
        FP = cf.sum(axis=0) - np.diag(cf)
        FN = cf.sum(axis=1) - np.diag(cf)
        TP = np.diag(cf)
        TN = cf.sum() - (FP + FN + TP)
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        FNR = FN/(FN+TP)
        FPR = FP/(FP+TN)
        ACC = (TP+TN)/(TP+TN+FP+FN)
        print 'Stats for each class (class is index in these arrays)'
        print 'TPR: {0}\nFPR: {1}\nFNR: {2}\nTNR: {3}\n'.format(list(TPR),list(FPR),list(FNR),list(TNR))
        print 'ACC: {0}\n'.format(list(ACC))

def usage():
    print 'usage: python lstm.py features/labels features/ models/ windowsize number_of_unique_api_calls'
    sys.exit(2)

def _main():
    if len(sys.argv) != 6:
        usage()

    label_fn = sys.argv[1]
    folder = sys.argv[2]
    model_folder = sys.argv[3]
    windowSize = int(sys.argv[4])
    numCalls = int(sys.argv[5])

    # Remove model folder if it already exists
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
        os.mkdir(model_folder)
    else:
        os.mkdir(model_folder)

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

    # Get label counts
    c = Counter([l for s,l in labels.iteritems()])

    # Lock in label ordering (and sort by popularity)
    labelMap = sorted(labelMap, key=lambda l: c[l], reverse=True)

    # Print class labels and counts
    print 'Total Dataset:'
    for e,l in enumerate(labelMap):
        sys.stdout.write('Class: {0: <10} Label: {1: <20} Count: {2: <10} ({3:.2f}% of dataset)\n'.format(e,l,c[l],100*float(c[l])/sum(c.values())))
    print ''

    # Train LSTM
    train_lstm(folder, sample, labels, labelMap, model_folder, windowSize, numCalls)

if __name__ == '__main__':
    _main()
