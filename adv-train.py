import sys
import os
import shutil
import cPickle as pkl
import numpy as np
import math
from multiprocessing import Pool

from keras.models import model_from_json
from keras import optimizers
from keras import callbacks as cb

# Trains on per sample (i.e., file)
def train_sequence_generator(folder, batchSize):
    # We want to loop infinitely because we're training our data on multiple epochs in build_LSTM_model()
    while 1:
        xSet = np.array([])
        ySet = np.array([])

        num = 0
        # For each sample
        for f in os.listdir(folder):
            # Extract metadata
            metafn = os.path.join(folder,f,'metadata.pkl')
            with open(metafn,'rb') as fr:
                # Window Size
                windowSize = pkl.load(fr)
                # Number of samples per label
                labelCount = pkl.load(fr)
                # Number of samples per data file
                fileMap = pkl.load(fr)

            # For each attack sample
            for fn,numSeq in fileMap.iteritems():
                path = os.path.join(folder,f,fn+'.pkl')
                with open(path, 'rb') as fr:
                    for i in range(numSeq):
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
                            num = 0

                            yield (xSet, ySet)

                            xSet = np.array([])
                            ySet = np.array([])


        # Yield remaining set
        if len(xSet) > 0:
            yield (xSet, ySet)

# Tests on original testing data
def test_sequence_generator(test_pkl, batchSize):
    # We want to loop infinitely because we're testing our data on multiple epochs in build_LSTM_model()
    while 1:
        xSet = np.array([])
        ySet = np.array([])

        num = 0
        with open(test_pkl, 'rb') as fr:
            # First entry is number of batches
            n = int(pkl.load(fr))

            # For each batch
            for i in range(n):
                # For each testing sample
                t = pkl.load(fr)
                xBatch = t[0]
                yBatch = t[1]
                for e,x in enumerate(xBatch):
                    y = yBatch[e]

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
                        num = 0

                        yield (xSet, ySet)

                        xSet = np.array([])
                        ySet = np.array([])

        # Yield remaining set
        if len(xSet) > 0:
            yield (xSet, ySet)

# Builds LSTM model
def build_LSTM_model(model, trainData, trainBatches, testData, testBatches):
    # https://keras.io/callbacks/#earlystopping
    early_stop = cb.EarlyStopping(monitor='sparse_categorical_accuracy', min_delta = 0.0001, patience = 3)

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
        use_multiprocessing = False,
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
def train_lstm(model, folder, test_pkl, model_folder):
    # Doesn't have to divide data evenly
    batchSize = 3000

    # Put features into format LSTM can ingest
    trainData = train_sequence_generator(folder, batchSize)
    testData = test_sequence_generator(test_pkl, batchSize)

    sys.stdout.write('Calculating number of batches...')
    sys.stdout.flush()

    # Calculate number of batches
    numTrainSeq = 0
    for f in os.listdir(folder):
        metafn = os.path.join(folder,f,'metadata.pkl')
        with open(metafn,'rb') as fr:
            # Window Size
            windowSize = pkl.load(fr)
            # Number of samples per label
            labelCount = pkl.load(fr)
            # Number of samples per data file (so we can determine folds properly)
            fileMap = pkl.load(fr)

        # Add number of sequences of each sample
        for k,v in fileMap.iteritems():
            numTrainSeq += v

    with open(test_pkl, 'rb') as fr:
        # First entry is number of batches when this test data was saved
        test_num_batches = int(pkl.load(fr))
        t = pkl.load(fr)
        numTestSeq = len(t[0])*test_num_batches

    train_num_batches = math.ceil(float(numTrainSeq)/batchSize)
    test_num_batches = math.ceil(float(numTestSeq)/batchSize)

    sys.stdout.write('Done\n')
    sys.stdout.flush()

    print 'Number of training sequences: {0}'.format(numTrainSeq)
    print 'Number of testing sequences: {0}'.format(numTestSeq)
    print 'Training batches: {0}'.format(train_num_batches)
    print 'Testing batches: {0}'.format(test_num_batches)

    # Train LSTM model
    lstm,hist = build_LSTM_model(model, trainData, train_num_batches, testData, test_num_batches)
    # Print accuracy histories over the folds
#   print ''
#   print hist.history

    # Save trained model
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # Convert model to JSON format to be stored
    modelJSON = lstm.to_json()
    # Store model in model_folder
    fn = os.path.join(model_folder,'model.json')
    with open(fn,'w') as fw:
        fw.write(modelJSON)
    # Store weights for model
    fn = os.path.join(model_folder,'weight.h5')
    lstm.save_weights(fn)

    # Save train/test fold to be used by eval.py
    fn = os.path.join(model_folder,'train.pkl')
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

    fn = os.path.join(model_folder,'test.pkl')
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

def usage():
    print 'usage: python adv-train.py model.json weights.h5 test.pkl adversarial-features/ adversarial-models/'
    sys.exit(2)

def _main():
    if len(sys.argv) != 6:
        usage()

    model_json = sys.argv[1]
    model_weights = sys.argv[2]
    test_pkl = sys.argv[3]
    feature_folder = sys.argv[4]
    adv_model_folder = sys.argv[5]

    # Remove model folder if it already exists
    if os.path.exists(adv_model_folder):
        shutil.rmtree(adv_model_folder)
        os.mkdir(adv_model_folder)
    else:
        os.mkdir(adv_model_folder)

    sys.stdout.write('Loading model...')
    sys.stdout.flush()

    # Read in LSTM  model
    with open(model_json,'r') as fr:
        lstm = model_from_json(fr.read())

    # Load weights
    lstm.load_weights(model_weights)

    sys.stdout.write('Done\n')
    sys.stdout.flush()

    # Train LSTM
    train_lstm(lstm, feature_folder, test_pkl, adv_model_folder)

if __name__ == '__main__':
    _main()
