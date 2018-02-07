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

def sequence_generator(self, componentType, foldIDs):
    xSet, ySet = [], []
    batch_count = 0
    while True:
        for foldID in foldIDs:
            fold = self.generate_file(componentType, foldID + 1)

            foldX = fold.x
            foldY = fold.y
            # foldX, foldY = shuffle(foldX, foldY, random_state=0)

            for i in range(len(foldX)):
                xSet.append(foldX[i])
                ySet.append(foldY[i])
                batch_count += 1

                if batch_count >= self.BATCH_SIZE:
                    x = np.array(xSet)
                    y = np.array(ySet)

                    batch_count = 0
                    xSet, ySet = [], []
                    yield (x, y)

#TODO
def train_lstm(samples):
    nFolds = 10
    foldCount = 0

    networks = {}  # component -> LSTM network

    # extract sequences sets of evvents from all apps in test and train apps
    predictions = {}
    ground_truths = {}
    yHatVecs = {}
    yTruthVecs = {}

    #obtain an oerdering of the folds that were created
    folds = KFold(n_splits=nFolds, shuffle=True)
    foldCount = 0

    for train, test in folds:
        foldCount += 1
        print("Training", componentType, "for partition", str(foldCount) + "/" + str(self.nFolds))

        # Train the LSTM on N - 1 folds
        dataGen = self.sequence_generator(componentType, train)
        lstm = self.build_LSTM_model(dataGen, self.NUM_TRAIN_BATCHES[componentType])
        self.networks[componentType] = lstm
        #generating for test data
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

#TODO
def build_LSTM_model(self, dataGen, num_batches):
    hidden_layers = 100#int((self.MAX_SEQ_LEN + self.CLASS_COUNT) / 2) ** 2
    early_stop = cb.EarlyStopping(monitor='acc', min_delta = 0.0001, patience = 3)

    model = Sequential()
    model.add(
        LSTM(
            hidden_layers,
            input_shape=(self.MAX_SEQ_LEN, self.SEQ_VEC_LEN),
            return_sequences=False
            )
        )

    model.add(Dense(self.CLASS_COUNT, activation='softmax'))
    # model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['acc'])
    # print(model.summary())

    model.fit_generator(
        dataGen,
        steps_per_epoch = num_batches,
        epochs = self.MAX_EPOCHS,
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

    #TODO - sena
    # I want a list of sample feature locations given the sample names in
    # features/labels file and the location features/
    # I do not want you to read in the feature data

    # Read in labels and sample locations

    #TODO
    # Train LSTM
#   train_lstm(samples)

if __name__ == '__main__':
    _main()
