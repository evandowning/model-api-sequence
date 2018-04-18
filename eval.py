import sys
import os
import cPickle as pkl
import numpy as np
from collections import Counter

from keras.models import model_from_json

from keras import backend as K
from keras import losses
from keras import metrics
from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix,roc_auc_score

# Retrieves stats from datasets
def stats(lstm,data,num_batches):
    # Run predictions over data
    # https://keras.io/models/model/#predict_generator
    p = lstm.predict_generator(data, steps=num_batches, use_multiprocessing=True)

    # Extract predicted classes for each sample in data
    # https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
    predictClasses = p.argmax(axis=-1)

    # NOTE: Sloppy way of doing this, but it'll work for now
    # Extract true labels for training and testing data
    trueClasses = list()
    for d in data:
        x = d[0]
        y = d[1]

        # Retrieve label for this input (x)
        for l in y:
            trueClasses.append(l[0])

    # Print counts of each label for fold
    c = Counter(trueClasses)
    print ''
    print 'Class Indices/Labels/Counts (for this dataset):'
    for e,l in enumerate(sorted(c.keys())):
        sys.stdout.write('Index: {0: <10} Class: {1: <20} Count: {2: <10} ({3:.4f}% of fold dataset)\n'.format(e,l,c[l],100*float(c[l])/sum(c.values())))

    # Print AUC
    # NOTE: there are issues with doing this currently:
    # https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class#39703870
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#multiclass-settings
    # https://github.com/scikit-learn/scikit-learn/issues/3298
#   auc = roc_auc_score(trueClasses,predictClasses)
#   print 'AUC: {0}'.format(auc)

    # Print confusion matrix
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    cf = confusion_matrix(trueClasses,predictClasses)
    print ''
    print 'Confusion Matrix (dataset): (x-axis: Actual, y-axis: Predicted)'
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
    print 'TPR: {0}\n\nFPR: {1}\n\nFNR: {2}\n\nTNR: {3}\n\n'.format(list(TPR),list(FPR),list(FNR),list(TNR))
    print 'ACC: {0}\n'.format(list(ACC))

    # https://stackoverflow.com/questions/46861966/how-to-find-loss-values-using-keras
    # https://keras.io/losses/#sparse_categorical_crossentropy
    # Print out loss and accuracy
    trueClasses = np.array(trueClasses)
    t = trueClasses.reshape((1,len(trueClasses),1))[0]
    y_true = K.variable(t)
    y_pred = K.variable(p)

    error = K.eval(losses.sparse_categorical_crossentropy(y_true, y_pred))
    acc = K.eval(metrics.sparse_categorical_accuracy(y_true,y_pred))

    avgerror = sum(error) / float(len(error))
    avgacc = sum(acc) / float(len(acc))

    print 'Average loss: {0}'.format(avgerror)
    print 'Average accuracy: {0}'.format(avgacc)

def sequence_generator(fn):
    with open(fn,'rb') as fr:
        # First entry is number of entries
        n = int(pkl.load(fr))

        for i in range(n):
            yield pkl.load(fr)

def usage():
    print 'usage: python eval.py model.json weight.h5 train.pkl test.pkl'
    sys.exit(2)

def _main():
    if len(sys.argv) != 5:
        usage()

    # Get parameters
    model_json = sys.argv[1]
    model_weights = sys.argv[2]
    train_fn = sys.argv[3]
    test_fn = sys.argv[4]

    # Read in LSTM  model
    with open(model_json,'r') as fr:
        lstm = model_from_json(fr.read())

    # Load weights
    lstm.load_weights(model_weights)

    # Print out summary of model
    # https://keras.io/models/about-keras-models/
    lstm.summary()

    print 'Loading in data...'

    # Load training and testing metadata
    with open(train_fn, 'rb') as fr:
        train_num_batches = int(pkl.load(fr))
    with open(test_fn, 'rb') as fr:
        test_num_batches = int(pkl.load(fr))

    trainData = sequence_generator(train_fn)
    testData = sequence_generator(test_fn)

    print 'Training Data:'

    # Get stats about training data
    stats(lstm,trainData,train_num_batches)

    print '================================='

    print 'Testing Data:'

    # Get stats about testing data
    stats(lstm,testData,test_num_batches)

if __name__ == '__main__':
    _main()
