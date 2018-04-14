import sys
import os
import cPickle as pkl
import numpy as np

from keras.models import model_from_json

from sklearn.metrics import confusion_matrix,roc_auc_score

def usage():
    print 'usage: python eval.py model.json weight.h5 train.pkl test.pkl'
    sys.exit(2)

def _main()
    if len(sys.argv) != 5
        usage()

    # Get parameters
    model_json = sys.argv[1]
    model_weights = sys.argv[2]
    train_fn = sys.argv[3]
    test_fn = sys.argv[4]

    # Read in LSTM  model
    with open(model_json,'r') as fr:
        lstm = keras.models.model_from_json(fr.read())

    # Load weights
    lstm.load_weights(model_weights)

    # Load training and testing data
    with open(train_fn, 'rb') as fr:
        trainData = pkl.load(fr)
    with open(test_fn, 'rb') as fr:
        test_num_batches = pkl.load(fr)
        testData = pkl.load(fr)

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
#   auc = roc_auc_score(trueClasses,predictClasses)
#   print 'AUC: {0}'.format(auc)

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

if __name__ == '__main__':
    _main()
