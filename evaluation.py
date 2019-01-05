import sys
import os
import cPickle as pkl
import numpy as np
from collections import Counter

from keras.models import model_from_json
from keras import backend as K
from keras import losses
from keras import metrics

from sklearn.metrics import confusion_matrix,roc_auc_score

def sequence_generator(fn,n):
    xSet = np.array([])
    ySet = np.array([])

    x = np.array([])
    y = np.array([])

    num = 0

    # Read in sample's sequences
    with open(fn, 'rb') as fr:
        for e in enumerate(range(n)):
            t = pkl.load(fr)
            x = t[0]
            y = t[1]

            if len(xSet) == 0:
                xSet = x
                ySet = y
            else:
                xSet = np.vstack([xSet,x])
                ySet = np.append(ySet,y)

    return xSet,ySet

def usage():
    print 'usage: python evaluation.py model.json weight.h5 features/ hash.label labels.txt predictions.csv'
    sys.exit(2)

def _main():
    if len(sys.argv) != 7:
        usage()

    # Get parameters
    model_json = sys.argv[1]
    model_weights = sys.argv[2]
    feature_folder = sys.argv[3]
    sample_fn = sys.argv[4]
    label_fn = sys.argv[5]
    prediction_fn = sys.argv[6]

    # Load model
    with open(model_json,'r') as fr:
        lstm = model_from_json(fr.read())
    # Load weights
    lstm.load_weights(model_weights)

    # Print out summary of model
    # https://keras.io/models/about-keras-models/
    lstm.summary()

    # Create a map between malware family label and their integer representation
    sampleMap = dict()
    with open(sample_fn,'r') as fr:
        for line in fr:
            line = line.strip('\n')
            s,c = line.split('\t')
            sampleMap[s] = c

    # Create a map between malware family label and their integer representation
    labelMap = dict()
    with open(label_fn,'r') as fr:
        for e,line in enumerate(fr):
            line = line.strip('\n')
            labelMap[line] = e

    # Extract metadata
    metafn = os.path.join(feature_folder,'metadata.pkl')
    with open(metafn,'rb') as fr:
        # Window Size
        windowSize = pkl.load(fr)
        # Number of samples per label
        labelCount = pkl.load(fr)
        # Number of samples per data file (so we can determine folds properly)
        fileMap = pkl.load(fr)

    numSamples = len(fileMap.keys())

    print 'WindowSize: {0}'.format(windowSize)
    print 'Number of samples: {0}'.format(numSamples)

    classes = set()

    # For stats at end
    predictClasses = np.array([])
    trueClasses = np.array([])

    # Open predictions file
    with open(prediction_fn,'w') as fw:
        fw.write('Hash,Label,Prediction,Count\n')

        # For each sample
        for k,v in fileMap.iteritems():
            # If we don't care about this sample
            if k not in sampleMap.keys():
                continue

            # Get real label
            realLabel = labelMap[sampleMap[k]]

            sys.stdout.write('Sample: {0} | Label: {1} ({2}) | Subsequences: {3}'.format(k,sampleMap[k],realLabel,v))
            sys.stdout.flush()

            # Get subsequences
            xdata,ydata = sequence_generator(os.path.join(feature_folder,k)+'.pkl',v)

            # https://keras.io/models/model/#predict_on_batch
            p = lstm.predict_on_batch(xdata)

            # Extract predicted classes for each sample in data
            # https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
            predict = p.argmax(axis=-1)

            sys.stdout.write(' | Predicted classes: {0}\n'.format(Counter(predict).most_common()))
            sys.stdout.flush()

            # Write prediction to file
            predictionCounter = Counter(predict)
            for k2,v2 in predictionCounter.most_common():
                # From: https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary#13149770
                fw.write('{0},{1},{2},{3}\n'.format(k,sampleMap[k],labelMap.keys()[labelMap.values().index(k2)],v2))

            if len(classes) == 0:
                classes = set(ydata)
            classes.update(ydata)
            classes.update(predict)

            # Append data
            if len(predictClasses) == 0:
                predictClasses = np.array(predict)
                trueClasses = np.array(ydata)
            else:
                predictClasses = np.append(predictClasses,predict)
                trueClasses = np.append(trueClasses,ydata)

    # Print stats
    cf = confusion_matrix(trueClasses,predictClasses)

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

    print 'Classes: {0}'.format(sorted(classes))

    print 'Stats for each class (class is index in these arrays)'
    print 'TPR: {0}\n\nFPR: {1}\n\nFNR: {2}\n\nTNR: {3}\n\n'.format(list(TPR),list(FPR),list(FNR),list(TNR))
    print 'ACC: {0}\n'.format(list(ACC))
    return

    # https://stackoverflow.com/questions/46861966/how-to-find-loss-values-using-keras
    # https://keras.io/losses/#sparse_categorical_crossentropy
    # Print out loss and accuracy
    t = trueClasses.reshape((1,len(trueClasses),1))[0]
    y_true = K.variable(t)

    p = predictClasses.reshape((1,len(predictClasses),1))[0]
    y_pred = K.variable(p)

    print t,y_true
    print p,y_pred

    #TODO - fix this bug
    error = K.eval(losses.sparse_categorical_crossentropy(y_true, y_pred))
    acc = K.eval(metrics.sparse_categorical_accuracy(y_true,y_pred))

    avgerror = sum(error) / float(len(error))
    avgacc = sum(acc) / float(len(acc))

    print 'Average loss: {0}'.format(avgerror)
    print 'Average accuracy: {0}'.format(avgacc)

if __name__ == '__main__':
    _main()
