#!/usr/bin/env python3

import sys
import os
import pickle as pkl
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
    sys.stderr.write('usage: python evaluation.py model.json weight.h5 features/ hash.label labels.txt predictions.csv [convert_classes.txt]\n')
    sys.exit(2)

def _main():
    if (len(sys.argv) != 7) and (len(sys.argv) != 8):
        usage()

    # Get parameters
    model_json = sys.argv[1]
    model_weights = sys.argv[2]
    feature_folder = sys.argv[3]
    sample_fn = sys.argv[4]
    label_fn = sys.argv[5]
    prediction_fn = sys.argv[6]

    convert = dict()
    if len(sys.argv) == 8:
        convert_fn = sys.argv[7]
        with open(convert_fn,'r') as fr:
            for line in fr:
                line = line.strip('\n')
                k,v = line.split()
                convert[int(k)] = int(v)

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

    numSamples = len(list(fileMap.keys()))

    sys.stdout.write('WindowSize: {0}\n'.format(windowSize))
    sys.stdout.write('Number of samples: {0}\n'.format(numSamples))

    classes = set()

    # For stats at end
    predictClasses = np.array([])
    trueClasses = np.array([])

    # Open predictions file
    with open(prediction_fn,'w') as fw:
        fw.write('Hash,Label,Prediction,Count\n')

        # For each sample
        for k,v in fileMap.items():
            # If we don't care about this sample
            if k not in list(sampleMap.keys()):
                continue

            # Get real label
            realLabel = labelMap[sampleMap[k]]

            sys.stdout.write('Sample: {0} | Label: {1} ({2}) | Subsequences: {3}'.format(k,sampleMap[k],realLabel,v))
            sys.stdout.flush()

            # Get subsequences
            xdata,ydata = sequence_generator(os.path.join(feature_folder,k)+'.pkl',v)

            # If no sequences
            if len(xdata) == 0:
                sys.stdout.write(' | ERROR, sequence length 0\n')
                continue

            # If it's just one sequence, restructure array
            if xdata.shape == (32,):
                xdata = xdata.reshape((1,32))
                ydata = np.array([ydata])

            # https://keras.io/models/model/#predict_on_batch
            p = lstm.predict_on_batch(xdata)

            # Extract predicted classes for each sample in data
            # https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
            predict = p.argmax(axis=-1)

            # Replace labels
            tmp = np.copy(predict)
            for k3,v3 in convert.items(): tmp[predict==k3] = v3
            predict = np.copy(tmp)

            sys.stdout.write(' | Predicted classes: {0}\n'.format(Counter(predict).most_common()))
            sys.stdout.flush()

            # Write prediction to file
            predictionCounter = Counter(predict)
            for k2,v2 in predictionCounter.most_common():
                # From: https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary#13149770
                fw.write('{0},{1},{2},{3}\n'.format(k,sampleMap[k],list(labelMap.keys())[list(labelMap.values()).index(k2)],v2))

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
#   TPR = TP/(TP+FN)
    TPR = np.divide(TP,TP+FN, out=np.ones_like(TP+FN)*-1, where=(TP+FN)!=0)
#   FPR = FP/(FP+TN)
    FPR = np.divide(FP,FP+TN, out=np.ones_like(FP+TN)*-1, where=(FP+TN)!=0)
#   FNR = FN/(FN+TP)
    FNR = np.divide(FN,FN+TP, out=np.ones_like(FN+TP)*-1, where=(FN+TP)!=0)
#   TNR = TN/(TN+FP)
    TNR = np.divide(TN,TN+FP, out=np.ones_like(TN+FP)*-1, where=(TN+FP)!=0)
    ACC = (TP+TN)/(TP+TN+FP+FN)

    sys.stdout.write('Classes: {0}\n'.format(sorted(classes)))

    sys.stdout.write('Stats for each class (class is index in these arrays)\n')
    sys.stdout.write('TPR: {0}\n\nFPR: {1}\n\nFNR: {2}\n\nTNR: {3}\n\n\n'.format(list(TPR),list(FPR),list(FNR),list(TNR)))
    sys.stdout.write('ACC: {0}\n\n'.format(list(ACC)))
    return

    # https://stackoverflow.com/questions/46861966/how-to-find-loss-values-using-keras
    # https://keras.io/losses/#sparse_categorical_crossentropy
    # Print out loss and accuracy
    t = trueClasses.reshape((1,len(trueClasses),1))[0]
    y_true = K.variable(t)

    p = predictClasses.reshape((1,len(predictClasses),1))[0]
    y_pred = K.variable(p)

    print(t,y_true)
    print(p,y_pred)

    #TODO - fix this bug
    error = K.eval(losses.sparse_categorical_crossentropy(y_true, y_pred))
    acc = K.eval(metrics.sparse_categorical_accuracy(y_true,y_pred))

    avgerror = sum(error) / float(len(error))
    avgacc = sum(acc) / float(len(acc))

    print('Average loss: {0}'.format(avgerror))
    print('Average accuracy: {0}'.format(avgacc))

if __name__ == '__main__':
    _main()
