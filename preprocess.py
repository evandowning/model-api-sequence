import sys
import os
import shutil
from multiprocessing import Pool
import cPickle as pkl
import numpy as np

from collections import Counter

# Map of API call to its integer representation
apiMap = dict()

# Map to keep track of mapping between labels and their unique integers
labelMap = dict()

# Extracts features from raw data file
def extract(folder,sample,label,feature_folder,windowSize,task):
    global apiMap
    global labelMap

    # Keep count of each label
    labelCount = Counter()

    # Get sample path
    path = os.path.join(folder,sample)

    # If path doesn't exist, return
    if not os.path.exists(path):
        return 'Path doesn\'t exist.',None,None,None,None

    # Read in entire file
    with open(path,'r') as fr:
        seq = fr.read()

    # If nothing was read in, return an error
    if len(seq) == 0:
        error = 'Sample {0} has no sequence\n'.format(sample)
        return error,None,None,None,None

    seq = seq.strip('\n')
    seq = seq.split('\n')
    seq = [api.split(' ')[1] for api in seq]
    seq = np.array(seq)

    # Get original length (for stats)
    original = len(seq)

    # If sequence length is less than windowSize, we cannot do prediction
    if task == 'regression':
        if len(seq) <= windowSize:
            error = 'Sample {0} has sequence size <= to window size\n'.format(sample)
            return error,None,None,None,None

    # Replace API calls with their unique integer value
    # https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array#3404089
    newseq = np.copy(seq)
    for k,v in apiMap.iteritems():
        newseq[seq==k] = v
    seq = newseq

    # Pad sequence to be evenly divisible by window size if necessary
    remainder = len(seq) % windowSize
    if remainder > 0:
        seq = np.append(seq,[0]*remainder)

    # Convert numpy array from array of strings to array of integers
    seq = seq.astype('int64')

    # Calculate window indices: https://stackoverflow.com/questions/15722324/sliding-window-in-numpy/42258242#42258242
    index = np.arange(windowSize)[None,:] + windowSize*np.arange(len(seq)/windowSize)[:,None]

    # Count number of sequences extracted for this sample
    numExtracted = 0

    # New file path to put features into
    newpath = os.path.join(feature_folder,sample+'.pkl')

    # If this is classification, we have to map the label to its integer representation
    if task == 'classification':
        label = labelMap[label]

    with open(newpath,'wb') as fw:
        # Insert each windowed sequence into the features file
        for e,w in enumerate(seq[index]): 
            # If this is regression, we have to extract the next API call after the sequence
            if task == 'regression':
                # For the last sequence, we won't be able to get the next API call
                if (e == len(index)-1) and (len(index) > 1):
                    break
                label = seq[e*windowSize+windowSize]

            # Increment how many sequences we've extracted
            numExtracted += 1

            # Count number of samples extracted per label
            if label not in labelCount:
                labelCount[label] = 0
            labelCount[label] += 1

            # Insert sequence/label into file
            pkl.dump((w,label),fw)

    return None,sample,numExtracted,labelCount,original

def extract_wrapper(args):
    return extract(*args)

# Extracts samples & labels (i.e., malware family) from file
def get_labels(folder,fn):
    rv = dict()

    with open(fn,'r') as fr:
        for line in fr:
            line = line.strip('\n')
            h = line.split('\t')[0]
            label = line.split('\t')[-1]
            rv[h] = label

    return rv

def usage():
    print 'usage: python preprocess.py api-sequence-folder/ api.txt label.txt hash.labels features-folder/ windowSize {classification | regression}'
    print ''
    print '    classification: classes are malware family label'
    print '    regression: classes are the next API call in the sequence immediately after the sliding window'
    print ''
    sys.exit(2)

def _main():
    if len(sys.argv) != 8:
        usage()

    global apiMap
    global labelMap

    folder = sys.argv[1]
    api_fn = sys.argv[2]
    label_fn = sys.argv[3]
    samples_fn = sys.argv[4]
    feature_folder = sys.argv[5]
    windowSize = int(sys.argv[6])
    task = sys.argv[7]

    # Test task parameter
    if (task != 'classification') and (task != 'regression'):
        print 'Error. "{0}" parameter is not "classification" or "regression"'.format(task)
        usage()

    # Remove the feature folder if it already exists
    if os.path.exists(feature_folder):
        shutil.rmtree(feature_folder)
        os.mkdir(feature_folder)
    else:
        os.mkdir(feature_folder)

    print 'Reading in api.txt file...'

    # Create a map between API calls and their integer representation
    apiMap = dict()
    with open(api_fn,'r') as fr:
        for e,line in enumerate(fr):
            line = line.strip('\n')
            # e+1 because we want 0 to be our padding integer
            apiMap[line] = e+1

    print 'Reading in label.txt file...'

    # Create a map between malware family label and their integer representation
    labelMap = dict()
    with open(label_fn,'r') as fr:
        for e,line in enumerate(fr):
            line = line.strip('\n')
            labelMap[line] = e

    print 'Reading in samples to preprocess...'

    # Read samples to preprocess
    samples = get_labels(folder,samples_fn)

    # If this is a classification problem, then potentially filter out classes
    # with fewer samples
    if task == 'classification':
        # Get counts of labels
        counts = Counter(samples.values())

        #TODO - 10 is an arbitrary number. I.e., I don't want to train my LSTM
        #       on any singletons (labels with just one sample), so I chose to
        #       limit choosing labels with at least 10 samples within them.
        #       Is there a smarter way to do this?

        # Get final labels we'll only consider.
        # For example, only consider labels with at least 100 samples
        final_labels = set()
        for l,c in counts.most_common():
            if c < 10:
                break
            else:
                final_labels.add(l)

        # Remove samples with the labels we're not interested in (i.e., not in "final_labels")
        final_samples = { k:v for k,v in samples.iteritems() if v in final_labels }

        # Remove samples with labels we don't have in "label.txt"
        final_samples = { k:v for k,v in samples.iteritems() if v in labelMap }

    # If this is a regression problem, we don't need to do this
    elif task == 'regression':
        final_samples = samples

    print 'Window Size: {0}'.format(windowSize)

    # Remove error file
    errorfn = 'errors.txt'
    if os.path.exists(errorfn):
        os.remove(errorfn)

    # Variable to keep track of number of sequences in each file
    fileMap = dict()

    # Variable to keep count of number of samples per label
    labelCount = Counter()

    # Create argument pools
    args = [(folder,s,l,feature_folder,windowSize,task) for s,l in final_samples.iteritems()]

    # Extract features of samples
    malTotal = 0
    malLongest = 0
    malShortest = -1
    malNum = 0
    malExtracted = 0
    malClasses = set()

    benTotal = 0
    benLongest = 0
    benShortest = -1
    benNum = 0
    benExtracted = 0

    pool = Pool(20)
    results = pool.imap_unordered(extract_wrapper, args)
    for e,r in enumerate(results):
        error,sample,numExtracted,lc,c = r

        sys.stdout.write('Extracting sample\'s traces: {0}/{1}\r'.format(e+1,len(final_samples.keys())))
        sys.stdout.flush()

        # If there was an error
        if error != None:
            with open(errorfn,'a') as fa:
                fa.write(error)
            continue

        # Keep track of number of extracted sequences for each sample
        fileMap[sample] = numExtracted

        # Keep count of number of samples per label
        labelCount = labelCount + lc

        # If this is a benign sample
        if samples[sample] == 'benign':
            # Increment extracted counter
            benExtracted += numExtracted

            # Keep track of length of the entire sequence of this sample
            if benShortest == -1:
                benShortest = c
            if c > benLongest:
                benLongest = c
            if c < benShortest:
                benShortest = c
            benTotal += c
            benNum += 1
        # Else if this is a malicious sample
        else:
            # Keep track of number of families
            malClasses.add(samples[sample])

            # Increment extracted counter
            malExtracted += numExtracted

            # Keep track of length of the entire sequence of this sample
            if malShortest == -1:
                malShortest = c
            if c > malLongest:
                malLongest = c
            if c < malShortest:
                malShortest = c
            malTotal += c
            malNum += 1

    pool.close()
    pool.join()

    sys.stdout.write('\n')
    sys.stdout.flush()

    print ''
    print 'Benign: (not counting erroneous traces)'
    print 'Total number of PE samples extracted from: {0}'.format(benNum)
    print 'Total number of subsequences extracted: {0}'.format(benExtracted)
    print 'Longest trace length: {0}'.format(benLongest)
    print 'Shortest trace length which is > 0: {0}'.format(benShortest)
    print 'Average trace length: {0:.2f}'.format(benTotal/float(benNum))

    print ''
    print 'Malicious: (not counting erroneous traces)'
    print 'Total number of PE samples extracted from: {0}'.format(malNum)
    print 'Number of malware families: {0}'.format(len(malClasses))
    print 'Total number of subsequences extracted: {0}'.format(malExtracted)
    print 'Longest trace length: {0}'.format(malLongest)
    print 'Shortest trace length which is > 0: {0}'.format(malShortest)
    print 'Average trace length: {0:.2f}'.format(malTotal/float(malNum))

    # Metadata file for lstm.py
    metafn = os.path.join(feature_folder,'metadata.pkl')
    with open(metafn,'wb') as fw:
        # Window Size
        pkl.dump(windowSize,fw)
        # Number of samples per label
        pkl.dump(labelCount,fw)
        # Number of subsequences per PE trace (so we can determine folds properly)
        pkl.dump(fileMap,fw)

if __name__ == '__main__':
    _main()
