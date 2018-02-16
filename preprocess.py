import sys
import os
import shutil
from multiprocessing import Pool
import cPickle as pkl

from collections import Counter

# Global list of api calls (for converting them to numbers)
apiMap = list()

# Extracts features from samples
def get_sequences(folder, sample, maxLen, feature_folder):
    global apiMap

    # To keep track of number of API calls
    count = 0

    path = os.path.join(folder,sample)
    newpath = os.path.join(feature_folder,sample + '.pkl')

    seq = list()

    # Read in API call sequence and map each unique API call to a number
    with open(path,'r') as fr:
        for line in fr:
            line = line.strip('\n')

            # Add API call if we've never seen it before
            if not line in apiMap:
                apiMap.append(line)

            # Get number of this API call
            # Doing +1 because we want the value zero to be our pad
            num = apiMap.index(line) + 1

            # Add number to list
            seq.append(num)

            # If sequence is longer than maxLen, break.
            # This is used to truncate data
            if len(seq) == maxLen:
                break

    # If no sequences for this sample, print this out
    if len(seq) == 0:
        return 'Sample {0} has no sequence'.format(sample)
    else:
        # Pad remaining sequences. Since we +1 all indices of the API call, the
        # number 0 is left for us to use as padding.
        # Padding can sound like a stupid idea, but it's been proven to work
        # and be used by others:
        # https://github.com/keras-team/keras/issues/2375
        # https://stackoverflow.com/questions/34670112/how-to-deal-with-batches-with-variable-length-sequences-in-tensorflow#34675264
        # https://www.tensorflow.org/versions/master/tutorials/seq2seq
        # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
        # https://www.mathworks.com/help/nnet/examples/classify-sequence-data-using-lstm-networks.html?requestedDomain=true

        # TODO: Maybe look into future solutions which don't require padding?
        #       But I believe we can't use batching at that point (batching is
        #       training the LSTM with batches of samples instead of one sample
        #       at a time, which can save us a lot of time).
        # https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/
        # https://github.com/keras-team/keras/issues/6776
        # https://stackoverflow.com/questions/46353720/how-to-process-a-large-image-in-keras

        delta = maxLen - len(seq)
        if delta > 0:
            seq.extend([0]*delta)

        # Write list of sequences to new file
        with open(newpath,'wb') as fw:
            pkl.dump(seq,fw)

        return None

# Get length of sequence of a sample
def get_length(folder, sample):
    rv = 0

    # Read file
    path = os.path.join(folder,sample)
    with open(path,'r') as fr:
        for line in fr:
            rv += 1

    return rv

def get_length_wrapper(args):
    return get_length(*args)

def extract_features(folder, samples, label_fn, feature_folder):
    print 'Samples to extract: {0}'.format(len(samples.keys()))

    print 'Getting longest API sequence...'

    # Create argument pools
    args = [(folder,s) for s,l in samples.iteritems()]

    # Determine longest API sequence length
    longest = 0
    pool = Pool(10)
    results = pool.imap_unordered(get_length_wrapper, args)
    for r in results:
        if r > longest:
            longest = r
    pool.close()
    pool.join()

    print 'Longest sequence: {0}'.format(longest)

    #TODO - Truncate datasets more intelligently
    # https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/
    longest = 1000
    print 'Truncating to length: {0}'.format(longest)

    # Create argument pools
    args = [(folder,s,longest,feature_folder) for s,l in samples.iteritems()]

    # Convert API call sequences into lists of numbers and write to file
    # NOTE: We don't do this next step in parallel because it can take a long time.
    #       The multiple processes have to share apiMap which can introduce
    #       large overheads in IPC

    count = 0
    errors = ''

    for e,t in enumerate(samples.iteritems()):
        s = t[0]
        l = t[1]

        rv = get_sequences(folder,s,longest,feature_folder)

        # Some error happened
        if rv != None:
            errors += rv + '\n'
        # Success
        else:
            # Write this sample/label to file
            with open(label_fn,'a') as fa:
                fa.write('{0} {1}\n'.format(s,l))

        sys.stdout.write('Extracting sample sequences: {0}\r'.format(e+1))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    # Write out errors to file
    with open('errors.txt','w') as fw:
        fw.write(errors)

# Extracts labels from file
def get_labels(fn):
    rv = dict()

    with open(fn,'r') as fr:
        for line in fr:
            line = line.strip('\n')

            h = line.split('\t')[0]
            label = line.split('\t')[-1]

            rv[h] = label

    return rv

def usage():
    print 'usage: python preprocess.py api-sequence-folder/ hash.labels features-folder/'
    sys.exit(2)

def _main():
    if len(sys.argv) != 4:
        usage()

    global apiMap

    folder = sys.argv[1]
    labels_fn = sys.argv[2]
    feature_folder = sys.argv[3]

    # Remove the feature folder if it already exists
    if os.path.exists(feature_folder):
        shutil.rmtree(feature_folder)
        os.mkdir(feature_folder)
    else:
        os.mkdir(feature_folder)

    print 'Read in labels of all samples...'

    # Read in labels of all samples
    samples = get_labels(labels_fn)
    # Get counts of labels
    counts = Counter(samples.values())

    final_labels = set()

    #TODO - 10 is an arbitrary number. I.e., I don't want to train my LSTM
    #       on any singletons (labels with just one sample), so I chose to
    #       limit choosing labels with at least 10 samples within them.
    #       Is there a smarter way to do this?
    # Get final labels we'll only consider.
    # For example, only consider labels with at least 100 samples
    for l,c in counts.most_common():
        # Only consider families with >= 10 samples
        if c < 10:
            break
        else:
            final_labels.add(l)

    # Remove samples with the labels we're not interested in (i.e., not in "final_labels")
    final_samples = { k:v for k,v in samples.iteritems() if v in final_labels }

    label_fn = os.path.join(feature_folder,'labels')

    # Extract features of files in data folder
    extract_features(folder, final_samples, label_fn, feature_folder)

    print 'Number of unique API calls found: {0}'.format(len(apiMap))

if __name__ == '__main__':
    _main()
