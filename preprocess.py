import sys
import os
import shutil
from multiprocessing import Pool, Lock

from collections import Counter

# Global list of api calls (for converting them to numbers)
apiMap = list()

# Semaphore for the apiMap
apiMapLock = Lock()

# Extracts features from samples
def get_sequences(folder, sample, label, labelMap, maxLen, feature_folder):
    global apiMapLock

    # Print out this sample's converted label to new file
    path = os.path.join(feature_folder,'labels')
    with open(path,'a') as fa:
        fa.write('{0} {1}\n'.format(sample,labelMap.index(label)))

    # To keep track of number of API calls
    count = 0

    path = os.path.join(folder,sample)
    newpath = os.path.join(feature_folder,sample)

    # Read in API call sequence and map each unique API call to a number
    with open(path,'r') as fr:
        # Open file to be outputted to
        with open(newpath,'a') as fa:

            for line in fr:
                line = line.strip('\n')
                count += 1

                apiMapLock.acquire()

                # Make sure list of API calls has no duplicates
                if not line in apiMap:
                    apiMap.append(line)

                # Get number of this API call
                # Doing +1 because we want the value zero to be our pad
                num = apiMap.index(line) + 1

                apiMapLock.release()

                # Write mapped API call number
                fa.write('{0}\n'.format(num))

    # Pad remaining sequences
    delta = maxLen - count
    if delta > 0:
        with open(newpath,'a') as fa:
            for i in range(delta):
                fa.write('0\n')

    # If no sequences for this sample, print this out
    if count == 0:
        return 'Sample {0} has no sequence'.format(sample)
    else:
        return None

def get_sequences_wrapper(args):
    return get_sequences(*args)

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

def extract_features(folder, samples, labels, feature_folder):
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

    # Create argument pools
    args = [(folder,s,l,labels,longest,feature_folder) for s,l in samples.iteritems()]

    # Convert API call sequences into numbers instead of strings AND
    # the malware family labels into numbers instead of strings and write
    # this data to a separate file
    # NOTE: We do this in parallel because this can take a long time

    count = 0
    zerocounts = ''

    pool = Pool(10)
    results = pool.imap_unordered(get_sequences_wrapper, args)
    for r in results:
        count += 1
        if r != None:
            zerocounts += r + '\n'

        sys.stdout.write('Extracting sample sequences: {0}\r'.format(count))
        sys.stdout.flush()
    pool.close()
    pool.join()

    sys.stdout.write('\n')
    sys.stdout.flush()

    print zerocounts

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

    folder = sys.argv[1]
    labels_fn = sys.argv[2]
    feature_folder = sys.argv[3]

    # Final set of labels
    final_labels = set()

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
    # Get final labels we'll only consider.
    # For example, only consider labels with at least 100 samples
    for l,c in counts.most_common():
        # Only consider families with >= 10 samples
        if c < 10:
            break
        else:
            final_labels.add(l)

    # Remove samples not with the labels we're interested in
    final_samples = { k:v for k,v in samples.iteritems() if v in final_labels }

    # Extract features of files in data folder
    # We convert "final_labels" into a list because we want the indices of the labels
    extract_features(folder, final_samples, list(final_labels), feature_folder)

if __name__ == '__main__':
    _main()
