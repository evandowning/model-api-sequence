# Produces images of api sequences
import sys
import os
import numpy as np
import cPickle as pkl
import png
from struct import unpack
from multiprocessing import Pool

from hashlib import md5

# Converts integers representing api calls to something in between [0,255]
# Locality insensitive (good contrast)
# 3 because 3 channels
def api_md5(api):
    return unpack('BBB', md5(api).digest()[:3])

# Extract API sequences and convert them to pixels
def extract(folder,fn,num,width):
    label = None
    seq = list()

    # Read in sample's sequence
    path = os.path.join(folder,fn+'.pkl')
    with open(path,'rb') as fr:
        for i in range(num):
            t = pkl.load(fr)
            label = t[1]

            # Replace API call integers with pixel values
            seq.extend([api_md5(str(api)) for api in t[0]])

    # Pad array if it's not divisible by width (3 channels for RGB)
    r = len(seq) % (width*3)
    if r != 0:
        seq.extend([api_md5('0')]*(width*3-r))

    # Reshape numpy array (3 channels)
    data = np.reshape(np.array(seq), (-1,width*3))
    data = data.astype(np.int8)

    return fn,data,label

def extract_wrapper(args):
    return extract(*args)

def usage():
    print 'usage: python color.py /data/arsa/api-sequences-features/ images/ image.labels errors.txt'
    sys.exit(2)

def _main():
    if len(sys.argv) != 5:
        usage()

    feature_folder = sys.argv[1]
    output_folder = sys.argv[2]
    output_labels = sys.argv[3]
    output_errors = sys.argv[4]

    #TODO - make these parameters
    # Width of image
    width = 496

    # RBG color scheme (3 channels), 8-bits per channel
    fmt_str = 'RGB;8'

    # If output folder doesn't exist, create it
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Extract metadata
    metafn = os.path.join(feature_folder,'metadata.pkl')
    with open(metafn,'rb') as fr:
        # Window Size
        windowSize = pkl.load(fr)
        # Number of samples per label
        labelCount = pkl.load(fr)
        # Number of samples per data file (so we can determine folds properly)
        fileMap = pkl.load(fr)

    #TODO - make these parameters
    # Create argument pools (limit it to sequences < 100000 and only 10k of them
    args = [(feature_folder,fn,fileMap[fn],width) for fn in fileMap.keys() if fileMap[fn]*windowSize < 100000][:10000]

    # Extract images
    pool = Pool(20)
    results = pool.imap_unordered(extract_wrapper, args)

    # Write labels
    with open(output_labels,'w') as fo, open(output_errors,'w') as fe:
        for e,r in enumerate(results):
            sys.stdout.write('Extracting sample\'s traces: {0}/{1}\r'.format(e+1,len(args)))
            sys.stdout.flush()

            fn,data,label = r

            # Error occurred. Couldn't extract sequence
            if len(data) == 0:
                fe.write('{0}\n'.format(fn))
                continue

            # Write sequence to image file
            out_path = os.path.join(output_folder,fn+'.png')
            png.from_array(data, fmt_str).save(out_path)

            # Write label
            fo.write('{0} {1}\n'.format(fn,label))

    pool.close()
    pool.join()

    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    _main()
