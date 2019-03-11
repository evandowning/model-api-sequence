#!/usr/bin/env python3

import sys
import os
from multiprocessing import Pool
import random
from collections import Counter

from datasketch import MinHash, MinHashLSH

# Reads file sequences into list
def readFile(folder,fn):
    rv = list()

    with open(os.path.join(folder,fn),'r') as fr:
        for line in fr:
            line = line.strip('\n')
            rv.append(line)

    return rv

# Extracts samples & labels (i.e., malware family) from file
def get_labels(folder,fn):
    rv = dict()

    with open(fn,'r') as fr:
        for line in fr:
            line = line.strip('\n')
            h = line.split('\t')[0]
            label = line.split('\t')[-1]

            # If this sample contains no data, ignore it
            if os.path.getsize(os.path.join(folder,h)) == 0:
                continue

            if label not in rv:
                rv[label] = list()

            rv[label].append(h)

    return rv

def usage():
    print('usage: python sim_stats.py /data/arsa/api-sequences /data/arsa/api-sequences.labels numSamplesPerClass outfile.txt')
    sys.exit(2)

def _main():
    if len(sys.argv) != 5:
        usage()

    folder = sys.argv[1]
    label_fn = sys.argv[2]
    k = int(sys.argv[3])
    outFn = sys.argv[4]

    # Get sample labels
    labels = get_labels(folder,label_fn)

    # Randomly choose samples from labels with at least 10 samples in them
    samples = dict()
    for c in labels:
        if len(labels[c]) < 5000:
            continue
        for s in random.sample(labels[c],k):
            samples[s] = c

    stats = dict()
    history = dict()

    # Iterate over samples and calculate their similarities
    for s1 in samples:
        c1 = samples[s1]

        if s1 not in history:
            history[s1] = set()
        if c1 not in stats:
            stats[c1] = dict()
            stats[c1]['jaccard'] = dict()
            stats[c1]['lsh'] = dict()
            stats[c1]['hamming'] = dict()

        for s2 in samples:
            # Don't duplicate similarity measurements
            if s1 == s2:
                continue
            if s2 in history:
                if s1 in history[s2]:
                    continue

            c2 = samples[s2]
            if c2 not in stats:
                stats[c2] = dict()
                stats[c2]['jaccard'] = dict()
                stats[c2]['lsh'] = dict()
                stats[c2]['hamming'] = dict()
            if c2 not in stats[c1]['jaccard']:
                stats[c1]['jaccard'][c2] = list()
                stats[c1]['lsh'][c2] = Counter()
                stats[c1]['hamming'][c2] = list()
            if c1 not in stats[c2]['jaccard']:
                stats[c2]['jaccard'][c1] = list()
                stats[c2]['lsh'][c1] = Counter()
                stats[c2]['hamming'][c1] = list()

            # Note that we've compared these samples now
            history[s1].add(s2)

            # Read API sequences
            lseq1 = readFile(folder,s1)
            lseq2 = readFile(folder,s2)

            seq1 = set(lseq1)
            seq2 = set(lseq2)

            # https://ekzhu.github.io/datasketch/lsh.html
            # Compare these two samples
            m1 = MinHash(num_perm=128)
            m2 = MinHash(num_perm=128)
            for d in seq1:
                m1.update(d.encode('utf8'))
            for d in seq2:
                m2.update(d.encode('utf8'))

            # Calculate LSH similarity
            lsh = MinHashLSH(threshold=0.7, num_perm=128)
            lsh.insert(samples[s1],m1)
            result = lsh.query(m2)
            if len(result) == 1:
                rl = True
            else:
                rl = False

            # Calculate Jaccard similarity
            rj = float(len(seq1.intersection(seq2)))/float(len(seq1.union(seq2)))

            # Pad smallest sequence
            if len(lseq1) < len(lseq2):
                diff = len(lseq2) - len(lseq1)
                lseq1.extend([0]*diff)
            elif len(lseq1) > len(lseq2):
                diff = len(lseq1) - len(lseq2)
                lseq2.extend([0]*diff)

            # Calculate Hamming distance
            rh = sum(s1 != s2 for s1,s2 in zip(lseq1,lseq2))

            # Keep track of similarities
            stats[c1]['jaccard'][c2].append(rj)
            stats[c1]['lsh'][c2][rl] += 1
            stats[c1]['hamming'][c2].append(rh)

            stats[c2]['jaccard'][c1].append(rj)
            stats[c2]['lsh'][c1][rl] += 1
            stats[c2]['hamming'][c1].append(rh)

            # Print status
            print('{0} {4}  {1} {5}: Jaccard similarity: {2}  |  > 0.7 LSH similarity: {3} | Hamming distance: {6}'.format(samples[s1],samples[s2],rj,rl,s1,s2,rh))

    # Print summary stats
    with open(outFn,'w') as fw:
        for c in stats:
            fw.write('{0}:\n'.format(c))
            for c2 in stats[c]['jaccard']:

                add = float(sum(stats[c]['jaccard'][c2]))
                total = float(len(stats[c]['jaccard'][c2]))

                add2 = float(sum(stats[c]['hamming'][c2]))
                total2 = float(len(stats[c]['hamming'][c2]))

                fw.write('    {0} {1} {2} {3}\n'.format(c2, add/total, stats[c]['lsh'][c2], add2/total2))

if __name__ == '__main__':
    _main()
