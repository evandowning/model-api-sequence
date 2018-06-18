# Test "compressing" traces
# I.e., remove back-to-back redundant API calls

import sys
import os
import shutil
from multiprocessing import Pool

# Compresses sequence
def compress(folder, s, target):
    fn = os.path.join(folder,s)
    out = os.path.join(target,s)

    last = ''
    count_before = 0
    count_after = 0

    with open(out,'w') as fw:
        # Read in trace
        with open(fn,'r') as fr:
            for line in fr:
                line = line.strip('\n')
                count_before += 1

                if last == '':
                    last = line
                    continue

                # If this is a repeated API call, skip it
                if line == last:
                    continue
                else:
                    last = line

                # Write call to new file
                fw.write('{0}\n'.format(line))
                count_after += 1

    # Return length of new sequence
    return count_before,count_after,s

def compress_wrapper(args):
    return compress(*args)

def getFiles(folder):
    rv = list()
    for root, dirs, files in os.walk(folder):
        for fn in files:
            rv.append(fn)
    return rv

def usage():
    print 'usage: python compress.py /data/arsa/api-sequences compressed-sequences/'
    sys.exit(2)

def _main():
    if len(sys.argv) != 3:
        usage()

    folder = sys.argv[1]
    target = sys.argv[2]

    # Remove the compressed sequences folder if it already exists
    if os.path.exists(target):
        shutil.rmtree(target)
    os.mkdir(target)

    # Get sequences files
    samples = getFiles(folder)

    # Create argument pools
    args = [(folder,s,target) for s in samples]

    before = 0
    after = 0

    # NOTE: debugging
#   for a in args:
#       c1,c2,s = compress_wrapper(a)
#       before += c1
#       after += c2
#       print 'Processed {0}'.format(s)
#   print 'Total traces: {0}'.format(len(args))
#   print 'Avg. length before: {0}'.format(float(before)/len(args))
#   print 'Avg. length after: {0}'.format(float(after)/len(args))
#   return

    # Compress sequences
    pool = Pool(20)
    results = pool.imap_unordered(compress_wrapper, args)
    for e,r in enumerate(results):
        c1,c2,s = r
        before += c1
        after += c2
        sys.stdout.write('Compressing traces: {0}/{1}\r'.format(e+1,len(args)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print 'Avg. length before: {0}'.format(float(before)/len(args))
    print 'Avg. length after: {0}'.format(float(after)/len(args))

if __name__ == '__main__':
    _main()
