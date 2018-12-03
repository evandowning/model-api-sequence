# This script creates an attack config file to be used by patchPE
import sys
from itertools import izip

def read_seq(fn):
    with open(fn,'r') as fr:
        for line in fr:
            line = line.strip('\n')
            yield line

def usage():
    print 'python attack-config.py original-sequence attack-sequence config-output'
    sys.exit(2)

def _main():
    if len(sys.argv) != 4:
        usage()

    original = sys.argv[1]
    attack = sys.argv[2]
    output = sys.argv[3]

    # Read contents of files
    seq_original = read_seq(original)
    seq_attack = read_seq(attack)

    pc,a = seq_original.next().split(' ')
    b = seq_attack.next()

    # Dictionary to hold calls to insert
    shells = dict()

    # Determine what needs to be inserted where
    while True:
        try:
            # Find the next mismatch
            while (a == b):
                pc,a = seq_original.next().split(' ')
                b = seq_attack.next()

            # Find the next match
            while (a != b):
#               print 'Insert {0} before {1} {2}\n'.format(b, pc, a)
                b = b.lower()
                if b not in shells:
                    shells[b] = dict()

                shells[b][pc] = a

                b = seq_attack.next()

        except StopIteration as e:
            break

    # Create shellcode config file
    with open(output,'w') as fw:
        for k,v in shells.iteritems():
            fw.write('[shellcode_{0}]\n'.format(k))
            fw.write('target_addr = (\n')

            for k2,v2 in v.iteritems():
                fw.write('    # {0}\n'.format(v2))
                fw.write('    {0},\n'.format(hex(int(k2))))

            fw.write('              )\n\n')

if __name__ == '__main__':
    _main()
