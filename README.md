# lstm api sequence

This models api call sequences using LSTM

## Requirements
  * Debian 8 (Jessie)
  * Python 2.7
  * keras 2.1.2
  * sklearn 0.19.1

## Install depedencies
`$ sudo ./setup.sh`

## NOTES
`preprocess.py` will delete the `features/` folder every time it is
run. Make sure you back the folder up before running the script
if you wish to keep your previously extracted features.

`preprocess.py` will also write to a file called `errors.txt` which lists the samples
which had no sequences within them or had errors whilst processing the samples.
These samples will not be transfered to the `features/` folder.

Below I reference a folder `api-sequences-folder/` which contains a list of files
of sample sequences of malware. Each file is named by the malware's SHA256 hash
for uniquely identify each sample. Each file contains the sequence of API calls
seen whilst executing the malware.

E.g.:
```
Open
Read
Write
Connect
Send
Receive
...
```

`malware.labels` is a file in which each line specifies a sample name and its
malware family label separated by a `tab` character.

E.g.:
```
2413FB3709B05939F04CF2E92F7D0897FC2596F9AD0B8A9EA855C7BFEBAAE892    familyA
F6FE187982FD924333B446C5FB9B96F328AC8994F88FA34007DBDF4D0FFFBE60    familyA
9711F36C1743E55CB0514C43DF74D981DC7775E11FC3465ADF0F80A2A07AB141    familyB
28E60DBEF52D6C2A2FA385FA04C2CD0880A71517B5C4F0E2ED28EFC393A9E9CE    familyC
E59905D5305FCC54909A875F4CC3F426A3A27584A461E1B16530FC2AD85A0693    familyA
...
```


NOTE that the label "benign" is for benign files I've gathered. This is not a malware
family.

If your data is sorted or formatted in any other way, you can modify `preprocess.py`
accordingly.

If you want to add a new API call, add it to "api.txt"
If you want to add a new malware famliy, add it to "label.txt"

## Usage
```
# Transform our sequences into data LSTM can work with
$ python preprocess.py api-sequences-folder/ malware.labels features/ windowSize {classification | regression}

# preprocess.py will use only the samples in malware.labels to extract the features
# of. This is nice so we can be easily selective over which malware to extract :)

# At this point we have a folder "features/" which contains a file "labels" which lists
# all of the sample file names within the folder "features/" and their corresponding labels

# "windowSize" is the size of the window of API calls to perform sliding window
# feature extraction for each malware sample.

# "regression" will ignore the labels within "malware.labels" and will instead label
# the next API call in a sequence.

# preprocess.py will also print out the number of unique API calls found
# which will be used as a parameter to lstm.py

# Train LSTM model
$ python lstm.py features/ models/ > out.txt

# "models/" will be removed every time lstm.py is run. This folder stores all
# of the models in JSON form to be imported and used by Keras in the future.

# "out.txt" will store the detailed output of training and testing your LSTM

# Evaluate LSTM model
$ python eval.py model.json weight.h5 train.pkl test.pkl > out.txt

# train.pkl and test.pkl are saved train/test fold to be used to construct
# confusion matrix and other statistics from running lstm.py



# Get stats of similarity of sequences both inter- and intra-family
$ python stats.py /data/arsa/api-sequences /data/arsa/api-sequences.labels numSamplesPerClass outfile.txt

# Compress/summarize sequences
$ python compress.py /data/arsa/api-sequences compressed-sequences/
```

## Example
```
$ python preprocess.py /data/arsa/api-sequences/ \
                       /data/arsa/api-sequences.labels \
                       /data/arsa/api-sequences-features/ \
                       32 \
                       classification

Reading in api.txt file...
Reading in label.txt file...
Reading in samples to preprocess...
Window Size: 32
Extracting sample's sequences: 8683/8683
Total number of sequences extracted: 6890790
Total number of PE samples extracted from: 7491
Longest sequence length: 4182014
Shortest sequence length which is > 0: 1
Average sequence length: 29437.39

$ python lstm.py /data/arsa/api-sequences-features/ \
                 /data/arsa/api-sequences-models/ > out-lstm.txt

$ python eval.py /data/arsa/api-sequences-models/fold1-model.json \
                 /data/arsa/api-sequences-models/fold1-weight.hd5 \
                 /data/arsa/api-sequences-models/fold1-train.pkl \
                 /data/arsa/api-sequences-models/fold1-test.pkl > out-eval.txt
```

## Create attack config for patchPE
```
$ python attack-config.py
```
