# model-api-sequence

This models api call sequences using LSTM

## Requirements
  * Debian 9 64-bit

## Clone repo
```
$ git clone --recurse-submodules git@github.com:evandowning/model-api-sequence.git
```

## Install depedencies
```
$ sudo ./setup.sh
```

## Usage
```
# Extract sequences from nvmtrace dumps (https://github.com/evandowning/nvmtrace/tree/kvm)
$ cd cuckoo-headless/extract_raw
$ python2.7 extract-existence.py

# Parse sequences into pickle files
$ python3 preprocess.py api-sequences/ cuckoo-headless/extract_raw/api.txt \
          label.txt malware_label.txt features/ windowSize {binary_classification | multi_classification | regression}

# Model data over 10-fold cross-validation & save models to file
$ python3 lstm.py features/ models/ save_model[True|False] save_data[True|False] \
          {binary_classification | multi_classification | regression} \
          convert_classes.txt

# Evaluate model
$ python3 evaluation.py models/model.json models/weight.h5 features/ \
          malware_label.txt labels.txt predictions.csv convert_classes.txt
```

## Measure similarity of sequences (both inter- and intra-family)
```
$ python3 sim_stats.py /data/arsa/api-sequences /data/arsa/api-sequences.labels numSamplesPerClass outfile.txt
```

## Create attack config for patchPE
```
$ python3 attack-config.py
```

## Create PNG images of sequences
```
$ python3 color.py /data/arsa/api-sequences-features/
```

## NOTES
`preprocess.py` will write to a file called `errors.txt` which lists the samples
which had no sequences within them or had errors whilst processing the samples.
These samples will not be transferred to the `features/` folder.

Below I reference a folder `api-sequences/` which contains a list of files
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

`malware_label.txt` is a file in which each line specifies a sample name and its
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

If your data is sorted or formatted in any other way, you can modify `preprocess.py`
accordingly.

If you want to add a new API call to keep track of, add it to "api.txt"
If you want to add a new malware family, add it to "label.txt"
