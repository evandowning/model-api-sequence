# lstm api sequence

This models api call sequences using LSTM

## Requirements
  * Python 2.7
  * keras 2.1.2
  * sklearn 0.19.1

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

If your data is sorted or formatted in any other way, you can modify `preprocess.py`
accordingly.

## Usage
```
# Transform our sequences into data LSTM can work with
$ python preprocess.py api-sequences-folder/ malware.labels features/

# At this point we have a folder "features/" which contains a file "labels" which lists
# all of the sample file names within the folder "features/" and their corresponding labels

# Train LSTM model
$ python lstm.py features/labels features/
```

## Example
```
$ python preprocess.py /data/arsa/api-sequences /data/arsa/api-sequences.labels /data/arsa/api-sequences-features/
$ python lstm.py /data/arsa/api-sequences-features/labels /data/arsa/api-sequences-features/
```
