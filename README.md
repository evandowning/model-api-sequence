# lstm api sequence

This models api call sequences using LSTM

## Requirements
  * Python 2.7
  * keras
  * sklearn

## NOTES
preprocess.py will delete the features/ folder every time it is
run. Make sure you back the folder up before running the script
if you wish to keep your previously extracted features. It will also
overwrite "errors.txt"

preprocess.py will also write to a file called "errors.txt" which lists the samples
which has no sequences within them or had errors whilst processing the samples

## Usage
```
$ python preprocess.py api-sequences-folder/ malware.labels features/
$ python lstm.py features/labels features/

# At this point we have a folder "features/" which contains a file "labels" which lists
# all of the samples within the folder "features/" and their corresponding labels

For example
$ python preprocess.py /data/arsa/api-sequences /data/arsa/api-sequences.labels /data/arsa/api-sequences-features/
$ python lstm.py /data/arsa/api-sequences-features/labels /data/arsa/api-sequences-features/
```
