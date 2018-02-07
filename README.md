# lstm api sequence

This models api call sequences using LSTM

## Requirements
  * Python 2.7
  * keras
  * sklearn

## Usage
```
$ python preprocess.py api-sequences-folder/ malware.labels features/
$ python lstm.py features/labels features/

For example
$ python preprocess.py /data/arsa/api-sequences /data/arsa/api-sequences.labels /data/arsa/api-sequences-features/
$ python lstm.py /data/arsa/api-sequences-features/labels /data/arsa/api-sequences-features/
```
