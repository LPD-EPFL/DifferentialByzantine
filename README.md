# Differential Privacy and Byzantine Resilience in SGD: Do They Add Up?

## Hardware dependency

There is no particular dependence in the hardware used.\
Any hardware which can run the following dependencies will do.

## Software dependency

Besides Python 3.7.3 (or above) and its standard library, this software depends on the following Python packages:

* numpy 1.19.1
* torch 1.6.0
* torchvision 0.7.0
* pandas 1.1.0
* matplotlib 3.0.2
* tqdm 4.40.2
* PIL 7.2.0
* six 1.15.0
* pytz 2020.1
* dateutil 2.8.1
* pyparsing 2.2.0
* cycler 0.10.0
* kiwisolver 1.0.1
* cffi 1.13.2

You may probably use any later version for most of these libraries.

## Reproducing the results

The entirety of our results can be reproduced in one command.\
From the directory containing `reproduce.py`, please run:
```
$ python3 reproduce.py
```

This script will automatically download the dataset, run the experiments and produce the final graphs.

