# TFCGAN

Ground motion simulation in time-frequency domain based on conditional 
generative adversarial network and phase retrieval approaches. 

Used as command line application, TFCGAN can be used to create .npy files
of synthetic seismic waveforms using a pre-trained ANN model.


### Install

Install python 3.10.0 (anaconda or venv), then:

Standard install:
```commandline
pip install --upgrade pip setuptools && pip install .
```

Developers install (install editable package and tests libraries):
```commandline
pip install --upgrade pip setuptools && pip install -e ".[dev]"
```

In the Python virtualenv and then:
Check versions: `pip freeze` should give `tensorflow==2.11.1` and `numpy==1.26.1`

For developers, run tests via `pytest ./tests`


### Usage

type `tfcgan` in your terminal. The printout should be something like this:
```commandline

```


TODO list:
- create model with tensorflow 12.16 or whatever,
- create models in hdf format (see tensorflow doc)
- check the running code by running the tests


Legacy doc (fix):
This repository contains codes for reproducing some figures in the TFCGAN article. 
The following figure shows the flowchart of the proposed approach. 



![alt text](./fig/Flowchart.jpg?raw=true)

## Model

## Data 

update require
