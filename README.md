# TFCGAN

Ground motion simulation in time-frequency domain based on conditional 
generative adversarial network and phase retrieval approaches 


### Install

Install python 3.10.0 (anacornda or venv), then:

```commandline
pip install --upgrade pip setuptools && pip install -e ".[dev]"
```

In the Python virtualenv and then:
Check versions: `pip freeze` should give `tensorflow==2.11.1` and `numpy==1.26.1`

Then to run tests: `pytest ./tests`

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
