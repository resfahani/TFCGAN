<div align=justify>

<div align=center>

TFCGAN
=======


<br>

</div>


`TFCGAN` is a python package for strong ground motion simulation in time-frequency domain (TF) based on based on conditional generative adversarial network and phase retrieval approaches. 


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

TFCGAN is a command line application to be launched from the terminal. For help, 
type `tfcgan --help` in your terminal

TFCGAN can also be used in your Python code 
(disclaimer: the snippet below has not been 
tested, please provide feedback in case of import errors):

```python
from tfcgan.tfcgan import TFCGAN
# setup your parameters. Example:

mag = 7
dist = 100
vs30 = 760
num_waveforms = 10
# Model
tfcgan = TFCGAN()

# Generate labels for a specific scenario
tfcgan.create_scenario(mag, dist, vs30, num_waveforms = num_waveforms)

# Generate the time-frequency representation
tf = model.get_tf_representation

# Generate waveform data
t, data = tfcgan.get_ground_shaking_synthesis

# Generate Fourier amplitude spectra
freq, fas = tfcgan.get_fas_response

# data is a Numpy Matrix of shape (num_waveforms, 4000). 
# Each waveform delta time is 0.01 sec 
# (i.e., each waveform is 40s long by default)
```


### TODO list:
- create model with tensorflow 12.16 or whatever,
- create models in hdf format (see tensorflow doc)
- check the running code by running the tests

<!-- 
## Model

## Data 

update require

-->

Performance
-----------

</div>