# TFCGAN


`TFCGAN` is a Python program and library for strong ground motion simulation 
in time-frequency domain (TF) based on conditional generative adversarial 
network and phase retrieval approaches.

Used as command line application, TFCGAN can be used to create .npy files
of synthetic seismic waveforms using a pre-trained ANN model.

## Install

Install **Python 3.10** (anaconda or venv), then:

Standard install:
```commandline
pip install --upgrade pip setuptools && pip install .
```

Developers install (install editable package and test libraries):
```commandline
pip install --upgrade pip setuptools && pip install -e . && pip install pytest
```

In the Python virtualenv and then:
Check versions: `pip freeze` should give `tensorflow==2.11.1` and `numpy==1.26.1`

For developers, run tests via `pytest ./tests`


## Usage

TFCGAN is a command line application to be launched from the terminal. For help, 
type `tfcgan --help` in your terminal

TFCGAN can also be used in your Python code:

```python
from tfcgan.tfcgan import get_ground_shaking_synthesis, get_fas_response
# setup your parameters. Example:

mag = 7
dist = 100
vs30 = 760
num_waveforms = 10

# Generate waveform data
t, data = get_ground_shaking_synthesis(num_waveforms, mw=mag, ryhp=dist, vs30=vs30)

# t is a numpy array of length L ~= 4000 with a dt of 0.01 seconds
# data is a Numpy Matrix of shape (num_waveforms, L).

# Generate Fourier amplitude spectra
freq, fas = get_fas_response(t[1] - t[0], data)
```

## Citation

**Research article:**


>Esfahani, Reza DD, Fabrice Cotton, Matthias Ohrnberger, and Frank Scherbaum. "TFCGAN: Nonstationary Ground‐Motion Simulation in the Time–Frequency Domain Using Conditional Generative Adversarial Network (CGAN) and Phase Retrieval Methods."  *Bulletin of the Seismological Society of America * 113, no. 1 (2023): 453-467. https://doi.org/10.1785/0120220068

**Software:**

>Esfahani, Reza D. D.; Zaccarelli, Riccardo (2024): TFCGAN package: Conditional Generative Models in Time-Frequency domain for Ground motion simulation. GFZ Data Services. https://doi.org/10.5880/GFZ.2.6.2024.002

 



