"""
core module
"""
import os
from tensorflow import keras
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from scipy.signal.windows import tukey

# from numba import prange

# TODO: check all single-letter variables and params and try to provide a more
#  meaningful name. Pyhon code conventions say: variables and function names
#  all LOWERCASE, and at least 3 letters. (for scientific code I usually relax some
#  conditions, e.g., "g", "pi", "x" "y" are fine, as long as they denote their scientific
#  counterpart)


# STFT parameters used in pre-processing step:
win_length = 128 + 64  # Window length
hop_length = 16  # Hop length
n_fft = 256  # nfft


# ###############
# Phase retrieval
# ###############

def pra_gla(tfr_m: np.ndarray,
            pr_int: np.int = 10
            )->np.ndarray:

    """TODO add doc"""

    mag = np.abs(tfr_m) #absolute magnitude
    #generate random phase
    phase = np.random.uniform(0, 2 * np.pi, (mag.shape[0], mag.shape[1]))

    recon_sig = librosa.istft(
        mag * np.exp(phase * 1j),
        hop_length=hop_length,
        win_length=win_length,
        length=4000)
     
    for i in range(pr_int):
        recon_tfr = librosa.stft(
            recon_sig,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )[:128, :248]
        phase = np.angle(recon_tfr)
        recon_tfr = mag * np.exp(1j * phase)
        recon_sig = librosa.istft(
            recon_tfr,
            hop_length=hop_length,
            win_length=win_length,
            length=4000)
        
    return recon_sig


def pra_admm(tf, rho, eps, pr_int=10, ab=0):
    # Code modified from https://github.com//phvial/PRBregDiv

    mag = np.abs(tf)
    phase = np.random.uniform(0, 0.2, (mag.shape[0], mag.shape[1]))
    tfr = mag * np.exp(1j * phase)
    x = librosa.istft(tfr, hop_length=hop_length, win_length=win_length, length=4000)
    a = 0
    x = my_filter(x, 0.05, 48, 100)
    for ii in range(pr_int):
        xx = librosa.stft(
            x,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft
        )[:128, :248]
        h = xx + (1/rho) * a
        ph = np.angle(h)
        u = compute_prox(abs(h), mag, rho, eps, ab)
        z = u * np.exp(1j * ph)
        x = librosa.istft(
            z - (1/rho) * a,
            hop_length=hop_length,
            win_length=win_length,
            length=4000)
        x_hat = librosa.stft(
            x,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft
        )[:128, :248]
        a = a + rho * (x_hat - z)
        x = my_filter(x, 0.05, 48, 100)

    return x


def compute_prox(y, r, rho, eps, ab: int):
    # Code modified from https://github.com//phvial/PRBregDiv
    eps = np.min(r) + eps
    if ab == 1:
        v = (rho * y + 2 * r) / (rho + 2)
    elif ab == 2:
        b = 1 / (r + eps) - rho * y
        delta = np.square(b) + 4 * rho
        v = (-b + np.sqrt(delta)) / (2 * rho)
    else:
        raise ValueError('compute_prox `ab` parameter should be in (1, 2)')

    return v


def butter_bandpass(lowcut, highcut, fs, order=4):
    """TODO add doc"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if low <= 0:
        w_n = high
        btype = "lowpass"
    elif high <= 0:
        w_n = low
        btype = "highpass"
    else:
        w_n = [low, high]
        btype = "bandpass"

    b, a = butter(order, w_n, btype=btype)
    
    return b, a


def my_filter(y, fmin, fmax, samp):  # FIXME: better function name
    """TODO: add doc"""
    b, a = butter_bandpass(fmin, fmax, samp)
    window_time = tukey(y.shape[-1], 0.1)
    return filtfilt(b, a, y * window_time, axis=-1)


# ######
# TFCGAN
# ######


class TFCGAN:
    
    def __init__(self,
                 dirc=None,
                 scalemin=-10,
                 scalemax=2.638887,
                 pwr=1,
                 noise=100,
                 mtype=1):
        """
        TODO provide doc
        
        :param dirc: Trained model directory
        :param scalemin: Scale factor in Pre-processing step
        :param scalemax: Scale factor in pre-processing step
        :param pwr: Power spectrum,
            1: means absolute value
            2: spectrogram
        :param noise: Noise vector
        :param mtype: Type of input label
            0: insert labels (Mw, R, Vs30) as a vector
            1: inset labels (Mw, R, Vs30) separately.
        """
        if dirc is None:
            dirc = os.path.abspath(os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'model'))
        self.dirc = dirc  # Model directory
        self.pwr = pwr  # Power or absolute
        self.scalemin = scalemin * self.pwr  # Scaling (clipping the dynamic range)
        self.scalemax = scalemax * self.pwr  # Maximum value
        self. noiseint = noise  # later space
        self.dt = 0.01

        # FIXME: do we just need Keras to load a model? is there a way maybe
        #  to just install keras and not the whole tensorflow package?
        self.model = keras.models.load_model(self.dirc)
        # self.mtype = mtype
        
    # Generate TFR
    def generator(self, mag, dis, vs, noise, ngen=1):
        """
        Generate TF representation for one scenario

        :param mag: Magnitude value
        :param dis: Distance value
        :param vs: Vs30 value
        :param noise: TODO provide doc
        :param ngen: Number of generated waveforms
        
        :return: Descaled Time-frequency representation
        
        """
        
        mag = np.ones([ngen, 1]) * mag
        dis = np.ones([ngen, 1]) * dis
        vs = np.ones([ngen, 1]) * vs / 1000
        
        label = np.concatenate([mag, dis, vs], axis=1)

        tf = self.model.predict([label, noise])[:, :, :, 0]
        # if self.mtype == 0:
        #     tf = self.model.predict([label,  noise])[:, :, :, 0]
        # elif self.mtype == 1:
        #     tf = self.model.predict(
        #         [label[:, 0], label[:, 1], label[:, 2],  noise]
        #     )[:, :, :, 0]
        # else:
        #     raise ValueError('TFCGAN `mtype` should be in (0, 1)')

        tf = (tf + 1) / 2
        tf = (tf * (self.scalemax-self.scalemin)) + self.scalemin
        tf = (10 ** tf) ** (1 / self.pwr)
        
        return tf
    
    # Calculate the TF, Time-history, and FAS
    def maker(self,
              mag,
              dis,
              vs,
              ngen=1,
              pr_int=10,
              mode="ADMM",
              rho=1e-5,
              eps=1e-3,
              ab=1):
        """
        Generate accelerogram for one scenario

        :param mag: Magnitude value
        :param dis: Distance value
        :param vs: Vs30 value
        :param ngen: Number of generated time-histories
        :param pr_int: Number of iteration in Phase retrieval
        :param mode: Type of Phase retireval algorithm
            "GLA": Griffin-Lim Algorithm
            "ADMM": ADMM algorithm for phase retireval based on Bregman
            divergences (https://hal.archives-ouvertes.fr/hal-03050635/document)
        :param rho: FIXME: add doc
        :param eps: FIXME add doc
        :param ab: FIXME add doc

        :return: a 4-elem,ent tuple  # FIXME: order mismatch? (see code below)
            tx: time vector
            freq = frequency vector
            xh: Generated time-history matrix
            S: Descaled Time-frequency representation matrix
        """
        if mode not in ("ADMM", "GLA"):
            raise ValueError('maker `mode` parameter should be in ("ADMM", "GLA")')

        noise = np.random.normal(0, 1, (ngen, self.noiseint))
        
        s = self.generator(mag, dis, vs, noise, ngen=ngen)

        # TODO: two lines below replaced by np.zeros. Check and cleanup in case:
        # x = np.empty((ngen, 4000))
        # x[:] = 0
        x = np.zeros((ngen, 4000))

        if mode == "ADMM":
            for i in range(ngen):
                x[i, :] = pra_admm(s[i, :, :], rho, eps, pr_int, ab)
        else:  # "GLA"
            for i in range(ngen):
                x[i, :] = pra_gla(s[i, :, :], pr_int)

        freq, xh = self.fft(x)
        tx = np.arange(x.shape[1]) * self.dt
        
        return tx, freq, xh, s, x  # FIXME: see docstring return
    
    def fft(self, s):
        # non-normalized fft without any norm specification
        
        if len(s.shape) == 1:
            s = s[np.newaxis, :]
        
        n = s.shape[1]//2
        lp = np.abs(np.fft.fft(s, norm="forward", axis=1))[:, :n]
        freq = np.linspace(0, 0.5, n)/self.dt
        
        return freq, lp.T
