"""
core module
"""
import os
#from tensorflow import keras
import keras
import numpy as np
from scipy.signal.windows import tukey
from scipy import signal
from scipy.signal import butter, sosfiltfilt
import typing as T


# ###############
# Short Time Fourier Transform
# ###############




class STFT(signal.ShortTimeFFT):
    def __init__(self, 
                 sr: int, 
                 window: int, 
                 hoplength: int,
                 nfft: int,
                 ) -> None:

        """
        Short Time Fourier Transform

        :param data: input signal
        :param fs: sampling frequency
        :param window: window length
        :param noverlap: overlap length
        :param nfft: number of FFT points
        """

        self.sr = sr
        self.window = window
        self.hoplength = hoplength
        self.nfft = nfft


    def stft(self, 
             signal: np.ndarray
             ) -> np.ndarray:
        
        """
        forward Short Time Fourier Transform

        """
        #self.spectrogram = self.stft_detrend(self.data, detrend = 'linear')
        self.freq, self.time, self.tfr = self.stft(signal, self.fs, self.window, self.hoplength, self.nfft)

        return self.freq, self.time, self.tfr
    
    def istft(self,
              tfr: np.ndarray,
              ) -> np.ndarray:
        
        """
        inverse Short Time Fourier Transform

        """

        _, rec_signal = self.istft(tfr, self.fs, self.window, self.hoplength, self.nfft)
        
        return rec_signal
    
    

# ###############
# Phase retrieval
# ###############

def phase_retrieval_gla(
        tfr_m: np.ndarray,
        iteration_pr: int = 10,
        stft_operator = None,
        ) -> np.ndarray:

    """ 
    phase retrieval algorithm based on Griffin-Lim Algorithm
    
    """

    mag = np.abs(tfr_m) #absolute magnitude
    #generate random phase
    phase = np.random.uniform(0, 2 * np.pi, (mag.shape[0], mag.shape[1]))

    recon_sig = stft_operator.istft(
        mag * np.exp(phase * 1j))
     
    for i in range(iteration_pr):
        recon_tfr = stft_operator.stft(
            recon_sig
        )
        phase = np.angle(recon_tfr)
        recon_tfr = mag * np.exp(1j * phase)
        recon_sig = stft_operator.istft(
            recon_tfr
            )
        
    return recon_sig


def pra_admm(
        tfr_m: np.ndarray, 
        rho, 
        eps, 
        iteration_pr: int = 10, 
        ab=0,
        ) -> np.ndarray: 
    
    """
    phase retrieval algorithm based on ADMM algorithm for phase retrieval based on Bregman divergences

    # Code modified from https://github.com//phvial/PRBregDiv
    """
    #get magnitude
    mag = np.absolute(tfr_m)
    #generate random phase
    phase = np.random.uniform(0, 0.2, (mag.shape[0], mag.shape[1]))
    #generate random phase
    a = 0 

    rec_signal = stft_operator.istft(mag * np.exp(1j * phase))

    for ii in range(iteration_pr):
        recon_tfr = stft_operator.stft(rec_signal)
        
        h = recon_tfr + (1/rho) * a
        ph = np.angle(h)
        u = compute_prox(abs(h), mag, rho, eps, ab)
        z = u * np.exp(1j * ph)

        rec_signal = stft_operator.istft(z - (1/rho) * a,)        
        x_hat = stft_operator.stft(rec_signal)
        a = a + rho * (x_hat - z)
        #x = my_filter(x, 0.05, 48, 100)

    return x


def compute_prox(
        y, 
        r, 
        rho, 
        eps, 
        ab: int,
        ) -> np.ndarray:
    
    """
    # Code modified from https://github.com//phvial/PRBregDiv
    # Compute the proximal operator of the l1 norm
    """
    
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



def filter_data(
        data: np.ndarray, 
        freqmin: T.Union[float, int, None],
        freqmax: T.Union[float, int, None],
        sr: int = 40, 
        filtertype: str = 'bp',
        filter_order: int = 10,
        ) -> np.ndarray:
    
    """
    # Filter the data using butterworth filter

    """
    
    if filtertype == 'bp' and (freqmin is not None or freqmax is not None):
        sos = butter(filter_order, [freqmin, freqmax], 'bandpass', fs=sr, output='sos')

    elif filtertype == 'lp' and freqmax is not None:
        sos = butter(filter_order, freqmax, 'lp', fs=sr, output='sos')

    elif filtertype == 'hp' and freqmin is not None:
        sos = butter(filter_order, freqmin, 'hp', fs=sr, output='sos')

    else:
        raise ValueError('filter_data `filtertype` parameter should be in ("bp", "lp", "hp")')

    datafilter = sosfiltfilt(sos, data, axis=-1)

    return datafilter


# ######
# TFCGAN
# ######


class TFCGAN:
    def __init__(
            self,
            dirc: str = None,
            scalemin: float = -10,
            scalemax:float = 2.638887,
            pwr:float = 1,
            noise_dim: int= 100,
            mtype: int=1,
            dt: float = 0.01,
            nfft: int = 256,
            hop_length: int = 16,
            win_length: int = 128 + 64,
            ) -> None:
        
        """        
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
        self. noise_dim = noise_dim  # later space
    
        # STFT parameters

        self.dt = 0.01
        self.nfft = nfft
        self.hop_length = hop_length
        self.win_length = win_length

        #self.stft_operator = STFT(self.dt, self.win_length, self.hop_length, self.nfft)
    
        self.model = keras.models.load_model(self.dirc)


    # Generate TFR
    def tf_generator(
            self,
            mag: int | float,
            dis: int | float, 
            vs: int | float, 
            noise_vec: np.ndarray,
            num_real: int = 1,
            ) -> np.ndarray:
        
        """
        Generate TF representation for one scenario

        :param mag: Magnitude value
        :param dis: Distance value
        :param vs: Vs30 value
        :param noise_vec: random noise vector
        :param ngen: Number of generated waveforms
        
        :return: Descaled Time-frequency representation

        """
        
        mag = np.ones([num_real, 1]) * mag
        dis = np.ones([num_real, 1]) * dis
        vs = np.ones([num_real, 1]) * vs / 1000
        
        label = np.concatenate([mag, dis, vs], axis=1)

        tf = self.model.predict([label, noise_vec])[:, :, :, 0]

        
        return tf
    
    # Calculate the TF, Time-history, and FAS

    def simulator(
            self,
            mw: int | float = 7, 
            rhyp: int | float = 10, 
            vs30: int | float = 760, 
            n_realization: int = 1,
            iter_pr:int=10,
            mode: str = "ADMM",
            rho: float = 1e-5,
            eps: float =1e-3,
            ab=1) -> tuple:
        
        """

        Generate accelerogram for one scenario

            :param mw: Magnitude value
            :param rhyp: Distance value
            :param vs30: Vs30 value
            :param n_realization: Number of generated time-histories
            :param iter_pr: Number of iteration in Phase retrieval
            :param mode: Type of Phase retrieval algorithm
                "ADMM": ADMM algorithm for phase retrieval based on Bregman divergences (https://hal.archives-ouvertes.fr/hal-03050635/document)
            :return: a 5-element tuple
                freq: frequency vector
                s: Descaled Time-frequency representation matrix
                x: Generated time-history matrix

        """
        if mode not in ("ADMM", "GLA"):
            raise ValueError('maker `mode` parameter should be in ("ADMM", "GLA")')

        noise = np.random.normal(0, 1, (n_realization, self.noise_dim))

        s = self.tf_generator(mw,
                              rhyp, 
                              vs30, 
                              noise, 
                              ngen = n_realization,
                              )

        # TODO: two lines below replaced by np.zeros. Check and cleanup in case:
        # x = np.empty((ngen, 4000))
        # x[:] = 0
        x = np.zeros((n_realization, 4000))

        if mode == "ADMM":
            for i in range(n_realization):

                x[i, :] = pra_admm(s[i, :, :], 
                                   rho, 
                                   eps, 
                                   iter_pr, 
                                   ab
                                   )
                
        else:  # "GLA"
            for i in range(n_realization):
                x[i, :] = phase_retrieval_gla(s[i, :, :], 
                                              iter_pr
                                              )


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
    

class Normalization:
    def __init__(self,
                    scalemin: float = -10,
                    scalemax: float = 2.638887,
                    pwr: float = 1,
                 ) -> None:
        
        """
        Should change with minmaxnormalization 
        sklearn.preprocessing.MinMaxScaler

        """
    
    def forward(self, 
                tf: np.ndarray
                ) -> np.ndarray:
        
        """
        Forward scaling
            
            """
        
        tf = (tf + 1) / 2
        tf = (tf * (self.scalemax-self.scalemin)) + self.scalemin
        tf = (10 ** tf) ** (1 / self.pwr)

        return tf
    
    def inverse(self, 
                tf: np.ndarray
                ) -> np.ndarray:
        
        """
        Inverse scaling 
            
            """
        
        tf = np.log10(tf ** self.pwr)
        tf = (tf - self.scalemin) / (self.scalemax - self.scalemin)
        tf = (tf * 2) - 1

        return tf
    
    def save(self, 
             dirc: str
             ) -> None:
        
        """
        Save the normalization parameters
            
            """
        
        np.save(dirc, [self.scalemin, self.scalemax, self.pwr])

    def load(self, 
             dirc: str
             ) -> None:
        
        """
        Load the normalization parameters
            
            """
        
        self.scalemin, self.scalemax, self.pwr = np.load(dirc)
        
        return self.scalemin, self.scalemax, self.pwr
