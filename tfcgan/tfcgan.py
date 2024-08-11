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
#import librosa

# ###############
# Short Time Fourier Transform
# ###############

class STFT():
    def __init__(self, 
                 sr: float = 100, 
                 window_length: int = 128 + 64, 
                 noverlap: int = 128+64-16,
                 n_fft: int = 256,
                 length: int = 4000,
                 ) -> None:
        
        """
        Short Time Fourier Transform
            :param data: input signal
            :param fs: sampling frequency
            :param window: window length
            :param noverlap: overlap length
            :param nfft: number of FFT points
            :param length: length of the signal
        """

        self.sr = sr
        self.window_length = window_length
        self.noverlap = noverlap
        self.n_fft = n_fft
        self.length = length

    def stft(self, x_signal: np.ndarray) -> np.ndarray:
        
        """
        forward Short Time Fourier Transform

        """
        #self.spectrogram = self.stft_detrend(self.data, detrend = 'linear')
        #self.freq, self.time, self.tfr = self.stft(signal, self.fs, self.window, self.hoplength, self.nfft)
        freq_ax, time_ax, tfr_complex = signal.stft(x_signal, window = self.window,  nperseg = self.window_length,
                                                    noverlap = self.noverlap, nfft = self.n_fft, return_onesided = True,)
        #self.tfr_complex = librosa.stft(x_signal, hop_length=self.hoplength, win_length=self.windowlength, n_fft = self.nfft)[:128,:248]

        return tfr_complex[:128,:]

    def istft(self, tfr: np.ndarray) -> np.ndarray:
        
        """
        inverse Short Time Fourier Transform
        """
        _, rec_signal =  signal.istft(tfr, window = self.window,  nperseg = self.window_length, 
                                      noverlap = self.noverlap, nfft = self.n_fft, )        
        #self.rec_signal = librosa.istft(tfr, hop_length=self.hoplength, win_length=self.windowlength, length=self.length)
        return rec_signal

    def __update__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    @property
    def dt(self) -> float:
        return 1 / self.sr
    
    @property
    def window(self) -> np.ndarray:
        return signal.get_window('hann', self.window_length)

# ###############
# Phase retrieval
# ###############

class PhaseRetrieval:
    def __init__(self, 
                 stft_operator: T.Callable = None, 
                 iteration_pr: int = 10,
                 rho: float = 1e-5, 
                 eps: float = 1e-3, 
                 contrain_mode: str = "type1",
                 ) -> None:
        """
        Phase retrieval algorithm based on ADMM algorithm for phase retrieval based on Bregman divergences and Griffin-Lim Algorithm

            :param tfr_m: Time-frequency representation
            :param stft_operator: STFT operator
            :param iteration_pr: Number of iteration in phase retrieval
            :param rho: ADMM parameter
            :param eps: ADMM parameter
            :param contrain_mode: Type of contrain mode (type1 or type2)

        """

        self.stft_operator = stft_operator
        self.iteration_pr = iteration_pr
        self.rho = rho
        self.eps = eps
        self.contrain_mode = contrain_mode

    def phase_retrieval_gla(self, tfr_m: np.ndarray) -> np.ndarray:
        """ 
        phase retrieval algorithm based on Griffin-Lim Algorithm
            :param tfr_m: Time-frequency representation
        """

        mag = np.abs(tfr_m) #absolute magnitude
        phase = np.random.uniform(0, 2 * np.pi, (mag.shape[0], mag.shape[1])) # phase initialization

        recon_signal = self.stft_operator.istft(mag * np.exp(phase * 1j)) # reconstruct the signal
        recon_signal = filter_data(recon_signal, 0.1, 20, sr=100, filtertype='bp', filter_order=4) # Filtering the signal

        for i in range(self.iteration_pr): # number of iteration
            recon_tfr = self.stft_operator.stft(recon_signal) #calculate the tfr
            #recon_tfr = recon_tfr[:128,:]
            phase = np.angle(recon_tfr) # get the phase
            recon_tfr = mag * np.exp(1j * phase) # mix with magnitude
            recon_signal = self.stft_operator.istft(recon_tfr) # update the reconstructed signal

        return recon_signal
    
    def phase_retrieval_admm(self, tfr_m: np.ndarray) -> np.ndarray:     
        """
        phase retrieval algorithm based on ADMM algorithm for phase retrieval based on Bregman divergences
        # Code modified from https://github.com//phvial/PRBregDiv
        """

        mag = np.absolute(tfr_m)     #get absolute magnitude
        phase = np.random.uniform(0, 0.2, (mag.shape[0], mag.shape[1]))     #initialization

        aux_var1 = 0 
        recon_signal = self.stft_operator.istft(mag * np.exp(1j * phase)) # reconstruct the signal (initial signal)
        #recon_signal = filter_data(recon_signal, 0.1, 20, sr=100, filtertype='bp', filter_order=4)

        for ii in range(self.iteration_pr):
            recon_tfr = self.stft_operator.stft(recon_signal)
            #recon_tfr = recon_tfr[:128,:]
            h = recon_tfr + (1/self.rho) * aux_var1
            ph = np.angle(h)
            aux_var_u = self.compute_prox(abs(h), mag)
            aux_var_z = aux_var_u * np.exp(1j * ph)

            recon_signal = self.stft_operator.istft(aux_var_z - (1/self.rho) * aux_var1)
            x_hat = self.stft_operator.stft(recon_signal)
            x_hat = x_hat[:128,:]   
            aux_var1 = aux_var1 + self.rho * (x_hat - aux_var_z)
            #x = my_filter(x, 0.05, 48, 100)

        return recon_signal

    def compute_prox(self, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        # Code modified from https://github.com//phvial/PRBregDiv
        # Compute the proximal operator of the l1 norm
        """

        eps = np.min(r) + self.eps
        if self.contrain_mode == "type1":
            v = (self.rho * y + 2 * r) / (self.rho + 2)
        elif self.contrain_mode == "type2":
            b = 1 / (r + eps) - self.rho * y
            delta = np.square(b) + 4 * self.rho
            v = (-b + np.sqrt(delta)) / (2 * self.rho)
        else:
            raise ValueError('compute_prox `contrain_mode` parameter should be in ("type1", "type2")')
        
        return v

    def apply_on_data(self, data: np.ndarray, mode: str = "ADMM") -> np.ndarray:

        x_rt = [] # Time-history list

        for i in range(data.shape[0]):

            if mode == "ADMM":
                x = self.phase_retrieval_admm(tfr_m = data[i])
            elif mode == "GLA":  # "GLA"
                x = self.phase_retrieval_gla(tfr_m = data[i])
            else:
                raise ValueError('apply_on_data `mode` parameter should be in ("ADMM", "GLA")')
            
            x_rt.append(x)

        return np.asarray(x_rt) # return the generated time-history
    
# ###############
# Filter
# ###############

def filter_data(data: np.ndarray, 
                freqmin: T.Union[float, int, None],
                freqmax: T.Union[float, int, None],
                sr: int = 40, 
                filtertype: str = 'bp',
                filter_order: int = 10,
                ) -> np.ndarray:
    
    """
    # Filter the data using butterworth filter
        :param data: input signal
        :param freqmin: Minimum frequency
        :param freqmax: Maximum frequency
        :param sr: Sampling rate
        :param filtertype: Type of filter (bp, lp, hp)
        :param filter_order: Order of the filter
    
    return: Filtered signal
    """

    if filtertype == 'bp' and (freqmin is not None or freqmax is not None):
        sos = butter(filter_order, [freqmin, freqmax], 'bandpass', fs=sr, output='sos') # bandpass filter

    elif filtertype == 'lp' and freqmax is not None:
        sos = butter(filter_order, freqmax, 'lp', fs=sr, output='sos') # lowpass filter
    
    elif filtertype == 'hp' and freqmin is not None:
        sos = butter(filter_order, freqmin, 'hp', fs=sr, output='sos') # highpass filter

    else: 
        raise ValueError('filter_data `filtertype` parameter should be in ("bp", "lp", "hp")') # raise error if the filter type is not correct
    
    return sosfiltfilt(sos, data, axis=-1) # Apply the filter on the last axis of data (time axis)

# ###############
# TFCGAN
# ###############

class TFCGAN:
    def __init__(self,
                 dirc: str = None,
                 scalemin: float = -10,
                 scalemax:float = 2.638887,
                 pwr:float = 1,
                 noise_dim: int= 100,
                  ) -> None:
        """        
        :param dirc: Trained model directory
        :param scalemin: Scale factor in Pre-processing step
        :param scalemax: Scale factor in pre-processing step
        :param pwr: Power spectrum,
            1: means absolute value
            2: spectrogram
        :param noise_dim: Noise dimension
        :param dt: Time step
        """

        if dirc is None:
            dirc = os.path.abspath(os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'model'))
            
        # Model directory and normalization parameters            
        self.dirc = dirc  # Model directory
        self.pwr = pwr  # Power or absolute
        self.scalemin = scalemin * self.pwr  # Scaling (clipping the dynamic range)
        self.scalemax = scalemax * self.pwr  # Maximum value
        self. noise_dim = noise_dim  # later space

        self.num_waveforms = 2
    
    def tf_synthesis(self,
                     mw: int | float = 7, 
                     rhyp: int | float = 10, 
                     vs30: int | float = 760, 
                     num_waveforms: int = 1,
                     mode: str = "ADMM",
                     iter_pr:int = 20,
                     rho: float = 1e-5,
                     eps: float =1e-3,
                     verbose: bool = True,
                     ) -> np.ndarray:
        """
        Generate accelerogram for one scenario
            :param mw: Magnitude value
            :param rhyp: Distance value
            :param vs30: Vs30 value
            :param num_waveforms: Number of generated time-histories
            :param mode: Type of Phase retrieval algorithm
                "ADMM": ADMM algorithm for  based on Bergman divergance (https://hal.archives-ouvertes.fr/hal-03050635/document)
                "GLA": Griffin-Lim Algorithm
            :param iter_pr: Number of iteration in Phase retrieval
            :param rho: ADMM parameter
            :param eps: ADMM parameter

            :return: a 5-element tuple
                self.tx: Time vector
                self.gm_synth: Generated time-history matrix
        """

        self.num_waveforms = num_waveforms # update the number of realization
        self.iter_pr = iter_pr # update the number of iteration
        self.rho = rho # update the ADMM parameter
        self.eps = eps # update the ADMM parameter
        self.mode = mode # update the mode
        self.mw = mw # update the magnitude
        self.rhyp = rhyp # update the distance
        self.vs30 = vs30 # update the Vs30

        label = np.ones([self.num_waveforms, 3]) # Label vector
        label[:, 0] = label[:, 0] * self.mw # Magnitude
        label[:, 1] = label[:, 1] * self.rhyp # Distance in km
        label[:, 2] = label[:, 2] * self.vs30/1000 # Vs30 in km/s
                
        tf_simulation = self.model.predict([label, self.noise_gen])[:, :, :, 0] # simulate Time-frequency representation
        self.tf_synth = self.normalization.inverse(tf_simulation) # Descaled Time-frequency representation

        if verbose:
            print(self.__repr__())

        return self.tf_synth
    
    def __repr__(self) -> str:

        return (
            f"Moment magnitude = {self.mw}, "
            f"hypocenter distance = {self.rhyp} km, "
            f"Vs30 = {self.vs30} m/s, "
            f"num_waveforms = {self.num_waveforms}, "
            f"phase retrieval mode = {self.mode}, "
            f"phase retrieval iteration = {self.iter_pr}, "
            )
    
    def fft(self, gm_synth: np.ndarray) -> tuple:
        # non-normalized fft without any norm specification

        if len(gm_synth.shape) == 1:
            gm_synth = gm_synth[np.newaxis, :]
        fas_synth = np.abs(np.fft.fft(gm_synth, norm="forward", axis=-1))

        return np.linspace(0, 0.5, gm_synth.shape[-1]//2)/self.stft_operator.dt, fas_synth[:, :gm_synth.shape[-1]//2]
    
    @property
    def get_ground_shaking_synthesis(self) -> tuple:
        """
        Generate the time-history
        """        
        self.gm_synth = self.phase_retrieval.apply_on_data(self.tf_synth, self.mode) # Reconstruct the time-history
        tx = np.arange(self.gm_synth.shape[-1]) * self.stft_operator.dt  # Time vector

        return  tx, self.gm_synth

    @property
    def filtered_data(self) -> np.ndarray:
        # Filter the generated time-history
        return filter_data(self.gm_synth, 0.1, 20, sr=self.stft_operator.sr, filtertype='bp', filter_order=4)

    @property
    def phase_retrieval(self) -> T.Callable:
        """
        Phase retrieval algorithm
        """
        return PhaseRetrieval(stft_operator = self.stft_operator, iteration_pr = self.iter_pr, rho = self.rho, eps = self.eps)
    
    @property
    def model(self) -> T.Callable:
        # Load the trained model
        return keras.models.load_model(self.dirc)
    
    @property
    def stft_operator(self) -> T.Callable:
        # return the STFT operator
        return STFT()
    
    @property
    def get_tfr(self) -> np.ndarray:
        # return the time-frequency representation
        return self.tf_synth

    @property
    def normalization(self) -> T.Callable:
        # return the normalization function
        return Normalization(scalemin = self.scalemin, scalemax = self.scalemax, pwr = self.pwr)

    @property
    def noise_gen(self) -> np.ndarray:
        # Generate random noise vector
        return np.random.normal(0, 1, (self.num_waveforms, self.noise_dim))
    
    @property
    def get_fas_response(self) -> tuple:
        # Frequency response of the generated time-history
        freq, fas_synth = self.fft(self.gm_synth) 
        return freq, fas_synth

# ###############
## Normalization
# ############### 

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
        
        self.scalemin = scalemin
        self.scalemax = scalemax
        self.pwr = pwr

    def inverse(self, tf: np.ndarray) -> np.ndarray:

        tf = (tf + 1) / 2
        tf = (tf * (self.scalemax-self.scalemin)) + self.scalemin
        tf = (10 ** tf) ** (1 / self.pwr)

        return tf
    
    def forward(self, tf: np.ndarray) -> np.ndarray:
        
        tf = np.log10(tf ** self.pwr)
        tf = (tf - self.scalemin) / (self.scalemax - self.scalemin)
        tf = (tf * 2) - 1

        return tf
    
    def save(self, dirc: str) -> None:
        """
        Save the normalization parameters
        """
        
        np.save(dirc, [self.scalemin, self.scalemax, self.pwr])

    def load(self, dirc: str) -> None:
        """
        Load the normalization parameters
        """

        self.scalemin, self.scalemax, self.pwr = np.load(dirc)

        return self.scalemin, self.scalemax, self.pwr