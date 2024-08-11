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
                 sr: int, 
                 window: int, 
                 hoplength: int,
                 nfft: int,
                 length: int = 4000,
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
        self.windowlength = window
        self.hoplength = hoplength
        self.nfft = nfft
        self.length = length


        #self.window = signal.get_window('hann', self.windowlength)


    def stft(self, 
             x_signal: np.ndarray,
             ) -> np.ndarray:
        
        """
        forward Short Time Fourier Transform

        """
        #self.spectrogram = self.stft_detrend(self.data, detrend = 'linear')
        #self.freq, self.time, self.tfr = self.stft(signal, self.fs, self.window, self.hoplength, self.nfft)

        freq_ax, time_ax, tfr_complex = signal.stft(x_signal, window = self.window,  nperseg = len(self.window),
                                                    noverlap = self.hoplength, nfft = self.nfft, return_onesided = True,
                                                    )


        #self.tfr_complex = librosa.stft(x_signal, hop_length=self.hoplength, win_length=self.windowlength, n_fft = self.nfft)[:128,:248]


        return tfr_complex[:128,:]

    def istft(self,
              tfr: np.ndarray,
              ) -> np.ndarray:
        
        """
        inverse Short Time Fourier Transform

        """
        
        _, rec_signal =  signal.istft(tfr, window = self.window,  nperseg = len(self.window), 
                                      noverlap = self.hoplength, nfft = self.nfft, 
                                      )
        
        #self.rec_signal = librosa.istft(tfr, hop_length=self.hoplength, win_length=self.windowlength, length=self.length)

        return rec_signal
    
    @property
    def window(self) -> np.ndarray:
        """
        windowing the signal
        """
        return signal.get_window('hann', self.windowlength)

# ###############
# Phase retrieval
# ###############

def phase_retrieval_gla(
        tfr_m: np.ndarray,
        stft_operator: T.Callable,
        iteration_pr: int = 10,
        ) -> np.ndarray:

    """ 
    phase retrieval algorithm based on Griffin-Lim Algorithm
    
    """

    mag = np.abs(tfr_m) #absolute magnitude
    #generate random phase
    phase = np.random.uniform(0, 2 * np.pi, (mag.shape[0], mag.shape[1]))

    recon_sig = stft_operator.istft(mag * np.exp(phase * 1j))
    recon_sig = filter_data(recon_sig, 0.1, 20, sr=100, filtertype='bp', filter_order=4)
    
    for i in range(iteration_pr):

        recon_tfr = stft_operator.stft(recon_sig)
        #recon_tfr = recon_tfr[:128,:]
        phase = np.angle(recon_tfr)
        recon_tfr = mag * np.exp(1j * phase)
        recon_sig = stft_operator.istft(recon_tfr)
        
    return recon_sig


def phase_retrieval_admm(
        tfr_m: np.ndarray, 
        stft_operator: T.Callable,
        rho: float = 1e-5, 
        eps: float = 1e-3, 
        iteration_pr: int = 10, 
        contrain_mode: str = "type1",
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
    aux_var1 = 0 

    rec_signal = stft_operator.istft(mag * np.exp(1j * phase))

    rec_signal = filter_data(rec_signal, 0.1, 20, sr=100, filtertype='bp', filter_order=4)

    for ii in range(iteration_pr):

        recon_tfr = stft_operator.stft(rec_signal)
        #recon_tfr = recon_tfr[:128,:]

        h = recon_tfr + (1/rho) * aux_var1
        ph = np.angle(h)
        aux_var_u = compute_prox(abs(h), mag, rho, eps, contrain_mode)
        aux_var_z = aux_var_u * np.exp(1j * ph)

        rec_signal = stft_operator.istft(aux_var_z - (1/rho) * aux_var1)
        x_hat = stft_operator.stft(rec_signal)
        x_hat = x_hat[:128,:]   
        aux_var1 = aux_var1 + rho * (x_hat - aux_var_z)
        #x = my_filter(x, 0.05, 48, 100)

    return rec_signal


def compute_prox(
        y: np.ndarray, 
        r: np.ndarray, 
        rho: float, 
        eps: float, 
        contrain_mode: str = "type1",
        ) -> np.ndarray:
    

    """
    # Code modified from https://github.com//phvial/PRBregDiv
    # Compute the proximal operator of the l1 norm
    """
    
    eps = np.min(r) + eps

    if contrain_mode == "type1":
        v = (rho * y + 2 * r) / (rho + 2)

    else:
        b = 1 / (r + eps) - rho * y
        delta = np.square(b) + 4 * rho
        v = (-b + np.sqrt(delta)) / (2 * rho)

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
        self.n_realization = 2


    # Generate TFR
    def tf_simulator(
            self,
            mag: int | float,
            dis: int | float, 
            vs: int | float, 
            noise_vec: np.ndarray,
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
        
        mag = np.ones([self.n_realization, 1]) * mag # Magnitude
        dis = np.ones([self.n_realization, 1]) * dis # Distance in km
        vs = np.ones([self.n_realization, 1]) * vs / 1000 # Vs30 in km/s
        
        label = np.concatenate([mag, dis, vs], axis=1)
        tf_simulation = self.model.predict([label, noise_vec])[:, :, :, 0]
        tf_simulation = self.normalization.inverse(tf_simulation)

        return tf_simulation
    


    # Calculate the TF, Time-history, and FAS
    
    def signal_simulator(self,
                  mw: int | float = 7, 
                  rhyp: int | float = 10, 
                  vs30: int | float = 760, 
                  n_realization: int = 1,
                  mode: str = "ADMM",
                  iter_pr:int = 20,
                  rho: float = 1e-5,
                  eps: float =1e-3,
                  ) -> tuple:
        

        """

        Generate accelerogram for one scenario

            :param mw: Magnitude value
            :param rhyp: Distance value
            :param vs30: Vs30 value
            :param n_realization: Number of generated time-histories
            :param iter_pr: Number of iteration in Phase retrieval
            :param mode: Type of Phase retrieval algorithm
                "ADMM": ADMM algorithm for  based on Bergman divergance (https://hal.archives-ouvertes.fr/hal-03050635/document)
                "GLA": Griffin-Lim Algorithm

            :return: a 5-element tuple
                freq: frequency vector
                s: Descaled Time-frequency representation matrix
                x: Generated time-history matrix

        """
        self.n_realization = n_realization

        if mode not in ("ADMM", "GLA"):
            raise ValueError('maker `mode` parameter should be in ("ADMM", "GLA")')

        noise = np.random.normal(0, 1, (self.n_realization, self.noise_dim))

        self.tf_synth = self.tf_simulator(mw, rhyp, vs30, noise)

        self.gm_synth = self.phase_retrieval(self.tf_synth, iter_pr, mode, rho, eps)

        self.gm_synth = filter_data(self.gm_synth, 0.1, 20, sr=100, filtertype='bp', filter_order=4)

        tx = np.arange(self.gm_synth.shape[-1]) * self.dt

        return  tx, self.gm_synth
    

    def frequency_response(self, gm_synth: np.ndarray) -> tuple:
        # Frequency response of the generated time-history
        
        freq, gm_synth = self.fft(gm_synth)
        
        return freq, gm_synth
    
    

    def phase_retrieval(self, 
                        tf_data : np.ndarray,
                        iter_pr: int = 20, 
                        mode: str = "ADMM", 
                        rho: float = 1e-5, 
                        eps: float = 1e-3, 
                        ) -> np.ndarray:
        
        """
        Phase retrieval algorithm

        """
        x_rt = []

        if mode == "ADMM":
            for i in range(self.n_realization):
                x = phase_retrieval_admm(tfr_m = tf_data[i, ...], rho = rho, eps = eps, stft_operator = self.stft_operator, iteration_pr = iter_pr)
                x_rt.append(x)

        else:  # "GLA"
            for i in range(self.n_realization):
                x = phase_retrieval_gla(tfr_m = tf_data[i, ...], iteration_pr = iter_pr, stft_operator = self.stft_operator )
                x_rt.append(x)

        return np.asarray(x_rt) # return the generated time-history
    
    

    def fft(self, s:np.ndarray) -> tuple:
        # non-normalized fft without any norm specification
        
        if len(s.shape) == 1:
            s = s[np.newaxis, :]
        
        n = s.shape[1]//2
        lp = np.abs(np.fft.fft(s, norm="forward", axis=1))[:, :n]

        freq = np.linspace(0, 0.5, n)/self.dt
        
        return freq, lp.T
    


    @property
    def model(self) -> T.Callable:
        # Load the trained model
        return keras.models.load_model(self.dirc)
    
    @property
    def stft_operator(self) ->T.Callable:
        return STFT(sr = int(1/self.dt), 
                    window = self.win_length, 
                    hoplength = self.hop_length, 
                    nfft = self.nfft)
    
    @property
    def get_tfr(self) -> np.ndarray:
        # return the time-frequency representation
        return self.tf_synth

    @property
    def normalization(self) -> T.Callable:
        # return the normalization function
        return Normalization(scalemin = self.scalemin, scalemax = self.scalemax, pwr = self.pwr)


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

    def inverse(self, 
                tf: np.ndarray
                ) -> np.ndarray:
        
        """
        Forward scaling
            
            """
        
        tf = (tf + 1) / 2
        tf = (tf * (self.scalemax-self.scalemin)) + self.scalemin
        tf = (10 ** tf) ** (1 / self.pwr)

        return tf
    
    def forward(self, 
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
