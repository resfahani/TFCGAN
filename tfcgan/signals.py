"""
module containing the signal processing functions for the TFCGAN project.
- Phase retrieval:
  - Phase retrieval algorithm based on ADMM algorithm and
  - Griffin-Lim Algorithm
- Filtering
"""
from scipy.signal import butter, sosfiltfilt, stft, istft, get_window
import numpy as np
from typing import Union


class PhaseRetrieval:
    """Phase Retrieval abstract base class"""

    def __init__(self, iteration_pr: int = 10):
        """
        Phase retrieval algorithm based on ADMM algorithm for phase retrieval
        based on Bregman divergences and Griffin-Lim Algorithm

        :param iteration_pr: Number of iteration in phase retrieval
        """
        self.iteration_pr = iteration_pr

    def apply_on_data(self, data: np.ndarray) -> np.ndarray:
        x_rt = []  # Time-history list
        for i in range(data.shape[0]):
            x = self.apply_on_array(tfr_m=data[i])
            x_rt.append(x)
        return np.asarray(x_rt)  # return the generated time-history

    def apply_on_array(self, tfr_m: np.ndarray) -> np.ndarray:
        """
        phase retrieval algorithm computation, to be implemented in subclasses

        :param tfr_m: Time-frequency representation
        """
        raise NotImplementedError('')

    def stft(self, x_signal: np.ndarray) -> np.ndarray:
        """
        forward Short Time Fourier Transform
        """
        window_length = 128 + 64
        noverlap = 128 + 64 - 16
        nfft = 256
        freq_ax, time_ax, tfr_complex = stft(x_signal,
                                             window=get_window('hann', window_length),
                                             nperseg=window_length,
                                             noverlap=noverlap,
                                             nfft=nfft,
                                             return_onesided=True)
        return tfr_complex[:128, :]

    def istft(self, tfr: np.ndarray) -> np.ndarray:
        """
        inverse Short Time Fourier Transform
        """
        window_length = 128 + 64
        noverlap = 128 + 64 - 16
        nfft = 256
        _, rec_signal = istft(tfr,
                              window=get_window('hann', window_length),
                              nperseg=window_length,
                              noverlap=noverlap,
                              nfft=nfft)
        return rec_signal


class GLA(PhaseRetrieval):
    """Phase Retrieval based on Bregman divergences and Griffin-Lim Algorithm"""

    def apply_on_array(self, tfr_m: np.ndarray) -> np.ndarray:
        """
        apply the phase retrieval on the given array, returning the reconstructed signal

        :param tfr_m: Time-frequency representation
        """
        mag = np.abs(tfr_m)  # absolute magnitude
        # phase initialization:
        phase = np.random.uniform(0, 2 * np.pi, (mag.shape[0], mag.shape[1]))
        # reconstruct the signal:
        recon_signal = self.istft(mag * np.exp(phase * 1j))
        recon_signal = filter_data(recon_signal, 0.1, 20,
                                   sr=100, filtertype='bp', filter_order=4)
        for i in range(self.iteration_pr):
            # calculate the tfr:
            recon_tfr = self.stft(recon_signal)
            # get the phase:
            phase = np.angle(recon_tfr)
            # mix with magnitude:
            recon_tfr = mag * np.exp(1j * phase)
            # update the reconstructed signal:
            recon_signal = self.istft(recon_tfr)

        return recon_signal


def filter_data(data: np.ndarray,
                freqmin: Union[float, None],
                freqmax: Union[float, None],
                sr: float = 40,
                filtertype: str = 'bp',
                filter_order: int = 10,
                ) -> np.ndarray:
    """
    Filter the data using butterworth filter

    :param data: input signal
    :param freqmin: Minimum frequency
    :param freqmax: Maximum frequency
    :param sr: Sampling rate
    :param filtertype: Type of filter (bp, lp, hp)
    :param filter_order: Order of the filter

    return: Filtered signal
    """
    if filtertype == 'bp' and (freqmin is not None or freqmax is not None):
        # bandpass filter:
        sos = butter(filter_order, [freqmin, freqmax], 'bandpass', fs=sr,
                     output='sos')
    elif filtertype == 'lp' and freqmax is not None:
        # lowpass filter:
        sos = butter(filter_order, freqmax, 'lp', fs=sr, output='sos')
    elif filtertype == 'hp' and freqmin is not None:
        # highpass filter
        sos = butter(filter_order, freqmin, 'hp', fs=sr, output='sos')
    else:
        raise ValueError('filter_data `filtertype` parameter should be in '
                         '("bp", "lp", "hp")')
    # Apply the filter on the last axis of data (time axis):
    return sosfiltfilt(sos, data, axis=-1)


class ADMM(PhaseRetrieval):
    """
    Phase retrieval based on ADMM algorithm.
    Code modified from https://github.com//phvial/PRBregDiv
    """

    def __init__(self,
                 iteration_pr: int = 10, *,
                 rho: float = 1e-5, eps: float = 1e-3, contrain_mode: str = "type1",
                 ):
        """
        Call the superclass __init__ and add class-specific parameters

        :param iteration_pr: Number of iteration in phase retrieval
        :param rho: ADMM parameter
        :param eps: ADMM parameter
        :param contrain_mode: Type of contrain mode (type1 or type2)
        """
        super().__init__(iteration_pr)
        self.rho = rho
        self.eps = eps
        self.contrain_mode = contrain_mode

    def apply_on_array(self, tfr_m: np.ndarray) -> np.ndarray:
        """
        apply the phase retrieval on the given array, returning the reconstructed signal

        :param tfr_m: Time-frequency representation
        """
        mag = np.absolute(tfr_m)  # get absolute magnitude
        phase = np.random.uniform(0, 0.2, (mag.shape[0], mag.shape[1]))  # initialization

        aux_var1 = 0
        # reconstruct the signal (initial signal):
        recon_signal = self.istft(mag * np.exp(1j * phase))

        for ii in range(self.iteration_pr):
            recon_tfr = self.stft(recon_signal)
            h = recon_tfr + (1 / self.rho) * aux_var1
            ph = np.angle(h)
            aux_var_u = self.compute_prox(abs(h), mag)
            aux_var_z = aux_var_u * np.exp(1j * ph)

            recon_signal = self.istft(aux_var_z - (1 / self.rho) * aux_var1)
            x_hat = self.stft(recon_signal)
            x_hat = x_hat[:128, :]
            aux_var1 = aux_var1 + self.rho * (x_hat - aux_var_z)

        return recon_signal

    def compute_prox(self, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Compute the proximal operator of the l1 norm
        """
        eps = np.min(r) + self.eps
        if self.contrain_mode == "type1":
            v = (self.rho * y + 2 * r) / (self.rho + 2)
        elif self.contrain_mode == "type2":
            b = 1 / (r + eps) - self.rho * y
            delta = np.square(b) + 4 * self.rho
            v = (-b + np.sqrt(delta)) / (2 * self.rho)
        else:
            raise ValueError('compute_prox `contrain_mode` parameter '
                             'should be in ("type1", "type2")')
        return v
