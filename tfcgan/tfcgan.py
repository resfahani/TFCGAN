"""
core module
"""
import os
import numpy as np

from keras.models import load_model
from keras.engine.functional import Functional

from tfcgan.normalization import DataNormalization
from tfcgan.signal import STFT, PhaseRetrieval


class TFCGAN:
    def __init__(self,
                 dirc: str = None,
                 scalemin: float = -10,
                 scalemax: float = 2.638887,
                 pwr: float = 1
                 ) -> None:
        """        
        :param dirc: Trained model directory. None (the default) will load the default
            model shipped with the package
        :param scalemin: Scale factor in pre-processing step (min. value)
        :param scalemax: Scale factor in pre-processing step (max. value)
        :param pwr: Power spectrum in pre-processing step:
            1: means absolute value
            2: spectrogram
        """
        if dirc is None:
            dirc = os.path.abspath(os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'model'))

        self.dirc = dirc  # Model directory
        self.normalization = DataNormalization(scalemin=scalemin,
                                               scalemax=scalemax,
                                               pwr=pwr)
        self._model = None  # lazy loaded (see property)

    @property
    def model(self) -> Functional:
        if self._model is None:
            # Load the trained model
            self._model = load_model(self.dirc)
        return self._model

    def get_tfr(self, mw: float = 7, rhyp: float = 10, vs30: float = 760,
                num_waveforms: int = 1, noise_dim: int = 100) -> np.ndarray:
        """
        Generate a scenario and return the time-frequency representation
            :param mw: Magnitude value
            :param rhyp: Distance value
            :param vs30: Vs30 value
            :param num_waveforms: Number of generated time-histories
            :param noise_dim: Noise dimension
        """
        label = np.ones([num_waveforms, 3])
        label[:, 0] = label[:, 0] * mw  # Magnitude
        label[:, 1] = label[:, 1] * rhyp  # Distance in km
        label[:, 2] = label[:, 2] * vs30 / 1000  # Vs30 in km/s

        noise_gen = np.random.normal(0, 1, (num_waveforms, noise_dim))
        # simulate Time-frequency representation and descale it:
        tf_synth = self.normalization.inverse(
            self.model.predict([label, noise_gen])[:, :, :, 0]
        )
        return tf_synth

    def get_ground_shaking_synthesis(self,
                                     n_waveforms: int, *,
                                     dt: float = 0.01,
                                     mw: float = 7,
                                     rhyp: float = 10,
                                     vs30: float = 760,
                                     mode: str = "ADMM",
                                     iter_pr: int = 20,
                                     rho: float = 1e-5,
                                     eps: float = 1e-3
                                     ) -> tuple:
        """
        Generate synthetic waveforms (time histories) from the given scenario

        :param n_waveforms: Number of generated time-histories
        :param dt: waveforms time step
        :param mw: Magnitude value
        :param rhyp: Distance value
        :param vs30: Vs30 value
        :param mode: Type of Phase retrieval algorithm
            "ADMM" (the default): ADMM algorithm for  based on Bergman divergance
            (https://hal.archives-ouvertes.fr/hal-03050635/document)
            "GLA": Griffin-Lim Algorithm
        :param iter_pr: Number of iteration in Phase retrieval
        :param rho: parameter of the ADMM algorithm. Ignored if mode is not 'ADMM'
        :param eps: parameter of the ADMM algorithm. Ignored if mode is not 'ADMM'

        :return: a tuple of two elements: the time axis and the synthetic waveforms data
        """
        stft_operator = STFT(sr=1./dt)
        phase_retrieval = PhaseRetrieval(stft_operator=stft_operator,
                                         iteration_pr=iter_pr,
                                         rho=rho,
                                         eps=eps)
        # Generate the ground shaking using phase retrieval algorithm
        tf_synth = self.get_tfr(mw=mw, rhyp=rhyp, vs30=vs30, num_waveforms=n_waveforms)
        # reconstruct the ground shaking using PR and genrated TFR:
        gm_synth = phase_retrieval.apply_on_data(tf_synth, mode)
        time_axs = np.arange(0, gm_synth.shape[-1]) * dt
        return time_axs, gm_synth

    @staticmethod
    def get_fas_response(dt: float, gm_synth: np.ndarray) -> tuple:
        """Return the Frequency response of the generated time-history"""
        # non-normalized fft without any norm specification
        fas_synth = np.abs(np.fft.fft(gm_synth, norm="forward", axis=-1))
        n_pts = gm_synth.shape[-1] // 2
        return np.linspace(0, 0.5, num=n_pts) / dt, fas_synth[:, :n_pts]
