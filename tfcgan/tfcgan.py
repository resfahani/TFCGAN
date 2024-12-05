"""
core module
"""
import os

import keras
import numpy as np
from keras.engine.functional import Functional

from tfcgan import signal_tfcgan as signal_
from tfcgan.normalization import DataNormalization
from tfcgan.signal_tfcgan import STFT


# ###############
# TFCGAN
# ###############

class TFCGAN:
    def __init__(self,
                 dirc: str = None,
                 scalemin: float = -10,
                 scalemax: float = 2.638887,
                 pwr: float = 1
                 ) -> None:
        
        """        
        :param dirc: Trained model directory
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
            self._model = keras.models.load_model(self.dirc)
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
                                     n_pts: int = 4000,
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
        :param n_pts: waveforms number of points (starting from 0 and equally spaced by
            `dt` sec.)
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
        stft_operator = STFT(sr=1./dt, length=n_pts)
        phase_retrieval = signal_.PhaseRetrieval(stft_operator=stft_operator,
                                                 iteration_pr=iter_pr,
                                                 rho=rho,
                                                 eps=eps)
        # Generate the ground shaking using phase retrieval algorithm
        tf_synth = self.get_tfr(mw=mw, rhyp=rhyp, vs30=vs30, num_waveforms=n_waveforms)
        # reconstruct the ground shaking using PR and genrated TFR:
        gm_synth = phase_retrieval.apply_on_data(tf_synth, mode)
        time_axs = np.arange(0, gm_synth.shape[-1]) * dt
        return time_axs, gm_synth

    #FIXME: unnecessary code:
    # @property
    # def normalization(self) -> DataNormalization:
    #     # return the normalization function
    #     return DataNormalization(scalemin = self.scalemin, scalemax = self.scalemax, pwr = self.pwr)
    # @property
    # def stft_operator(self) -> signal_.STFT():
    #     return signal_.STFT()
    # def __repr__(self) -> str:
    #     return (
    #         f"Mw: {self.mw}, "
    #         f"Rhyp: {self.rhyp} km, "
    #         f"Vs30: {self.vs30} m/s, "
    #         f"n_wave: {self.num_waveforms}, "
    #         f"PR mode: {self.mode}, "
    #         f"PR iter: {self.iter_pr}."
    #         )
    # @property
    # def delete_attr(self) -> None:
    #     # Delete the attributes for new scenario
    #     for value in ["tf_synth", "gm_synth", "fas_synth", 'label', 'mw', 'rhyp',
    #                   'vs30']:
    #         if hasattr(self, value):
    #             delattr(self, value)
    # @property
    # def phase_retrieval(self) -> signal_.PhaseRetrieval:
    #     return signal_.PhaseRetrieval(stft_operator=self.stft_operator,
    #                                   iteration_pr=self.iter_pr, rho=self.rho,
    #                                   eps=self.eps)
    # @property
    # def get_tfr(self) -> np.ndarray:
    #     # return the time-frequency representation
    #     return self.tf_synth
    # @property
    # def noise_gen(self) -> np.ndarray:
    #     # Generate random noise vector
    #     return np.random.normal(0, 1, (self.num_waveforms, self.noise_dim))
    # @property
    # def get_time_axs(self) -> np.ndarray:
    #     # get time axis
    #     if not hasattr(self, "gm_synth"):
    #         raise ValueError("Run the get_ground_shaking_synthesis method first")
    #     else:
    #         return np.arange(0, self.gm_synth.shape[-1]) * self.stft_operator.dt
    # @property
    # def get_tf_representation(self) -> np.ndarray:
    #     # Return the time-frequency representation
    #     if  hasattr(self, "tf_synth"):
    #         return self.tf_synth
    #     else:
    #         raise ValueError("Run the create_scenario method first")



    # unused code (please add tests when restoring it)

    # @property
    # def get_fas_response(self) -> tuple:
    #     if not hasattr(self, "gm_synth"):
    #         _, _ = self.get_ground_shaking_synthesis
    #     # Frequency response of the generated time-history
    #     self.freq, self.fas_synth = self.fft(self.gm_synth)
    #     return self.freq, self.fas_synth
    #
    # def fft(self, gm_synth: np.ndarray) -> tuple:
    #     # non-normalized fft without any norm specification
    #
    #     if len(gm_synth.shape) == 1:
    #         gm_synth = gm_synth[np.newaxis, :]
    #     fas_synth = np.abs(np.fft.fft(gm_synth, norm="forward", axis=-1))
    #
    #     return (np.linspace(0, 0.5, gm_synth.shape[-1]//2) /
    #             self.stft_operator.dt, fas_synth[:, :gm_synth.shape[-1]//2])
    #
    # @property
    # def filtered_data(self) -> np.ndarray:
    #     # Filter the generated time-history
    #     return signal_.filter_data(self.gm_synth, 0.1, 20, sr=self.stft_operator.sr,
    #                                filtertype='bp', filter_order=4)
    #

