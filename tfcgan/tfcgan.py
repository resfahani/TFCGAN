"""
core module
"""
import os
from typing import Callable

import keras
import numpy as np
from keras.engine.functional import Functional

from tfcgan import signal_tfcgan as signal_
from tfcgan.normalization import DataNormalization


# ###############
# TFCGAN
# ###############

class TFCGAN:
    def __init__(self,
                 dirc: str = None,
                 scalemin: float = -10,
                 scalemax: float = 2.638887,
                 pwr: float = 1,
                 noise_dim: int = 100
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
        self.scalemin = scalemin   # Scaling (clipping the dynamic range)
        self.scalemax = scalemax   # Maximum value
        self. noise_dim = noise_dim  # later space

        self.num_waveforms = 2
    
    def create_scenario(self,
                        mw: float = 7,
                        rhyp: float = 10,
                        vs30: float = 760,
                        num_waveforms: int = 1,
                        mode: str = "ADMM",
                        iter_pr: int = 20,
                        rho: float = 1e-5,
                        eps: float =1e-3,
                        verbose: bool = True
                        ) -> None:
        
        """
        Generate accelerogram for one scenario
            :param mw: Magnitude value
            :param rhyp: Distance value
            :param vs30: Vs30 value
            :param num_waveforms: Number of generated time-histories
            :param mode: Type of Phase retrieval algorithm
                "ADMM": ADMM algorithm for  based on Bergman divergance
                (https://hal.archives-ouvertes.fr/hal-03050635/document)
                "GLA": Griffin-Lim Algorithm
            :param iter_pr: Number of iteration in Phase retrieval
            :param rho: ADMM parameter
            :param eps: ADMM parameter

            :return: a 5-element tuple
                self.tx: Time vector
                self.gm_synth: Generated time-history matrix
        """

        self.delete_attr  # delete the attributes

        self.num_waveforms = num_waveforms # update the number of realization
        self.iter_pr = iter_pr # update the number of iteration
        self.rho = rho # update the ADMM parameter
        self.eps = eps # update the ADMM parameter
        self.mode = mode # update the mode
        self.mw = mw # update the magnitude
        self.rhyp = rhyp # update the distance
        self.vs30 = vs30 # update the Vs30

        self.label = np.ones([self.num_waveforms, 3]) # Label vector
        self.label[:, 0] = self.label[:, 0] * self.mw # Magnitude
        self.label[:, 1] = self.label[:, 1] * self.rhyp # Distance in km
        self.label[:, 2] = self.label[:, 2] * self.vs30/1000 # Vs30 in km/s

        self.tf_synth = self.normalization.inverse(self.model.predict([self.label, self.noise_gen])[:, :, :, 0]) # simulate Time-frequency representation and descale it

        return self.__repr__()
    
    def __repr__(self) -> str:

        return (
            f"Mw: {self.mw}, "
            f"Rhyp: {self.rhyp} km, "
            f"Vs30: {self.vs30} m/s, "
            f"n_wave: {self.num_waveforms}, "
            f"PR mode: {self.mode}, "
            f"PR iter: {self.iter_pr}."
            )

    def fft(self, gm_synth: np.ndarray) -> tuple:
        # non-normalized fft without any norm specification

        if len(gm_synth.shape) == 1:
            gm_synth = gm_synth[np.newaxis, :]
        fas_synth = np.abs(np.fft.fft(gm_synth, norm="forward", axis=-1))

        return np.linspace(0, 0.5, gm_synth.shape[-1]//2)/self.stft_operator.dt, fas_synth[:, :gm_synth.shape[-1]//2]
    
    @property 
    def get_time_axs(self) -> np.ndarray:
        # get time axis

        if not hasattr(self, "gm_synth"):
            raise ValueError("Run the get_ground_shaking_synthesis method first")
        else:
            return np.arange(0, self.gm_synth.shape[-1]) * self.stft_operator.dt

    @property
    def delete_attr(self) -> None:
        # Delete the attributes for new scenario

        for value in ["tf_synth", "gm_synth", "fas_synth", 'label', 'mw', 'rhyp', 'vs30']:
            if hasattr(self, value):
                delattr(self, value)
        
    @property
    def get_tf_representation(self) -> np.ndarray:
        # Return the time-frequency representation
        if  hasattr(self, "tf_synth"):
            return self.tf_synth
        else:
            raise ValueError("Run the create_scenario method first")

    @property
    def get_ground_shaking_synthesis(self) -> tuple:
        # Generate the ground shaking using phase retrieval algorithm
        if hasattr(self, "gm_synth"):
            print(self.__repr__)
            return self.get_time_axs, self.gm_synth
        else:
            # reconstruct the ground shaking using PR and genrated TFR:
            self.gm_synth = self.phase_retrieval.apply_on_data(self.tf_synth, self.mode)
            return self.get_time_axs, self.gm_synth

    @property
    def filtered_data(self) -> np.ndarray:
        # Filter the generated time-history
        return signal_.filter_data(self.gm_synth, 0.1, 20, sr=self.stft_operator.sr,
                                   filtertype='bp', filter_order=4)

    @property
    def phase_retrieval(self) -> signal_.PhaseRetrieval:
        return signal_.PhaseRetrieval(stft_operator=self.stft_operator,
                                      iteration_pr=self.iter_pr, rho=self.rho,
                                      eps=self.eps)
    
    @property
    def model(self) -> Functional:
        # Load the trained model
        return keras.models.load_model(self.dirc)
    
    @property
    def stft_operator(self) -> signal_.STFT():
        return signal_.STFT()
    
    @property
    def get_tfr(self) -> np.ndarray:
        # return the time-frequency representation
        return self.tf_synth

    @property
    def normalization(self) -> DataNormalization:
        # return the normalization function
        return DataNormalization(scalemin = self.scalemin, scalemax = self.scalemax, pwr = self.pwr)

    @property
    def noise_gen(self) -> np.ndarray:
        # Generate random noise vector
        return np.random.normal(0, 1, (self.num_waveforms, self.noise_dim))
    
    @property
    def get_fas_response(self) -> tuple:
        if not hasattr(self, "gm_synth"):
            _, _ = self.get_ground_shaking_synthesis
        # Frequency response of the generated time-history
        self.freq, self.fas_synth = self.fft(self.gm_synth) 
        return self.freq, self.fas_synth

    # def save_ground_shaking(self, dirc: str = None) -> None:
    #     # Save the generated ground shaking
    #     if dirc is None:
    #         dirc = os.path.abspath(os.path.join(
    #             os.path.dirname(os.path.dirname(__file__)), 'results'))
    #
    #     os.makedirs(dirc, exist_ok=True)
    #     dirc = os.path.join(dirc, f"mw_{self.mw}_rhyp_{self.rhyp}_vs30_{self.vs30}.npz")
    #
    #     if (hasattr(self, "gm_synth") and hasattr(self, "gm_synth")
    #             and hasattr(self, "tf_synth") and hasattr(self, "fas_synth")):
    #         np.savez(dirc, label=self.label, GM_synthesis=self.gm_synth,
    #                  fas_synthesis=self.fas_synth, tf_synthesis=self.tf_synth,
    #                  freq=self.freq)
    #         print(f"Ground shaking scenario with mw:{self.mw}, "
    #               f"Rhyp:{self.rhyp}, vs30:{self.vs30} is saved in {dirc}")
    #     else:
    #         raise ValueError("Run the get_ground_shaking_synthesis "
    #                          "and get_fas_response first")
