"""
core module
"""
import os
import numpy as np

from keras.models import load_model  # noqa
from keras.engine.functional import Functional  # noqa

from tfcgan.normalization import DataNormalization
from tfcgan.signals import ADMM, GLA


def get_ground_shaking_synthesis(
        n_waveforms: int, *, mw: float = 7, rhyp: float = 10, vs30: float = 760,
        mode: str = "ADMM", iter_pr: int = 20, rho: float = 1e-5, eps: float = 1e-3,
        scalemin: float = -10, scalemax: float = 2.638887, pwr: float = 1
) -> tuple:
    """
    Generate synthetic waveforms (time histories) from the given scenario

    :param n_waveforms: Number of generated time-histories
    :param mw: Magnitude value
    :param rhyp: Distance value
    :param vs30: Vs30 value
    :param mode: Type of Phase retrieval algorithm
        "ADMM" (the default): ADMM algorithm for  based on Bergman divergence
        (https://hal.archives-ouvertes.fr/hal-03050635/document)
        "GLA": Griffin-Lim Algorithm
    :param iter_pr: Number of iteration in Phase retrieval
    :param rho: parameter of the ADMM algorithm. Ignored if mode is not 'ADMM'
    :param eps: parameter of the ADMM algorithm. Ignored if mode is not 'ADMM'
    :param scalemin: Scale factor in pre-processing step (min. value)
    :param scalemax: Scale factor in pre-processing step (max. value)
    :param pwr: Power spectrum in pre-processing step:
        1: means absolute value
        2: spectrogram

    :return: a tuple of two elements: the time axis and the synthetic waveforms data
    """

    if mode == 'ADMM':
        phase_retrieval = ADMM(iter_pr, rho=rho, eps=eps)
    elif mode == 'GLA':
        phase_retrieval = GLA(iter_pr)
    else:
        raise ValueError('mode should be in ("ADMM", "GLA")')
    # Generate the ground shaking using phase retrieval algorithm
    tf_synth = get_tfr(mw=mw, rhyp=rhyp, vs30=vs30, num_waveforms=n_waveforms,
                       scalemin=scalemin, scalemax=scalemax, pwr=pwr)
    # reconstruct the ground shaking using PR and generated TFR:
    gm_synth = phase_retrieval.apply_on_data(tf_synth)
    dt = 0.01
    time_axs = np.arange(0, gm_synth.shape[-1]) * dt
    return time_axs, gm_synth


def get_tfr(
        mw: float = 7, rhyp: float = 10, vs30: float = 760, num_waveforms: int = 1,
        noise_dim: int = 100, scalemin: float = -10, scalemax: float = 2.638887,
        pwr: float = 1
) -> np.ndarray:
    """
    Generate a scenario and return the time-frequency representation

    :param mw: Magnitude value
    :param rhyp: Distance value
    :param vs30: Vs30 value
    :param num_waveforms: Number of generated time-histories
    :param noise_dim: Noise dimension
    :param scalemin: Scale factor in pre-processing step (min. value)
    :param scalemax: Scale factor in pre-processing step (max. value)
    :param pwr: Power spectrum in pre-processing step:
        1: means absolute value
        2: spectrogram
    """
    model = init_model()

    label = np.ones([num_waveforms, 3])
    label[:, 0] = label[:, 0] * mw  # Magnitude
    label[:, 1] = label[:, 1] * rhyp  # Distance in km
    label[:, 2] = label[:, 2] * vs30 / 1000  # Vs30 in km/s

    noise_gen = np.random.normal(0, 1, (num_waveforms, noise_dim))
    normalization = DataNormalization(scalemin=scalemin, scalemax=scalemax, pwr=pwr)
    # simulate Time-frequency representation and descale it:
    tf_synth = normalization.inverse(
        model.predict([label, noise_gen])[:, :, :, 0]
    )
    return tf_synth


_model = None  # global var (lazy loaded)


def init_model(cache=True):
    """Reload and return the trained model in this package
    :param cache: bool (default true) use already loaded model if available.
        When False: force reload from file
    """
    global _model
    if not cache or _model is None:
        _model = load_model(os.path.abspath(os.path.join(
                            os.path.dirname(os.path.dirname(__file__)), 'model')))
    return _model


def get_fas_response(dt: float, gm_synth: np.ndarray) -> tuple:
    """Return the Frequency response of the time-histories generated
    with `get_ground_shaking_synthesis`

    :param dt: the delta t, in s. Called t the first argument of
        `self.get_ground_shaking_synthesis`, then `dt = t[1] - t[0]`
    :param gm_synth: the synthetic waveforms (second argument of
        `self.get_ground_shaking_synthesis`)
    """
    # non-normalized fft without any norm specification
    fas_synth = np.abs(np.fft.fft(gm_synth, norm="forward", axis=-1))
    n_pts = gm_synth.shape[-1] // 2
    return np.linspace(0, 0.5, num=n_pts) / dt, fas_synth[:, :n_pts]
