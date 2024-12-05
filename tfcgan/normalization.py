import numpy as np


class DataNormalization:
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
        return self.scalemin, self.scalemax, self.pwr  # noqa