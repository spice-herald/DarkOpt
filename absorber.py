import numpy as np


class Absorber:

    def __init__(self, name, h, r, w_safety, tc, W, rho):
        """
        Absorber Medium of Detector. Assumed in shape of cylinder. 
        
        :param h: Height of cylinder [m]
        :param r: Radius of cylinder [m] 
        :param w_safety: Safety margin from edge where TES are not put [m]
        :param tc: Critical Temperature [K]
        :param W: Phonon Coupling Constant [Some intense SI unit combo which I forgot] 
        :param rho: Density [kg m^-3]
        """
        self._name = name
        self._h = h
        self._r = r
        self._w_safety = w_safety
        self._tc = tc
        self._W = W
        self._rho = rho
        self._volume = np.pi * (self._r ** 2) * self._h
        self._SA_face = np.pi * (self._r ** 2)
        self._SA_pattern = np.pi * (self._r - self._w_safety) ** 2
        self._SA = 2 * (self._SA_face + np.pi * self._r * self._h)
        self._m = self._rho * self._volume
        pass

    def get_name(self):
        return self._name

    def get_H(self):
        return self._h

    def get_R(self):
        return self._r

    def get_w_safety(self):
        return self._w_safety

    def get_Tc(self):
        return self._tc

    def get_phonon_coupling(self):
        return self._W

    def get_density(self):
        return self._rho

    def get_face_SA(self):
        return self._SA_face

    def get_pattern_SA(self):
        return self._SA_pattern

    def get_SA(self):
        return self._SA

    def get_mass(self):
        return self._m

    def scattering_length(self, volume, SA):
        pass