from math import *
import numpy as np
from _MaterialProperties import DetectorMaterial

class Absorber:

    def __init__(self, name, shape, h, r, w_safety):
        """
        Absorber Medium of Detector. Assumed in shape of cylinder. 
        
        :param h: Height of cylinder [m]
        :param r: Radius of cylinder [m] 
        :param w_safety: Safety margin from edge where TES are not put [m]
        :param tc: Critical Temperature [K]
        :param W: Phonon Coupling Constant [Some intense SI unit combo which I forgot] 
        :param rho: Density [kg m^-3]
        """
        material = DetectorMaterial(name)

        self._name = name
        self._shape = shape 
        self._h = h # if cube - length of one side
        self._r = r # if cube r = h, if square r is side length of square
        self._w_safety = w_safety  
        self._rho = material.get_rho_mass()
        if shape == "cylinder":
            self._volume = np.pi * (self._r**2) * self._h
            self._SA_face = np.pi * (self._r**2)
            self._SA_pattern = np.pi * (self._r - self._w_safety)**2
            self._SA = 2 * (self._SA_face + np.pi * self._r * self._h)
        elif shape == "square":
            self._volume = self._h*(self._r**2)
            self._SA_face = self._r**2
            self._SA_pattern = (self._r - 2*self._w_safety)**2
            self._SA = 2*self._SA_face + self._h*4*self._r  
        elif shape == "cube": 
            self._volume = self._h**3
            self._SA_face = self._h**2
            self._SA_pattern = (self._h -2*self._w_safety)**2
            self._SA = 6*self._SA_face
        else:
            print("Wrong Shape.")
        self._m = self._rho * self._volume
        pass

        
        
    def print(self):
        print("---------------- ABSORBER PARAMETERS ----------------")
        print("Absorber lscat ", 4*self._volume/self._SA)
        print("Absorber SA_face %s" % self._SA_face)
        print("Absorber SA %s" % self._SA)
        print("Absorber mass %s" % self._m)
        print("------------------------------------------------\n")

    def get_name(self):
        return self._name

    def get_H(self):
        return self._h

    def get_R(self):
        return self._r

    def get_w_safety(self):
        return self._w_safety

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

    def scattering_length(self):
    # In ballistic limited transport, the average scattering length will depend upon the absorber geometry 
    # From the super simple sabine equation 
    # (http://courses.physics.illinois.edu/phys406/Lecture_Notes/P406POM_Lecture_Notes/Derivation_of_the_Sabine_Equation.pdf)
    # Inputs:
    #   1) Absorber Volume
    #    2) Absorber Surface Area
    # 12/6/13: MCP
        return 4 * self._volume / self._SA



