import numpy as np
class TES:

    def __init__(self, t, l, w, foverlap, n_fin, resistivity, sigma, V, n, T_eq, T_c=40e-3, fOp=0.45, L=0, Qp=0):
        """
        TES Class
        
        :param t: Thickness of TES [m]
        :param l: Length of TES [m]
        :param w: Width of TES [m] 
        :param foverlap: Fraction of the Al fin edge that is adjacent to the TES which is covered with Al 
        :param n_fin: Number of Fins to form QET
        :param resistivity: Resistivity
        :param fOp: TES Operating point resistance ratio 
        :param L: Inductance [H] 
        """
        self._t = t
        self._l = l
        self._w = w
        self._foverlap_width = foverlap
        self._resistivity = resistivity
        self._fOp = fOp
        self._volume_TES = self._t * self._l * self._w
        self._L = L
        self._K = sigma * V  # P_bath vs T, eq 3.1 in thesis.
        self._n = n  # used to define G, refer to eqs 3.1 and 3.3
        self._T_eq = T_eq  # equilibrium temperature

        # Phonon electron thermal coupling
        self._G = n * self._K * (T_eq ** (n-1))

        # Critical temperature, default 40mK from MaterialProperties.m line 427
        self._T_c = T_c
        self._Qp = Qp  # Parasitic heating

        wTc_1090 = 1.4e-3 * self._T_c / 68e-3  # [K], line 65-66 Tc_ResPt.m
        self._wTc = wTc_1090 / 2 / np.log(3)  # Same as above, putting this in due to SimpleEquilibrium line 51


        # Volume of the W/Al overlap
        self._vol_WAl_overlap = 10e-6 * 2 * self._l * self._foverlap_width * self._t

        #  Volume of the W only Fin connector
        self._vol_WFinCon =  2.5e-6 * (n_fin * 4e-6 * self._t + (2 * self._l + self._foverlap_width))

        # Volume of the W only portion of the fin connector
        # Since the temperature in the fin connector is lower than the temperature
        # in the TES, the effective volume is smaller than the true volume
        # This is the efficiency factor for the volume of the fin connector
        # contributing to Gep ... we're assuming that this is also the efficiency
        # factor for the volume contributing to heat capacity as well.
        self._veff_WFinCon = 0.88

        # Volume of the W/Al overlap portion of the fin connector
        # The W/Al portion is completely proximitized ... it should have a very low
        # effective volume
        self._veff_WAloverlap = 0.13

        self._volume = self._volume_TES + self._veff_WFinCon * self._vol_WFinCon + \
                       self._veff_WAloverlap * self._vol_WAl_overlap

        self._res = self._resistivity * self._l / (self._w * self._t)

        # Operating Resistance
        self._res_o = self._res * self._fOp

        # ------ Parameters to be set later when simulating equilibrium -----

        # Alpha
        self._alpha = 0
        # Beta
        self._beta = 0
        # Heat Capacity
        self._C = 0
        # Power (relies on fridge information so not set in here)
        self._Po = 0
        # Loop Gain
        self._LG = 0
        # Equilibrium Current
        self._Io = 0
        # Bias Voltage
        self._Vbias = 0
        # Inverse naive bandwidth T_0
        self._tau0 = 0
        # Inverse effective bandwidth T_etf
        self._tau_etf = 0
        # Effective Bandwidth
        self._w_etf = 0


    def get_T(self):
        return self._t

    def get_L(self):
        return self._l

    def get_W(self):
        return self._w

    def get_overlap_width(self):
        return self._foverlap_width

    def get_volume(self):
        return self._volume

    def get_R(self):
        return self._res

    def get_Ro(self):
        return self._res_o

    def get_G(self):
        return self._G

    def get_Tc(self):
        return self._T_c

    def get_To(self):
        return self._T_eq

    def get_fOp(self):
        return self._fOp

    def set_To(self, T):
        self._T_eq = T

    def set_Qp(self, q):
        self._Qp = q

    def get_wTc(self):
        return self._wTc

    def get_n(self):
        return self._n

    def get_K(self):
        """K = Sigma * Volume defined from thesis eq 3.1 page 18"""
        return self._K

    def set_alpha(self, a):
        self._alpha = a

    def set_beta(self, b):
        self._beta = b

    def set_Po(self, p):
        self._Po = p

    def set_LG(self, lg):
        self._LG = lg

    def set_Io(self, I):
        self._Io = I

    def set_Vbias(self, V):
        self._Vbias = V

    def set_C(self, c):
        self._C = c

    def set_tau0(self, t):
        self._tau0 = t

    def set_tau_etf(self, t):
        self._tau_etf = t

    def set_w_etf(self, w):
        self._w_etf = w