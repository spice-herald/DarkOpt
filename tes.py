class TES:

    def __init__(self, t, l, w, foverlap, n_fin, resistivity, fOp=0.45, L=0):
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
