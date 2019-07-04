import numpy as np
from MaterialProperties import TESMaterial

class TES:
    # n_TES always a derived quantity, shouldn't be an input  
    def __init__(self, l, w, foverlap, n_fin, sigma, T_eq, total_res_n, 
                 material=TESMaterial(), fOp=0.45, L=0, Qp=0):
        """
        TES Class
        
        :param t: Thickness of TES [m]
        :param l: Length of TES [m]
        :param w: Width of TES [m] 
        :param foverlap: Fraction of the Al fin edge that is adjacent to the TES which is covered with Al
         w_fincon = (Perimeter/n_fin)*foverlap                 
        :param n_fin: Number of Fins to form QET
        :param resistivity: Resistivity
        :param fOp: TES Operating point resistance ratio 
        :param L: Inductance [H] 
        """
        self._t = 40e-9 # thickness is limited by fabrication constraints
        self._l = l
        self._w = w
        self._foverlap_width = foverlap # fraction overlap 
        self._resistivity = material.get_rho_electric()
        self._fOp = fOp
        self._volume_TES = self._t * self._l * self._w
        self._L = L
        #self._K = sigma * V  # P_bath vs T, eq 3.1 in thesis.
        self._n = 5  # used to define G, refer to eqs 3.1 and 3.3
        self._T_eq = T_eq  # equilibrium temperature
        self._material = material

        # Critical temperature, default 40mK
        self._T_c = material.get_Tc() #T_c
        self._Qp = Qp  # Parasitic heating

        wTc_1090 = 1.4e-3 * self._T_c / 68e-3  # [K], line 65-66 Tc_ResPt.m
        self._wTc = material.get_wTc() #0.000177496649192233 #wTc_1090 #/ 2 / np.log(3)  # Same as above, putting this in due to SimpleEquilibrium line 51
        self._material.get_gPep_v()

        # Volume of the W/Al overlap
        self._vol_WAl_overlap = 10e-6 * 2 * self._l * self._foverlap_width * self._t

        #  Volume of the W only Fin connector
        #self._vol_WFinCon =  2.5e-6 * (n_fin * 4e-6 * self._t + (2 * self._l + self._foverlap_width))
        self._vol_WFinCon = 2.5e-6 * n_fin * 4e-6 * self._t + 2.5e-6 * (2 * self._l * self._foverlap_width) * self._t

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
        # TODO this value is uncertain! Needs to be properly measured.
        self._veff_WAloverlap = 0.35

        self._volume = self._volume_TES + self._veff_WFinCon * self._vol_WFinCon + \
                       self._veff_WAloverlap * self._vol_WAl_overlap

        # Have a desired output resistance and optimise length to fix n_TES.
        self._total_res_n = total_res_n
        self._nTES = self._resistivity * self._l  / (self._w * self._t * self._total_res_n)

        self._tot_volume = self._volume * self._nTES #1185 but number of TES changed later??? 
        self._K = self._tot_volume * sigma
        # Phonon electron thermal coupling
        self._G = self._n * self._K * (T_eq ** (self._n-1))
        
        # Operating Resistance
        self._res_o = self._total_res_n * self._fOp

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

        # ------ Parameters set when getting dynamic response  -----

        # Exponential rise time for current biased circuit
        self._tau_I = 0
        # Bandwidth associated with current rise time.
        self._w_I = 0
        # L/R time constant under assumption of no pole mixing
        self._tau_el = 0
        self._w_el = 0
        # ETF time constant under assumption of no pole mixing
        self._tau_etf_simp = 0
        self._w_etf_simp = 0
        # Pole frequencies taking into account pole mixing
        self._wp_p = 0
        self._wp_m = 0
        # Pole frequency time constants
        self._taup_p = 0
        self._taup_m = 0

        # ------ Parameters set when simulating noise -----
        self._fSp_xtra = 0


        # Debugging Printing Info
       
        print("---------------- TES PARAMETERS ----------------")
        print("wTc %s" % self._wTc)
        print("Tc %s" % self._T_c)
        print("rho %s" % self._resistivity)
        print("t %s" % self._t)
        print("l %s" % self._l)
        print("w %s" % self._w)
        print("vol1TES %s" % self._volume_TES)
        print("vol1 %s" % self._volume)
        print("nTES %s" % self._nTES)
        print("volFinCon %s" % self._vol_WFinCon)
        print("WAlOverlap %s" % self._vol_WAl_overlap)
        print("veff_WFinCon %s" % self._veff_WFinCon)
        print("veff_WAloverlap %s" % self._veff_WAloverlap)
        print("Rn %s" % self._total_res_n)
        print("Ro %s" % self._res_o)
        print("fOp %s" % self._fOp)
        print("Ro %s" % self._res_o)
        print("L %s" % self._L)
        print("------------------------------------------------\n")
      

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

    def get_total_volume(self):
        return self._tot_volume
    
    # Normal Resistance of N TES in Parallel  
    def get_R(self):
        return self._total_res_n

    def get_Ro(self):
        return self._res_o

    def get_G(self):
        return self._G

    def set_G(self, val):
        self._G = val

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

    def get_material(self):
        return self._material

    def set_alpha(self, a):
        self._alpha = a

    def get_alpha(self):
        return self._alpha

    def set_beta(self, b):
        self._beta = b

    def get_beta(self):
        return self._beta

    def set_Po(self, p):
        self._Po = p

    def get_Po(self):
        return self._Po

    def set_LG(self, lg):
        self._LG = lg

    def get_LG(self):
        return self._LG

    def set_Io(self, I):
        self._Io = I

    def get_Io(self):
        return self._Io

    def set_Vbias(self, V):
        self._Vbias = V

    def get_Vbias(self):
        return self._Vbias

    def set_C(self, c):
        self._C = c

    def get_C(self):
        return self._C

    def set_tau0(self, t):
        self._tau0 = t

    def get_tau0(self):
        return self._tau0

    def set_tau_etf(self, t):
        self._tau_etf = t

    def get_tau_etf(self):
        return self._tau_etf

    def set_w_etf(self, w):
        self._w_etf = w

    def get_w_etf(self):
        return self._w_etf

    def set_tau_I(self, val):
        self._tau_I = val

    def get_tau_I(self):
        return self._tau_I

    def set_w_I(self, val):
        self._w_I = val

    def get_w_I(self):
        return self._w_I

    def set_tau_el(self, val):
        self._tau_el = val

    def get_tau_el(self):
        return self._tau_el

    def set_w_el(self, val):
        self._w_el = val

    def get_w_el(self):
        return self._w_el

    def set_tau_etf_simp(self, val):
        self._tau_etf_simp = val

    def get_tau_etf_simp(self):
        return self._tau_etf_simp

    def set_w_etf_simp(self, val):
        self._w_etf_simp = val

    def get_w_etf_simp(self):
        return self._w_etf_simp

    def set_wpp(self, val):
        self._wp_p = val

    def get_wpp(self):
        return self._wp_p

    def set_wpm(self, val):
        self._wp_m = val

    def get_wpm(self):
        return self._wp_m

    def set_taupp(self, val):
        self._taup_p = val

    def get_taupp(self):
        return self._taup_p

    def set_taupm(self, val):
        self._taup_m = val

    def get_taupm(self):
        return self._taup_m

    def set_fSp_xtra(self, val):
        self._fSp_xtra = val

    def get_fSp_xtra(self):
        return self._fSp_xtra

    def get_ntes(self):
        return self._nTES

