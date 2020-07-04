import numpy as np
import math as m
from MaterialProperties import TESMaterial

class TES:
    # n_TES always a derived quantity, shouldn't be an input  
    def __init__(self, l, w, l_overlap, n_fin, sigma, T_eq, total_res_n, e_WAl, con_type,
                 material=TESMaterial(), printing=False,  fOp=0.45, Qp=0):
        """
	:param foverlap: Fraction of the Al fin edge that is adjacent to the TES which is covered with Al
         w_fincon = (Perimeter/n_fin)*foverlap                 
        """
        self._t = 40e-9 # thickness of the TES. limited by fabrication constraints + noise. same as matlab. 
        self._l = l # length of tes
        self._w = w # width of tes 
        #self._foverlap_width = foverlap # fraction overlap SZ: this will now be calculated
        # The next two shouldn't change (to avoid shorts)
        self._width_no_Al = 12e-6 # width around TES where no Al
        self._Al_erase = 6e-6 # width between fins with no Al 
        self._foverlap_width = (2*l+2*self._width_no_Al+4*l_overlap-n_fin*self._Al_erase)/(2*l+2*self._width_no_Al+4*l_overlap)
        self._l_overlap = l_overlap # length of W/Al overlap 
        self._n_fin = n_fin # number of fins to form QET  
        self._resistivity = material._rho_electrical # electrical resistivity of TES 
        self._fOp = fOp # Operating Resistance/Normal Resistance ratio 
        # volume of a single TES 
        self._volume_TES = self._t * self._l * self._w
        self._L = 0 # [H] set Inductance to zero for now
        #self._K = sigma * V  # P_bath vs T, eq 3.1 in thesis.
        self._sigma = sigma
        self._n = 5  # used to define G, refer to eqs 3.1 and 3.3
        self._T_eq = T_eq  # equilibrium temperature
        self._material = material
        self._con_type = con_type 

        # Critical temperature, default 40mK
        self._T_c = material._Tc # Critical temperature of W 
        self._Qp = Qp  # Parasitic heating

        wTc_1090 = 1.4e-3 * self._T_c / 68e-3  # [K], line 65-66 Tc_ResPt.m [Not Used in TES]
        self._wTc = material._wTc #0.000177496649192233 #wTc_1090 #/ 2 / np.log(3)  # Same as above, putting this in due to SimpleEquilibrium line 51
        self._material._gPep_v


        # Volume of the W/Al overlap
        # These two assume PD2 Style Fin: 
        #self._vol_WAl_overlap = l_overlap * 2 * self._l * self._foverlap_width * self._t # Matt's estimate
        #self._vol_WAl_overlap = (2*l+ 2*self._width_no_Al+ 4*l_overlap)*l_overlap*self._foverlap_width*self._t #SZ: new estimate
        # need new estimate for "modern" fin connectors...
        # instead of l_overlap want r of fin... l_overlap = radius of circle
        if con_type == 'modern':
                self._A_overlap = (0.5*3.1415*l_overlap*l_overlap+36e-12)*(n_fin -2) + 2*(0.5*3.1415*l_overlap*l_overlap+36e-12)
                self._vol_WAl_overlap = (0.5*3.1415*l_overlap*l_overlap+36e-12)*(n_fin -2)*self._t + 2*(0.5*3.1415*l_overlap*l_overlap+36e-12)*self._t
	
	# elliptical connector type for small TES designs (cm squares/cubes)
        elif con_type == 'ellipse':
                wempty = 6e-6 
                wempty_tes = 7.5e-6
                con_major = (self._l/2) + l_overlap 
                con_minor = l_overlap+ wempty_tes 
                con_ellipse = 3.1415*con_major*con_minor 
                self._A_overlap = con_ellipse - 2*self._l*wempty_tes - wempty*n_fin*l_overlap 
                self._vol_WAl_overlap = self._A_overlap*self._t  

        #  Volume of the W only Fin connector
        #self._vol_WFinCon =  2.5e-6 * (n_fin * 4e-6 * self._t + (2 * self._l + self._foverlap_width))
        #self._vol_WFinCon = 2.5e-6 * n_fin * 4e-6 * self._t + 2.5e-6 * (2 * self._l * self._foverlap_width) * self._t
        # re-estimate for new desing (PD4):
        self._vol_WFinCon = n_fin*(19.040e-12+ 2*0.605e-12)*self._t  

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
        # self._veff_WAloverlap = 0.35
        self._veff_WAloverlap = e_WAl # -- changed to .45 to match matlab - SZ 2019  

        self._volume = self._volume_TES + self._veff_WFinCon * self._vol_WFinCon + \
                       self._veff_WAloverlap * self._vol_WAl_overlap

        # Resistance of 1 TES 
        self._res1tes = self._resistivity*self._l/(self._w*self._t)
        # Have a desired output resistance and optimise length to fix n_TES.
        self._total_res_n = total_res_n
        self._nTES = m.ceil(self._resistivity * self._l  / (self._w * self._t * self._total_res_n))
        self._total_res_n = self._res1tes/self._nTES
        
        self._tot_volume = self._volume * self._nTES
        #self._tot_volume = 9.7e-14 # MESSING 
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

        self._print = printing
        if self._print == True:
            print("---------------- TES PARAMETERS ----------------")
            print("sigma %s" % self._sigma)
            print("wTc %s" % self._wTc)
            print("Tc %s" % self._T_c)
            print("rho %s" % self._resistivity)
            print("t %s" % self._t)
            print("l %s" % self._l)
            print("w %s" % self._w)
            print("foverlap %s" % self._foverlap_width)
            print("res1tes %s" % self._res1tes)
            print("n_fin %s" % self._n_fin)
            print("vol1TES %s" % self._volume_TES)
            print("vol1 %s" % self._volume)
            print("nTES %s" % self._nTES)
            print("tot_volume %s" % self._tot_volume)
            print("K %s " % self._K)
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
      
    
    def set_G(self, val):
        self._G = val
   
    def set_To(self, T):
        self._T_eq = T

    def set_Qp(self, q):
        self._Qp = q
    
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

    def set_tau_I(self, val):
        self._tau_I = val

    def set_w_I(self, val):
        self._w_I = val

    def set_tau_el(self, val):
        self._tau_el = val

    def set_w_el(self, val):
        self._w_el = val

    def set_tau_etf_simp(self, val):
        self._tau_etf_simp = val

    def set_w_etf_simp(self, val):
        self._w_etf_simp = val

    def set_wpp(self, val):
        self._wp_p = val

    def set_wpm(self, val):
        self._wp_m = val

    def set_taupp(self, val):
        self._taup_p = val

    def set_taupm(self, val):
        self._taup_m = val

    def set_fSp_xtra(self, val):
        self._fSp_xtra = val

