import numpy as np
import math as m
from MaterialProperties import TESMaterial

class TES:
    """
    Class to store TES properties
    """
    def __init__(self, length, width, l_overlap, n_fin, sigma, rn, rl, L, h=40e-9,
                 zeta_WAl_fin=0.45, zeta_W_fin=0.88, con_type='ellipse',
                 material=TESMaterial(), operating_point=0.45, alpha=None, beta=0, 
                 n=5, Qp=0, t_mc=10e-3, lgcprint=False):
        
        """
        length : float
            length of TES in [m]
        width : float
            width of TES in [m]
        l_overlap : float
            lenght of Al/W overlap region in [m]
        n_fin : int
            number of Al fins for QET
        sigma : float
            Electron-phonon coupling constant [W/K^5/m^3]
        rn : float 
            Normal state resistance of channel in [Ohms]
        rl : float
            The load resistance of the TES bias circuit, 
            i.e. the shunt resistance + the parasitic resistance
            on the TES side [Ohms]
        L : float
            Inductance (SQUID input coil + parasitic wire
            inductance) [H]
        h : float
            thickness of TES in [m]
        zeta_WAl_fin : float, optional
            eff factor for the contribution of the W/Al overlap 
            region to the volume and heat capacity (0,1)
        zeta_W_fin : float, optional
            eff factor for the contribution of the W only part 
            of the fin connector to the volume and heat capacity (0,1)
        con_type : string, optional
            Either 'ellipse' or 'modern' fin connector style
        material : Material Object, optional
            Object containing W properties
        operating_point : float, optional
            The operational resistance [%Rn]
            (fractional percentage of normal resistance)
        alpha : float, optional
            The logarithmic temperature sensitivity.
            If None, it will be estimated.
        beta : float, optional
            The logarithmic current sensitivity.
            If not changed, the small beta (beta=0) 
            approximation will be used.
        n : int, optional
            Thermal powerlaw exponent
        Qp : float, optional
            Parasitic heating [J]
        t_mc : float, optional
            Temperature of the mixing chamber [K]
        lgcprint : bool, optional
            If True, the TES parameters are printed
        
            
        
        """
  
        self._h = h # thickness of the TES. limited by fabrication constraints + noise. same as matlab. 
        self._l = length # length of tes
        self._w = width # width of tes 
        self._rn = rn
        self._rl = rl
        self._L = L
        #self._foverlap_width = foverlap # fraction overlap SZ: this will now be calculated
        # The next two shouldn't change (to avoid shorts)
        self._width_no_Al = 12e-6 # width around TES where no Al
        self._Al_erase = 6e-6 # width between fins with no Al 
        self._foverlap_width = (2*l+2*self._width_no_Al+4*l_overlap-n_fin*self._Al_erase)/(2*l+2*self._width_no_Al+4*l_overlap)
        self._l_overlap = l_overlap # length of W/Al overlap 
        self._n_fin = n_fin # number of fins to form QET  
        self._resistivity = material._rho_electrical # electrical resistivity of TES 
        self._fOp = operating_point # Operating Resistance/Normal Resistance ratio 
        # volume of a single TES 
        self._volume_TES = self._h * self._l * self._w
        self._L = 0 # [H] set Inductance to zero for now
        #self._K = sigma * V  # P_bath vs T, eq 3.1 in thesis.
        self._sigma = sigma
        self._n = n  # used to define G, refer to eqs 3.1 and 3.3
        self._beta = beta
        self._material = material
        self._con_type = con_type 
    
        # Critical temperature, default 40mK
        self._T_c = material._Tc # Critical temperature of W 
        self._Qp = Qp  # Parasitic heating
        self._t_mc = t_mc
        
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
                self._vol_WAl_overlap = (0.5*3.1415*l_overlap*l_overlap+36e-12)*(n_fin -2)*self._h + 2*(0.5*3.1415*l_overlap*l_overlap+36e-12)*self._h

        # elliptical connector type for small TES designs (cm squares/cubes)
        elif con_type == 'ellipse':
                wempty = 6e-6 
                wempty_tes = 7.5e-6
                con_major = (self._l/2) + l_overlap 
                con_minor = l_overlap+ wempty_tes 
                con_ellipse = 3.1415*con_major*con_minor 
                self._A_overlap = con_ellipse - 2*self._l*wempty_tes - wempty*n_fin*l_overlap 
                self._vol_WAl_overlap = self._A_overlap*self._h  

        #  Volume of the W only Fin connector
        #self._vol_WFinCon =  2.5e-6 * (n_fin * 4e-6 * self._t + (2 * self._l + self._foverlap_width))
        #self._vol_WFinCon = 2.5e-6 * n_fin * 4e-6 * self._t + 2.5e-6 * (2 * self._l * self._foverlap_width) * self._t
        # re-estimate for new desing (PD4):
        self._vol_WFinCon = n_fin*(19.040e-12+ 2*0.605e-12)*self._h  

        # Volume of the W only portion of the fin connector
        # Since the temperature in the fin connector is lower than the temperature
        # in the TES, the effective volume is smaller than the true volume
        # This is the efficiency factor for the volume of the fin connector
        # contributing to Gep ... we're assuming that this is also the efficiency
        # factor for the volume contributing to heat capacity as well.
        self._veff_WFinCon = zeta_W_fin

        # Volume of the W/Al overlap portion of the fin connector
        # The W/Al portion is completely proximitized ... it should have a very low
        # effective volume
        # TODO this value is uncertain! Needs to be properly measured.
        # self._veff_WAloverlap = 0.35
        self._veff_WAloverlap = zeta_WAL_fin # -- changed to .45 to match matlab - SZ 2019  

        self._volume = self._volume_TES + self._veff_WFinCon * self._vol_WFinCon + \
                       self._veff_WAloverlap * self._vol_WAl_overlap

        # Resistance of 1 TES 
        self._res1tes = self._resistivity*self._l/(self._w*self._h)
        # Have a desired output resistance and optimise length to fix n_TES.
        self._total_res_n = total_res_n
        self._nTES = m.ceil(self._resistivity * self._l  / (self._w * self._h * self._total_res_n))
        self._total_res_n = self._res1tes/self._nTES
        
        self._tot_volume = self._volume * self._nTES
        #self._tot_volume = 9.7e-14 # MESSING 
        self._K = self._tot_volume * sigma

        
        
        # Operating Resistance
        self._ro = self._rn * self._fOp
        
        
        
        
        ######
        
    


        #  ---- Bias Point Temperature ----
        # let's calculate the temperature of the operating point resistance.
        # [Notice that if the resistance of the TES changes with current, then this doesn't work]

        zeta_o = np.log(_TES._fOp/(1 - _TES._fOp))/2
    
        # Attempting to replicate SimpleEquilibrium_1TES line 51 with Tc_ResPt.m line 65-66. wTc doesn't show up
        # anywhere else! wTc calculated using this way in TES.py 32-33



        self._t0 = zeta_o * self._wTc + self._T_c # K
    
        self._Gep = self._n * self._K * self._t0 ** (self._n-1)
    

        # ----- Alpha/Beta at Transition Point -----

        self._alpha = 2*self._t0/self._wTc/np.exp(zeta_o)/(np.exp(zeta_o)+np.exp(-zeta_o))



        # ---- TES properties at equilibrium ----

        # Bias Power (Phonon/Electron coupling G already set in TES.py)
        self._p0 = self._K * ((self._t0 ** self._n) - (self._t_mc ** self._n)) - self._Qp # W

        # Loop Gain
        self._LG = self._alpha * self._p0 / (self._t0 * self._G)


        # Current At Equilibrium
        self._i0 = np.sqrt(self._p0/self._r0)

        # Bias Voltage
        self._vbias = self._i0 * (self._rl + self._r0)  # V


        # Heat Capacity

        # Tungsten values taken from MaterialProperties.m line 385 / 376
        fCsn = material._fCsn # matches matlab 
        gC_v = _material._gC_v # matches matlab 
        self._C = fCsn * gC_v * self._t0 * self._tot_volume


        # Inverse Bandwidth
        self._tau0 = self._C/self._G


        # Sensor Bandwidth
        r_ratio = self._rl / self._r0
        self._tau_etf = self._tau0 / (1 + self._LG * (1 - r_ratio)/(1 + self._beta + r_ratio))
        self._w_etf(1/self._tau_etf)


        
        
        #####
        
        

        # ------ Parameters to be set later when simulating equilibrium -----

       

        # ------ Parameters set when getting dynamic response  -----

        # Exponential rise time for current biased circuit
        self._tau_I = self._tau0 / (1 - self._LG)
        # Bandwidth associated with current rise time.
        self._w_I = 1/self_.tau_I
        # L/R time constant under assumption of no pole mixing
        self._tau_el = self._L / (self._rl + self._r0 * (1 + self._beta))
        self._w_el = 1/self_.tau_el
   
        # Pole frequencies taking into account pole mixing
        wp_avg = (1/self._tau_el)/2+(1/self._tau_I)/2
        dw = np.sqrt(((1 / self._tau_el) - (1 / self._tau_I))**2-4*(self._r0/self._L)*self._LG*(2+self._beta)/self._tau0)/2
        self._wp_p = wp_avg + dw
        self._wp_m = wp_avg - dw
        # Pole frequency time constants
        self._taup_p = 1/self._wp_p
        self._taup_m = 1/self._wp_m

        # ------ Parameters set when simulating noise -----
        self._fSp_xtra = 0
        
        
        
        if lgcprint:
            print("---------------- TES PARAMETERS ----------------")
            print("sigma %s" % self._sigma)
            print("wTc %s" % self._wTc)
            print("Tc %s" % self._T_c)
            print("rho %s" % self._resistivity)
            print("t %s" % self._h)
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
      
    
    

