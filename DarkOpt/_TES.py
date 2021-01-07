import numpy as np
import math as m
from MaterialProperties import TESMaterial

class TES:
    """
    Class to store TES properties. Calculated the TES and fin connector volumes.
    """
    def __init__(self, length, width, l_overlap, n_fin, sigma, rn, rl, L_tot, h=40e-9,
                 zeta_WAl_fin=0.45, zeta_W_fin=0.88, con_type='ellipse',
                 material=TESMaterial(), operating_point=0.45, alpha=None, beta=0, 
                 n=5, Qp=0, t_mc=10e-3):
        
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
        L_tot : float
            total inductance (SQUID input coil + parasitic wire
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
        """
  
        self.h = h # thickness of the TES. limited by fabrication constraints + noise. same as matlab. 
        self.l = length # length of tes
        self.w = width # width of tes 
        self.rn = rn
        self.fOp = operating_point # Operating Resistance/Normal Resistance ratio 
        self.r0 = self.rn * self.fOp # operating resistance
        self.rl = rl
        self.L = L_tot
       
        # The next two shouldn't change (to avoid shorts)
        self.width_no_Al = 12e-6 # width around TES where no Al
        self.Al_erase = 6e-6 # width between fins with no Al 
        
        self.foverlap_width = (2*length+2*self.width_no_Al+4*l_overlap-n_fin*self.Al_erase)/(2*length+2*self.width_no_Al+4*l_overlap)
        self.l_overlap = l_overlap # length of W/Al overlap 
        self.n_fin = n_fin # number of fins to form QET  
        self.resistivity = material._rho_electrical # electrical resistivity of TES 
        
        self.volume_TES = self.h * self.l * self.w # volume of a single TES 
        self.sigma = sigma
        self.n = n  # used to define G, refer to eqs 3.1 and 3.3
        self.beta = beta
        self.material = material
        self.con_type = con_type 
    
        self.T_c = material._Tc # Critical temperature of W 
        self.Qp = Qp  # Parasitic heating
        self.t_mc = t_mc
        wTc_1090 = 1.4e-3 * self.T_c / 68e-3  # [K], line 65-66 Tc_ResPt.m [Not Used in TES]
        self.wTc = material._wTc 
        self.material._gPep_v

        
        # need new estimate for "modern" fin connectors...
        # instead of l_overlap want r of fin... l_overlap = radius of circle
        if con_type == 'modern':
                self.A_overlap = (0.5*3.1415*l_overlap*l_overlap+36e-12)*(n_fin -2) + 2*(0.5*3.1415*l_overlap*l_overlap+36e-12)
                self.vol_WAl_overlap = (0.5*3.1415*l_overlap*l_overlap+36e-12)*(n_fin -2)*self.h + 2*(0.5*3.1415*l_overlap*l_overlap+36e-12)*self.h

        # elliptical connector type for small TES designs (cm squares/cubes)
        elif con_type == 'ellipse':
                wempty = 6e-6 
                wempty_tes = 7.5e-6
                con_major = (self.l/2) + l_overlap 
                con_minor = l_overlap+ wempty_tes 
                con_ellipse = 3.1415*con_major*con_minor 
                self.A_overlap = con_ellipse - 2*self.l*wempty_tes - wempty*n_fin*l_overlap 
                self.vol_WAl_overlap = self.A_overlap*self.h  

        #  Volume of the W only Fin connector
        #self._vol_WFinCon =  2.5e-6 * (n_fin * 4e-6 * self._t + (2 * self._l + self._foverlap_width))
        #self._vol_WFinCon = 2.5e-6 * n_fin * 4e-6 * self._t + 2.5e-6 * (2 * self._l * self._foverlap_width) * self._t
        # re-estimate for new desing (PD4):
        self.vol_WFinCon = n_fin*(19.040e-12+ 2*0.605e-12)*self.h  

        # Volume of the W only portion of the fin connector
        # Since the temperature in the fin connector is lower than the temperature
        # in the TES, the effective volume is smaller than the true volume
        # This is the efficiency factor for the volume of the fin connector
        # contributing to Gep ... we're assuming that this is also the efficiency
        # factor for the volume contributing to heat capacity as well.
        self.veff_WFinCon = zeta_W_fin

        # Volume of the W/Al overlap portion of the fin connector
        # The W/Al portion is completely proximitized ... it should have a very low
        # effective volume
        # TODO this value is uncertain! Needs to be properly measured.
        # self._veff_WAloverlap = 0.35
        self.veff_WAloverlap = zeta_WAl_fin # -- changed to .45 to match matlab - SZ 2019  

        self.volume = self.volume_TES + self.veff_WFinCon * self.vol_WFinCon + \
                       self.veff_WAloverlap * self.vol_WAl_overlap

        # Resistance of 1 TES 
        self.res1tes = self.resistivity*self.l/(self.w*self.h)
        # Have a desired output resistance and optimise length to fix n_TES.
        self.nTES = m.ceil(self.resistivity * self.l  / (self.w * self.h * self.rn))
        self.tot_volume = self.volume * self.nTES
        self.K = self.tot_volume * sigma
        

        #  ---- Bias Point Temperature ----
        # let's calculate the temperature of the operating point resistance.
        # [Notice that if the resistance of the TES changes with current, then this doesn't work]
        zeta_o = np.log(self.fOp/(1 - self.fOp))/2
        self.t0 = zeta_o * self.wTc + self.T_c # K    
        self.Gep = self.n * self.K * self.t0 ** (self.n-1)
    
        # ----- Alpha/Beta at Transition Point -----
        self.alpha = 2*self.t0/self.wTc/np.exp(zeta_o)/(np.exp(zeta_o)+np.exp(-zeta_o))

        # ---- TES properties at equilibrium ----
        self.p0 = self.K * ((self.t0 ** self.n) - (self.t_mc ** self.n)) - self.Qp # W

        # Loop Gain
        self.LG = self.alpha * self.p0 / (self.t0 * self.Gep)

        # Current At Equilibrium
        self.i0 = np.sqrt(self.p0/self.r0)

        # Bias Voltage
        self.vbias = self.i0 * (self.rl + self.r0)  # V

        # Heat Capacity
        fCsn = material._fCsn # matches matlab 
        gC_v = material._gC_v # matches matlab 
        self.C = fCsn * gC_v * self.t0 * self.tot_volume

        # Inverse Bandwidth
        self.tau0 = self.C/self.Gep

        # Sensor Bandwidth
        r_ratio = self.rl / self.r0
        self.tau_etf = self.tau0 / (1 + self.LG * (1 - r_ratio)/(1 + self.beta + r_ratio))
        self.w_etf = 1/self.tau_etf

        # Exponential rise time for current biased circuit
        self.tau_I = self.tau0 / (1 - self.LG)
        # Bandwidth associated with current rise time.
        self.w_I = 1/self.tau_I
        # L/R time constant under assumption of no pole mixing
        self.tau_el = self.L / (self.rl + self.r0 * (1 + self.beta))
        self.w_el = 1/self.tau_el
   
        # Pole frequencies taking into account pole mixing
        wp_avg = (1/self.tau_el)/2+(1/self.tau_I)/2
        dw = np.sqrt(((1 / self.tau_el) - (1 / self.tau_I))**2-4*(self.r0/self.L)*self.LG*(2+self.beta)/self.tau0)/2
        self.wp_p = wp_avg + dw
        self.wp_m = wp_avg - dw
        # Pole frequency time constants
        self.taup_p = 1/self.wp_p
        self.taup_m = 1/self.wp_m

        # ------ Parameters set when simulating noise -----
        self.fSp_xtra = 0
        
        
        
    def print(self):
        """
        Method to print TES parameters
        """

        print("---------------- TES PARAMETERS ----------------")
        print(f"sigma = {self.sigma}"  )
        print("wTc =  %s" % self.wTc)
        print("Tc =  %s" % self.T_c)
        print("rho =  %s" % self.resistivity)
        print("t =  %s" % self.h)
        print("l =  %s" % self.l)
        print("w =  %s" % self.w)
        print("foverlap =  %s" % self.foverlap_width)
        print("res1tes =  %s" % self.res1tes)
        print("n_fin =  %s" % self.n_fin)
        print("vol1TES =  %s" % self.volume_TES)
        print("vol1 =  %s" % self.volume)
        print("nTES =  %s" % self.nTES)
        print("tot_volume =  %s" % self.tot_volume)
        print("K =  %s " % self.K)
        print("volFinCon =  %s" % self.vol_WFinCon)
        print("WAlOverlap =  %s" % self.vol_WAl_overlap)
        print("veff_WFinCon =  %s" % self.veff_WFinCon)
        print("veff_WAloverlap =  %s" % self.veff_WAloverlap)
        print("Rn =  %s" % self.rn)
        print("P0 =  %s" % self.r0)
        print("fOp =  %s" % self.fOp)
        print("P0 =  %s" % self.p0)
        print("L =  %s" % self.L)
        print(f"tau_el = {self.tau_el}")
        print(f"tau_etf = {self.tau_etf}")
        print(f"tau_0 = {self.tau0}")
        print(f"tau_+ = {self.taup_p}")
        print(f"tau_- = {self.taup_m}")
        print("------------------------------------------------\n")
      
    
    

