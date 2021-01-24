import numpy as np
import math as m
from scipy import constants
from darkopt.materials._MaterialProperties import TESMaterial

class TES:
    """
    Class to store TES properties. Calculated the TES and fin connector volumes.
    """
    def __init__(self, length, width, l_overlap, n_fin, sigma, rn, rsh, rp, L_tot, tload=30e-3,
                 w_overlap='circle', w_fin_con=2.5e-6, h=40e-9, veff_WAloverlap=0.45, veff_WFinCon=0.88, con_type='ellipse',
                 material=TESMaterial(), operating_point=0.3, alpha=None, beta=0, 
                 wempty_fin=6e-6, wempty_tes=6e-6, n=5, Qp=0, t_mc=10e-3, l_c=5e-6, 
                 w_overlap_stem=4e-6,  l_overlap_pre_ellipse=2e-6):
        
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
        rsh : float
            The shunt resistance of the TES bias circuit [Ohms] 
        rp : float
            The parasitic resistance on the TES side [Ohms]
        L_tot : float
            total inductance (SQUID input coil + parasitic wire
            inductance) [H]
        tload : float, optional
            The effective noise temperature for the passive johnson 
            noise from the shunt resistor and parasitic resistance
            in [Ohms]
        w_overlap : float, NoneType, string, optional
            Width of the W/Al overlap region (if None, a rough estimate is used
            for area calulations) if 'circle', then w_overlap
            is set to be 2*l_overlap to make a circle. [m]
        w_fin_con : float, optional
            Width of the W only part of the fin connector. Defaults
            to the standard width of the TES of 2.5e-6. [m]
        h : float
            thickness of TES in [m]
        veff_WAloverlap : float, optional
            eff factor for the contribution of the W/Al overlap 
            region to the volume and heat capacity (0,1)
        veff_WFinCon : float, optional
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
        wempty_fin : float, optional
            ? width of empty slot in the fin? [m]
        wempty_tes : float, optional
            ? width of empty space between TES and Al [m]
        n : int, optional
            Thermal powerlaw exponent
        Qp : float, optional
            Parasitic heating [J]
        t_mc : float, optional
            Temperature of the mixing chamber [K]
        l_c : float, optional
            The length of the fin connector before it widens
            to connect to the Al
        w_overlap_stem : float, optional
            The wider part of the conector at the Al [m]
        l_overlap_pre_ellipse : float, optional
            The length of the rectangular part of the overlap
            region before the half ellipse [m]
        """
  
        self.h = h # thickness of the TES. limited by fabrication constraints + noise. same as matlab. 
        self.l = length # length of tes
        self.w = width # width of tes 
        self.rn = rn
        self.fOp = operating_point # Operating Resistance/Normal Resistance ratio 
        self.r0 = self.rn * self.fOp # operating resistance
        self.rsh = rsh
        self.rp = rp
        self.rl = rsh + rp
        self.L = L_tot
        self.tload = tload
        self.l_c = l_c
        self.w_overlap_stem = w_overlap_stem
        self.l_overlap_pre_ellipse = l_overlap_pre_ellipse
        
       
        # The next two shouldn't change (to avoid shorts)
        self.width_no_Al = 12e-6 # width around TES where no Al # what is the difference between this and 
                                                                # wempty_tes?
        self.Al_erase = 6e-6 # width between fins with no Al # what is the difference between this and 
                                                             # wempty_fin?
        self.wempty_fin = wempty_fin
        self.wempty_tes = wempty_tes
        
        self.foverlap_width = (2*length+2*self.width_no_Al+4*l_overlap-n_fin*self.Al_erase)/(2*length+2*self.width_no_Al+4*l_overlap) # what is this used for???? 
        self.l_overlap = l_overlap # length of W/Al overlap 
        if w_overlap == 'circle':
            w_overlap = 2*l_overlap
        self.w_overlap = w_overlap
        self.w_fin_con = w_fin_con
        self.n_fin = n_fin # number of fins to form QET  
        self.resistivity = material._rho_electrical # electrical resistivity of TES 
        
        self.volume_TES = self.h * self.l * self.w # volume of a single TES 
        self.sigma = sigma
        self.n = n  # used to define G, refer to eqs 3.1 and 3.3
        self.beta = beta
        self.material = material
        self.con_type = con_type 
    
        self.tc = material._Tc # Critical temperature of W 
        self.Qp = Qp  # Parasitic heating
        self.t_mc = t_mc
        wTc_1090 = 1.4e-3 * self.tc / 68e-3  # [K], line 65-66 Tc_ResPt.m [Not Used in TES]
        self.wTc = material._wTc 
        self.material._gPep_v

        
        # need new estimate for "modern" fin connectors...
        # instead of l_overlap want r of fin... l_overlap = radius of circle
        if con_type == 'modern':
                self.A_overlap = (0.5*3.1415*l_overlap*l_overlap+36e-12)*(n_fin -2) + 2*(0.5*3.1415*l_overlap*l_overlap+36e-12)
                self.vol_WAl_overlap = (0.5*3.1415*l_overlap*l_overlap+36e-12)*(n_fin -2)*self.h + 2*(0.5*3.1415*l_overlap*l_overlap+36e-12)*self.h

        # elliptical connector type for small TES designs (cm squares/cubes)
        elif con_type == 'ellipse':
            if self.w_overlap is None:
                # not really sure where this calculation is comming from...
                con_major = (self.l/2) + l_overlap 
                con_minor = l_overlap + self.wempty_tes 
                con_ellipse = np.pi*con_major*con_minor 
                self.A_overlap = con_ellipse - 2*self.l*self.wempty_tes - self.wempty_fin*n_fin*l_overlap 
                self.vol_WAl_overlap = self.A_overlap*self.h  
            else:
                self.A_overlap = ((np.pi*self.w_overlap/2*self.l_overlap)/2 +  \
                                  self.l_overlap_pre_ellipse*self.w_overlap_stem)*n_fin
                self.vol_WAl_overlap = self.A_overlap*self.h
           
        
        
        # re-estimate for new desing (PD4):
        #self.vol_WFinCon = n_fin*(19.040e-12+ 2*0.605e-12)*self.h  #### where do these number come from?????
        self.vol_WFinCon =  ((self.w_fin_con*self.l_c * (self.n_fin - 2) )+\
                             (self.wempty_tes-self.l_c)*w_overlap_stem * \
                             (self.n_fin - 2)) * self.h # area of W part of fin connector
                                                                                        # -2 because end fins don't have same 
                                                                                        # excess W
        # Volume of the W only portion of the fin connector
        # Since the temperature in the fin connector is lower than the temperature
        # in the TES, the effective volume is smaller than the true volume
        # This is the efficiency factor for the volume of the fin connector
        # contributing to Gep ... we're assuming that this is also the efficiency
        # factor for the volume contributing to heat capacity as well.
        # This comes from 2017 UCB measurements of QP chips
        self.veff_WFinCon = veff_WFinCon #zeta_W_fin

        # Volume of the W/Al overlap portion of the fin connector
        # The W/Al portion is completely proximitized ... it should have a very low
        # effective volume
        # This comes from 2017 UCB measurements of QP chips
        self.veff_WAloverlap = veff_WAloverlap 

        self.volume = self.volume_TES + self.veff_WFinCon * self.vol_WFinCon + \
                       self.veff_WAloverlap * self.vol_WAl_overlap
        
        self.zeta = self.volume_TES/(self.volume + self.volume_TES)
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
        self.t0 = zeta_o * self.wTc + self.tc # K    
        self.Gep = self.n * self.K * self.t0 ** (self.n-1)
    
        # ----- Alpha/Beta at Transition Point -----
        if alpha is None:
            self.alpha = 2*self.t0/self.wTc/np.exp(zeta_o)/(np.exp(zeta_o)+np.exp(-zeta_o))
        else:
            self.alpha = alpha
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
        
        self.phase_sep_legth()
        
    def phase_sep_legth(self):
        """
        Function to check maximum phase seperation legth based on QET parameters.
        Eq 4.25 in Matt Pyles Thesis (Page 102) 
        Matt Christopher Pyle. Optimizing the design and analysis of 
        cryogenic semiconductor dark matter detectors for maximum 
        sensitivity. PhD thesis, Stanford U., 2012.
        """
        
        beta_wf = 1/3*(np.pi*constants.k/constants.e)**2 #weidermann-franz coeff
        zeta = self.zeta
        sig = self.sigma
        n = self.n
        tc = self.tc
        rho = self.resistivity
        alpha = self.alpha
        tb = self.t_mc
        num = np.pi**2*beta_wf*zeta
        denom = n*sig*tc**(n-2)*rho*(alpha/n * (1-tb**n/tc**n) - 1 )
        
        self.max_phase_length =  np.sqrt(num/denom)
        self.is_phase_sep = self.max_phase_length < self.l
        
        
    def print(self):
        """
        Method to print TES parameters
        """

        print("---------------- TES PARAMETERS ----------------")
        print(f"sigma = {self.sigma}"  )
        print("wTc =  %s" % self.wTc)
        print("Tc =  %s" % self.tc)
        print("rho =  %s" % self.resistivity)
        print("t =  %s" % self.h)
        print("l =  %s" % self.l)
        print("w =  %s" % self.w)
        print("foverlap =  %s" % self.foverlap_width)
        print("res1tes =  %s" % self.res1tes)
        print("n_fin =  %s" % self.n_fin)
        print("vol1TES =  %s" % self.volume_TES)
        print("vol1 =  %s" % self.volume)
        print(f'Zeta = {self.zeta}')
        print(f'Max TES length before phase sep = {self.max_phase_length} [m]')
        print(f'Is TES phase seperated = {self.is_phase_sep}')
        print("nTES =  %s" % self.nTES)
        print("tot_volume =  %s" % self.tot_volume)
        print("K =  %s " % self.K)
        print("volFinCon =  %s" % self.vol_WFinCon)
        print("WAlOverlap =  %s" % self.vol_WAl_overlap)
        print("veff_WFinCon =  %s" % self.veff_WFinCon)
        print("veff_WAloverlap =  %s" % self.veff_WAloverlap)
        print("Rn =  %s" % self.rn)
        print("R0 =  %s" % self.r0)
        print("fOp =  %s" % self.fOp)
        print("P0 =  %s" % self.p0)
        print("L =  %s" % self.L)
        print(f"tau_el = {self.tau_el}")
        print(f"tau_etf = {self.tau_etf}")
        print(f"tau_0 = {self.tau0}")
        print(f"tau_+ = {self.taup_p}")
        print(f"tau_- = {self.taup_m}")
        print("------------------------------------------------\n")
      
    
    

