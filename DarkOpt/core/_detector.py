from darkopt.core import TES, QET, Absorber
from darkopt.materials._MaterialProperties import TESMaterial, DetectorMaterial, QETMaterial
from darkopt.utils._utils import _line, arc_patch, calc_angles
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm  
from matplotlib import patches
import numpy as np
import scipy.constants as constants
import qetpy as qp


#import matplotlib.colors as mcolors
from matplotlib import cm  

nice_fonts = {
        # Use LaTeX to write all text
        #"text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 14,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
}

rcParams.update(nice_fonts)



class Detector:

    def __init__(self, 
                 absorber, 
                 QET, 
                 n_channel=1, 
                 w_rail_main=6e-6, 
                 w_railQET=4e-6, 
                 bonding_pad_area=4.5e-8,
                 freqs=None,
                 passive=1,
                 equal_spaced=True,
                ):
        
        """
        Parameters:
        -----------
        absorber : Absorber object
            Absorber object that holds parameters about
            target geometry and material
        QET : QET object
            QET object that hold all QET and TES 
            related parameters
        n_channel : int, optional
            The number of channels in the detector
        w_rail_main : float, optional
            The width of the main bias rials. By
            default this is set to 6e-6 [m]
        w_railQET : float, optional
            The width of the secondary bias lines
            connecting the the QETs to the main rails.
            by default it is set to 4e-6. [m]
        bonding_pad_area : float, optional
            The area of passive Al used for the total
            number of bonding pads needed. The default
            is set to the area of 2 150um by 150um pads. [m^2]
        freqs : Array, None-type, optional
            The array of frequencies to be used
            for noise modeling and energy resolution
            calculationg. If None, freqs will be set
            to freqs=np.linspace(.1, 1e6, int(1e5))
        passive : int, optional
            leftover parameter for testing purposes. 
            leave set to 1
        equal_spaced : bool, optional
            If True, the QETs are spread out evenly 
            accross the instrumented surface area
            of the detector. If False, the QETs
            are spread out equally in one dimension, 
            but not in the other. (ie, the secondary
            bias rails are not used so a sparse design
            can still be close packed)
        """
        # Width of Main Bias Rails and QET Rails 
        self.w_rail_main = w_rail_main
        self.w_railQET = w_railQET

        #self._name = name
        self._absorber = absorber
        self._n_channel = n_channel
        self._l_fin = QET.l_fin
        self._h_fin = QET.h_fin
        self._l_overlap = QET.TES.l_overlap
        self.QET = QET 
        tes = QET.TES
        if freqs is None:
            freqs = np.linspace(.1, 1e6, int(1e5))
        self.freqs = freqs
        self.bonding_pad_area = bonding_pad_area
        self.eres = None
        self.equal_spaced = equal_spaced
 

        # ------------- QET Fins ----------------------------------------------
        # Surface area covered by QET Fins 
        self._SA_active = n_channel * tes.nTES * QET.a_fin

        # Average area per cell, and corresponding length
        a_cell = self._absorber.get_pattern_SA() / (n_channel * tes.nTES) # Assuming one sided detector
       
        # calculate length and width of QET
        y_qet = 2 * QET.l_fin + tes.l # length QET 
        x_qet = 2*QET.l_fin + 2*tes.wempty_tes+tes.w # width of QET
        QET_block = y_qet * x_qet
        
        if QET_block*tes.nTES > self._absorber.get_pattern_SA(): 
            #print("----- ERROR: Invalid Design - QET cells don't fit.")
            self._cells_fit = False
        else: self._cells_fit = True
        
        self._l_cell = np.sqrt(a_cell)
        self._w_cell = np.sqrt(a_cell/2) # hypothetical optimum but only gives a couple percent decrease in passive Al
        self._h_cell = 2*self._w_cell

        if self.equal_spaced:
            if self._l_cell > y_qet:
                #print("---- Not Close Packed")
                # Design is not close packed. Get passive Al/QET
                a_passiveQET = self._l_cell * self.w_rail_main + (self._l_cell - y_qet) * self.w_railQET
                self._close_packed = False
            else:
                #print("---- Close Packed")
                # Design is close packed. No vertical rail to QET
                x_cell = a_cell / y_qet
                a_passiveQET = x_cell* self.w_rail_main
                self._close_packed = True
        else:
            a_passiveQET = x_qet*1.1 * self.w_rail_main

        tes_passive = a_passiveQET * n_channel * tes.nTES
        
        # Passive Al Rails for PD2 Like Layout
        
        # Calc Alignment Mark Passive Area 
        one_alignment_window = 20772e-12 
        total_alignment = 5*one_alignment_window
        self.total_alignment = total_alignment
        #total_alignment = 0.0

        # Total Passive Surface Area
        # 1. TES Passive Area
        # 2. Outer Ring
        # 3. Inner Ring
        # 4. Inner Vertical Rail
        # 5. Outer Vertical Rail
        # 6. Alignment Marks
        # 7. bonding pads
        if absorber._shape == "cylinder": # Indicates PD2-like Rail Layout
            outer_ring = 2 * np.pi * (self._absorber.get_R() - self._absorber.get_w_safety()) * self.w_rail_main
            inner_ring = outer_ring / (np.sqrt(2))
            inner_vertical_rail = 3 * (self._absorber.get_R() - self._absorber.get_w_safety()) * self.w_rail_main * (1 - np.sqrt(2)/2.0)
            outer_vertical_rail = (self._absorber.get_R() - self._absorber.get_w_safety()) * self.w_rail_main * (1 + np.sqrt(2)/2.0)

            self._SA_passive = tes_passive + outer_ring + inner_ring + inner_vertical_rail \
                                + outer_vertical_rail + total_alignment + self.bonding_pad_area
        if ((absorber._shape == "square") | (absorber._shape == 'cube')): # New Square Rail Layout Design
            if passive == 1:
#                 self._SA_passive = tes_passive + 2*(self._absorber._width - 2*self._absorber._w_safety)*self.w_rail_main \
#                                                  + self.bonding_pad_area
                self._SA_passive = tes_passive
            elif passive == 0:            
                self._SA_passive = 0 # FOR THEORETICAL UNDERSTANDING, DELETE  
        
        # Fraction of surface area which has phonon absorbing aluminum
        self._fSA_qpabsorb = (self._SA_passive + self._SA_active) / self._absorber.get_SA()

        # Fraction of Al which is QET fin which can produce signal
        self._ePcollect = self._SA_active / (self._SA_active + self._SA_passive)
 
        PD2_absb_time = 20e-6
        absb_lscat = absorber.scattering_length()
        PD2_fSA_qpabsb = 0.0214 # percentage of total surface area 
        PD2_lscat = 0.0019488

        #print("-- -- Phonon Absorption Time -- --")
        #print("      PD2 Abs Time:    ", PD2_absb_time)
        #print("      PD2 Scat Length: ", PD2_lscat)
        #print("      PD2 fSA QP Absb: ", PD2_fSA_qpabsb)
        #print("      Scat Length      ", absb_lscat)
        #print("      fSA WP Absb      ", self._fSA_qpabsorb)
        
        # Scale collection time based of PD2 time. Where does this come from?
        self._t_pabsb = PD2_absb_time * (absb_lscat / PD2_lscat) * (PD2_fSA_qpabsb / self._fSA_qpabsorb)

        self._w_collect = 1/self._t_pabsb

        # ------------ Total Phonon Collection Efficiency -------------

        # The loss mechanisms in our detector are:
        # 1) subgap downconversion of athermal phonons in the crystal
        # 2) collection of athermal phonons by passive metal on the surface of our detector ( Det.ePcollect)
        # 3) Efficiency of QP production in Al fin (QET.ePQP)
        # 4) Efficiency of QP transport to TES (QET.eQPabsb)
        # 5) Energy conversion efficiency at W/Al interface
        # 6) ?

        # Phonon Downconversion Factor
        self._e_downconvert = 1/1000 
        self._e_downconvert = 1/4000

        aluminum = QETMaterial("Al")

        #### these are not used?
        #pE_thresh = 2*1.76*constants.k*aluminum._Tc
        #p_baresurface = (self._absorber.get_SA() - self._SA_active - self._SA_passive)/self._absorber.get_SA()
        #p_subgap = p_baresurface**3000
        #p_notsubgap = 1-p_subgap
        
        # Let's combine 1), 5), and 6) together and assume that it is the same as the measured/derived value from iZIP4
        self._e156 = 0.8690 # should scale with Al coverage.... 

        # Total collection efficiency:
        self._eEabsb = self._e156 * self._ePcollect * self.QET.eQPabsb * self.QET.ePQP # * self._e_downconvert * self._fSA_qpabsorb 
        
        self.fSA_active = self._SA_active/self._absorber.get_SA()
        self.fSA_passive = self._SA_passive/self._absorber.get_SA()

        # ------------ Thermal Conductance to Bath ---------------
        self._kpb = 1.55e-4 #### where did this come from/does it even do anything?
        # Thermal conductance coefficient between detector and bath
        self._nkpb = 4

        #Simulate noise
        self.noise = qp.sim.TESnoise(freqs=self.freqs,
                                     rload=tes.rl,
                                     r0=tes.r0,
                                     rshunt=tes.rsh,
                                     beta=tes.beta,
                                     loopgain=tes.LG,
                                     inductance=tes.L,
                                     tau0=tes.tau0,
                                     G=tes.Gep,
                                     qetbias=tes.vbias/tes.rsh,
                                     tc=tes.t0,
                                     tload=tes.tload,
                                     tbath=tes.t_mc,
                                     n=tes.n,
                                     )
        
    def calc_res(self):
        eres = qp.sim.energy_res_estimate(self.freqs, 
                                          self._t_pabsb, 
                                          self.noise.s_ptot(),
                                          self._eEabsb)
        self.eres = eres
        return self.eres
    
    def plot_si(self, figsize=(6.75, 4.455), xlims=None, ylims=None):
        noise = self.noise
        f = self.freqs
        
        s_ielec = noise.s_isquid()
        s_itfn = noise.s_itfn()
        s_ites = noise.s_ites()
        s_iload = noise.s_iload()
        s_itot = noise.s_itot()
  
        cmap = 'viridis'
        cmap = cm.get_cmap(cmap)
        cs = cmap(np.linspace(0, 1,5))
       
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(f, np.sqrt(s_itfn), color=cs[3],
               linewidth=1.8, label='TFN', zorder = 30, linestyle='-')
        
        ax.plot(f, np.sqrt(s_ites), color=cs[1],
               linewidth=1.8, label='TES', zorder = 10, linestyle='-')
        
        ax.plot(f, np.sqrt(s_iload), color=cs[0],
               linewidth=1.8, label='Load', zorder = 20, linestyle=':')
        ax.plot(f, np.sqrt(s_ielec),  color=cs[4],
               linewidth=1.8, label='Electronics', zorder = 5, linestyle='--')
        ax.plot(f, np.sqrt(s_itot), color='k',
               linewidth=1.8, label='Total', zorder = 200, linestyle='-')
        ax.set_ylabel(r'Current Noise [A/$\sqrt{\mathrm{Hz}}$]')
        ax.tick_params(which="both", direction="in", right=True, top=True, zorder=300)
        ax.set_xlabel(r'Frequency [Hz]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.grid(True, alpha=.5, linestyle='--')
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        plt.legend(facecolor='white', framealpha=1)
        ax.set_title('Current Referenced Noise')
        fig.tight_layout()
        
    def plot_sp(self, figsize=(6.75, 4.455), xlims=None, ylims=None):
        noise = self.noise
        f = self.freqs
        
        s_pelec = noise.s_psquid()
        s_ptfn = noise.s_ptfn()
        s_ptes = noise.s_ptes()
        s_pload = noise.s_pload()
        s_ptot = noise.s_ptot()

        cmap = 'viridis'
        cmap = cm.get_cmap(cmap)
        cs = cmap(np.linspace(0, 1,5))
       
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(f, np.sqrt(s_ptfn), color=cs[3],
               linewidth=1.8, label='TFN', zorder = 30, linestyle='-')
        
        ax.plot(f, np.sqrt(s_ptes), color=cs[1],
               linewidth=1.8, label='TES', zorder = 10, linestyle='-')
        
        ax.plot(f, np.sqrt(s_pload), color=cs[0],
               linewidth=1.8, label='Load', zorder = 20, linestyle=':')
        ax.plot(f, np.sqrt(s_pelec),  color=cs[4],
               linewidth=1.8, label='Electronics', zorder = 5, linestyle='--')
        ax.plot(f, np.sqrt(s_ptot), color='k',
               linewidth=1.8, label='Total', zorder = 200, linestyle='-')
        ax.set_ylabel(r'NEP [W/$\sqrt{\mathrm{Hz}}$]')
        ax.tick_params(which="both", direction="in", right=True, top=True, zorder=300)
        plt.grid(True, alpha=.5, linestyle='--')
        ax.set_xlabel(r'Frequency [Hz]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        #plt.loglog(f, np.abs(1e-16/(1+1j*2*np.pi*f*20e-6)), linestyle ='-', lw=1.5, 
        #           color = 'xkcd:periwinkle', zorder=5000000) #pulse shape
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        plt.legend(facecolor='white', framealpha=1)
        ax.set_title('Power Referenced Noise')
        fig.tight_layout()
        
    def plot_responsivity(self, figsize=(6.75, 4.455), xlims=None, ylims=None):
        noise = self.noise
        f = self.freqs
        fig, ax = plt.subplots(figsize=figsize)
        ax.loglog(f, np.abs(noise.dIdP())/np.abs(noise.dIdP())[0], color='k')
        ax.set_xlabel(r'Frequency [Hz]')
        ax.tick_params(which="both", direction="in", right=True, top=True, zorder=300)
        plt.grid(True, alpha=.5, linestyle='--')
        plt.title('Responsivity')
        ax.set_ylabel(r'$\left|\frac{\partial I}{\partial P}\left(\omega\right)\right|/\left|\frac{\partial I}{\partial P}\left(0\right)\right|$')
        
        
        
    def print(self):    
        print("---------------- DETECTOR PARAMETERS ----------------")
        print("nP =  %s" % self._n_channel)
        print("SAactive =  %s" % self._SA_active)
        print("fSA_active =  %s" % self.fSA_active)
        print("lcell =  %s" % self._l_cell)
        print("SApassive =  %s" % self._SA_passive)
        print(f'fraction total Al cov =  {self._fSA_qpabsorb}')
        print("fSA_passive =  %s" % self.fSA_passive)
        print("Alignment_area =  %s" % self.total_alignment)
        #print("fSA_QPabsb =  %s" % self._fSA_qpabsorb)
        print("ePcollect =  %s" % self._ePcollect)
        print("tau_pabsb =  %s" % self._t_pabsb)
        print("w_pabsb =  %s" % (1/self._t_pabsb))
        print("eE156 =  %s" % self._e156)
        print("QP_eff =  %s" % self.QET.eQPabsb)
        print("eEabsb =  %s" % self._eEabsb)
        print("Kpb =  %s" % self._kpb)
        print("nKpb =  %s" % self._nkpb)
        print("NQET.TES =  %s" % self.QET.TES.nTES)
        print("total_L =  %s" % self.QET.TES.L)
        print(f"equal spaced = {self.equal_spaced}")
        print("------------------------------------------------\n")
        
    def plot_qet(self, xlims=None, ylims=None, figsize=(6.75, 6.75)):
        """
        Plots visualization of QET design. Currently only
        works for 4 fin design.
        
        Note, in order to maintain an equal aspect ratio, 
        the distance of the range of xlims and ylims must 
        be the same.
        
        xlims : 2-tuple, optional
            limits of plotting [um]
        ylims : 2-tuple, optional
            limits of plotting [um]
        figsize : tuple, optional
            Size of figure to be drawn
        """
        
        if self.QET.TES.n_fin == 4:
            _plot_qet(self, xlims=xlims, ylims=ylims, figsize=figsize)
        else:
            print('Currently only supported for 4 fin designs')
            
    def calc_angles(self):
        """
        Calculated the angles of the line dividing the Al ellipse.
        Currently only works with 4 fin desings
        """
        if self.QET.TES.n_fin == 4:
            return calc_angles(self)
        else:
            print('Currently only supported for 4 fin designs')
       


def create_detector(tes_length, tes_l_overlap, rn,  l_fin, h_fin, n_fin,
                    abs_type, abs_shape, abs_height, abs_width, w_safety,
                    tes_width, sigma, rp, L_tot,  ahole, n_channel=1,
                    rsh=5e-3, tload=30e-3, tes_h=40e-9, zeta_WAl_fin=0.45, 
                    zeta_W_fin=0.88, con_type='ellipse', material=TESMaterial(), 
                    operating_point=0.45, alpha=None, beta=0,  n=5, Qp=0, 
                    t_mc=10e-3, ePQP=0.52, eff_absb = 1.22e-4, wempty=6e-6, 
                    wempty_tes=7.5e-6, type_qp_eff=0, freqs=None, equal_spaced=True):
    """
    Helper function to create Absorber, TES, QET, and Detector 
    objects. A detector object is returned
    
    Parameters:
    -----------
    tes_length : float
        length of TES in [m]
    tes_l_overlap : float
        lenght of Al/W overlap region in [m]
    l_fin : float, 
        Length of Al fins [m]
    h_fin : float, 
        hight of Al fins [m]
    n_fin : int
        number of Al fins for QET  
    abs_type : string
        detector substrate type, can be
        either 'Si' or 'Ge'
    abs_shape : string 
        the geometric shape of the 
        absorber 'cylinder', 'square', or 'cube' 
    abs_height : float
        the height of the absorber [m]
    abs_width : float
        If a square or cube, the length is assumed
        to be the same as the width. If the shape
        is a cylinder, then the width
    w_safety : float, 
        Safety margin from edge where TES are not put [m]
    tes_width : float
        width of TES in [m]
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
    ahole : float
        area of holes in fin [m^2]
    n_channel : int, optional
        Number of chennels in detector 
    tload : float, optional
        The effective noise temperature for the passive johnson 
        noise from the shunt resistor and parasitic resistance
        in [Ohms]
    tes_h : float
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
    
    ePQP : float, optional
        Phonon to QP Conversion Effciency, 
        Kaplan downconversion limits this to 52%
    eff_absb : float, optional
        W/Al transmition/trapping probability
    wempty_fin : float, optional
        ?
    wempty_tes : float, optional
        ?
    w_overlap : float, optional
    type_qp_eff : int, optional
        how the efficiency should be calculated.
        0 : 'modern' estimate of overlap radius (small)
        1 : 'modern' estimate, but different effective l_overlap
        2 : Matt's method
    freqs : array-like
        frequencies used for plotting noise and 
        to calculate the expected energy resolution
    equal_spaced : bool, optional
            If True, the QETs are spread out evenly 
            accross the instrumented surface area
            of the detector. If False, the QETs
            are spread out equally in one dimension, 
            but not in the other. (ie, the secondary
            bias rails are not used so a sparse design
            can still be close packed)  
    """
    
    absorb = Absorber(name=abs_type, shape=abs_shape, 
                      height=abs_height, width=abs_width, 
                      w_safety=w_safety)
    
    tes = TES(length=tes_length, width=tes_width, l_overlap=tes_l_overlap, 
              n_fin=n_fin, sigma=sigma, rn=rn, rp=rp, L_tot=L_tot, rsh=rsh, 
              tload=tload, h=tes_h, zeta_WAl_fin=zeta_WAl_fin, zeta_W_fin=zeta_W_fin,
              con_type=con_type, material=material, operating_point=operating_point,
              alpha=alpha, beta=beta, n=n, Qp=Qp, t_mc=t_mc)
    
    qet = QET(l_fin=l_fin, h_fin=h_fin, TES=tes, ahole=ahole, ePQP=ePQP,
              eff_absb=eff_absb, wempty=wempty, wempty_tes=wempty_tes, 
              type_qp_eff=type_qp_eff)
    
    det = Detector(absorber=absorb, QET=qet, passive=1, 
                   n_channel=n_channel, freqs=freqs, equal_spaced=equal_spaced)
    
    return det
    
              
    
def _plot_qet(det, xlims=None, ylims=None, figsize=(6.75, 6.75)):
    """
    Function to plot the QET based on the optimum params. 
    This is only a visual aid and not exact. 
    
    Note, in order to maintain an equal aspect ratio, 
    the distance of the range of xlims and ylims must 
    be the same.
    
    Parameters:
    -----------
    det : detector object
    xlims : 2-tuple, optional
        limits of plotting [m]
    ylims : 2-tuple, optional
        limits of plotting [m]
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    
    fig, ax = plt.subplots(figsize=figsize)

    qet = det.QET
    tes = det.QET.TES
    
    b = qet.l_fin + tes.l/2
    a = qet.l_fin + tes.wempty_tes + tes.w/2

    # Al ellipse
    e1 = patches.Ellipse((0, 0), a*2, b*2,
                         angle=0, linewidth=2, fill=True, color='xkcd:blue',
                         zorder=2) 
    # remove white space
    e2 = patches.Rectangle((-(tes.w+tes.wempty_tes*2)/2,-tes.l/2),
                           tes.w+tes.wempty_tes*2, 
                           tes.l,
                           color='xkcd:white', zorder=4)
    # TES line
    e3 = patches.Rectangle((-tes.w/2,-tes.l/2), tes.w, 
                           tes.l,
                           color='xkcd:purple', zorder=4)

    # W fin connector
    x_ = tes.l_c + tes.w/2
    qet_con_L = patches.Rectangle((-x_,-tes.w_fin_con/2), tes.l_c, 
                           tes.w_fin_con,
                           color='xkcd:purple', zorder=4)

    # Wider part of W fin connector
    x_ = tes.l_c + tes.w/2 + tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c  
    y_ = tes.w_overlap_stem/2
    l_ = tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c
    qet_con_L_ = patches.Rectangle((-x_,-y_),  l_  , 
                           tes.w_overlap_stem,
                           color='xkcd:purple', zorder=4)
    # W/Al overlap half ellipse
    arc_L = arc_patch((-x_,0), width=tes.l_overlap, 
                      height=tes.w_overlap/2, theta1=90, 
                      theta2=270, zorder=4)

    # W fin connector
    qet_con_R = patches.Rectangle((tes.w/2,-tes.w_fin_con/2), tes.l_c, 
                           tes.w_fin_con,
                           color='xkcd:purple', zorder=4)
    # Wider part of W fin connector
    x_ = tes.w/2+ tes.l_c  
    y_ = tes.w_overlap_stem/2
    l_ = tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c
    qet_con_R_ = patches.Rectangle((x_, -y_), l_, 
                           tes.w_overlap_stem,
                           color='xkcd:purple', zorder=4)
    # W/Al overlap half ellipse
    arc_R = arc_patch((x_+l_,0), width=-tes.l_overlap, 
                      height=tes.w_overlap/2, theta1=90, 
                      theta2=270, zorder=4)


    # W fin connector
    x_ = tes.w_overlap_stem/2
    y_ = tes.l/2
    w_ = tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c
    qet_con_T_ = patches.Rectangle((-x_, y_), tes.w_overlap_stem, 
                           tes.l_overlap_pre_ellipse,
                           color='xkcd:purple', zorder=4)
    # W/Al overlap half ellipse
    arc_T = arc_patch((0,y_+tes.l_overlap_pre_ellipse), width=tes.w_overlap/2, 
                      height=tes.l_overlap, theta1=0, 
                      theta2=180, zorder=4)
    
    # W fin connector
    x_ = tes.w_overlap_stem/2
    y_ = tes.l/2 + tes.l_overlap_pre_ellipse
    w_ = tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c
    qet_con_B_ = patches.Rectangle((-x_, -y_), tes.w_overlap_stem, 
                           tes.l_overlap_pre_ellipse,
                           color='xkcd:purple', zorder=4)
    # W/Al overlap half ellipse
    arc_B = arc_patch((0,-y_), width=tes.w_overlap/2,
                      height=-tes.l_overlap, theta1=0, 
                      theta2=180, zorder=4)

    
    
    ax.add_patch(e1)
    ax.add_patch(e2)
    ax.add_patch(e3)
    ax.add_patch(qet_con_L)
    ax.add_patch(qet_con_L_)
    ax.add_patch(arc_L)
    ax.add_patch(qet_con_R)
    ax.add_patch(qet_con_R_)
    ax.add_patch(arc_R)

    ax.add_patch(qet_con_T_)
    ax.add_patch(arc_T)

    ax.add_patch(qet_con_B_)
    ax.add_patch(arc_B)
    
    x = np.linspace(-a, a, 50)
    angles = calc_angles(det)
    plt.plot(x, _line(angles[0], x), color='w', linewidth=4)
    plt.plot(-x, _line(angles[-1], x), color='w', linewidth=4)
    ax.tick_params(which="both", direction="in", right=True, top=True, zorder=300)
    ax.set_title('Sample QET')

   
    
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
        
    locs, labels = plt.xticks() 
 
    ax.set_xticklabels(np.round(locs*1e6, decimals=0));

    locs, labels = plt.yticks() 

    ax.set_yticklabels(np.round(locs*1e6, decimals=0));

    ax.set_aspect('equal')
    
    ax.set_xlabel(r'$\mu$m')
    ax.set_ylabel(r'$\mu$m')
    
    return fig, ax

