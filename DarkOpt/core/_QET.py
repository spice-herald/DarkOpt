import numpy as np
from scipy.special import iv as besseli
from scipy.special import kv as besselk

class QET:
    """
    Class for storing QET related quantities. Calculates the QP Collection Effciency 
    (efficiency of quasiparticles reaching the W trap) based on TES parameters and 
    Al fin dimentions.
    """

    def __init__(self, l_fin, h_fin, TES, ahole, ePQP=0.52, eff_absb = 1.22e-4,
                  nhole_per_fin=3, type_qp_eff=0):
        
        """
        l_fin : float, 
            Length of Al fins [m]
        h_fin : float, 
            hight of Al fins [m]
        TES : TES Object, 
            TES Object
        ahole : float
            area of holes in fin [m^2]
        ePQP : float, optional
            Phonon to QP Conversion Effciency, 
            Kaplan downconversion limits this to 52%
        eff_absb : float, optional
            W/Al transmition/trapping probability
        nhole_per_fin : int, optional,
            Number of holes per Al fin
        type_qp_eff : int, optional
            how the efficiency should be calculated.
            0 : 'modern' estimate of overlap radius (small)
            1 : 'modern' estimate, but different effective l_overlap
            2 : Matt's method
        """
        

        self.l_fin = l_fin
        self.h_fin = h_fin
        self.TES = TES
        n_fin = TES.n_fin
        l_tes = TES.l
        w_tes = TES.w
        l_overlap = TES.l_overlap
        self.nhole_per_fin = nhole_per_fin
        self.eQPabsb = None #gets set by method
        self.ePQP = ePQP # efficiency of phonon in subrate breaking cooper pair
                          # in Al
        self.eff_absb = eff_absb
        # ---- QET Active Area ----
        self.nhole = nhole_per_fin * n_fin 
        self.ahole = ahole 
        
        
        ##################
        # Area of Al fin is calculated by calulating the 
        # area of the overal ellipse, then subracting the non
        # active Al; ie the area of the holes, the channel 
        # cutouts, and the area of the TES + empty space around
        # the TES
        
        # self.afin_empty = n_fin * l_fin * wempty_fin + 2 * l_tes * wempty_tes + self.nhole * ahole #Summers calc, missing
        # area of TES itself
        
        a_fin_chan = n_fin * l_fin * self.TES.wempty_fin # area of cutout channels seperating fins
        a_tes_space = l_tes*(w_tes + 2*self.TES.wempty_tes) # area of tes + empty space on either side
        aholes = self.nhole * ahole # area of all holes in Al fins
        self.afin_empty = a_fin_chan + a_tes_space + aholes
        self.a_fin = np.pi*l_fin*(l_fin + (l_tes/2)) - self.afin_empty
        self.type_qp_eff = type_qp_eff
        
        
        if type_qp_eff == 0: # Updated estimate with small ci, changing effective l_overlap
            if TES.con_type == 'modern':            
                self.ci = n_fin*2*l_overlap
            elif TES.con_type == 'ellipse':
                #self.ci = 2*l_tes + (7.5e-6)*4 - n_fin*(6e-6)
                self.ci = 2*l_tes + (self.TES.wempty_tes)*4 - n_fin*(self.TES.wempty_fin) # this is confusing? 
            self.set_qpabsb_eff() 
        if type_qp_eff == 1: # Updated estimate with same ci, changing effective l_overlap 
            self.ci = 2*l_tes
            self._qet.set_qpabsb_eff() 
        if type_qp_eff == 2: # Original estimate that assumes entire perimeter is W/Al overlap with ci = 2*l_TES
            self.set_qpabsb_eff_matt()



    def set_qpabsb_eff_matt(self):
        """
        Calculate the QP collection efficiciency using Matt's method. Sets class atribute
        slef.eQPabsb
        
        Parameters:
        -----------
        eff_absb : float, optional
            W/Al transmition/trapping probability
                
        """
        # From Effqp_2D_moffatt.m in Matt's dropbox 
        # Here we are using Robert Moffatt's full QP model. There are some pretty big assumptions:
        # 1) Diffusion length scales with Al thickness (trapping surface dominated and diffusion thickness limited)
        # 2) Absorption length scales with Al thickness**2/l_overlap 
        # Future: scale boundary impedance with W thickness 
        # INPUTS: 
        #    1) fin length [um]
        #    2) fin height [um]
        #    3) W/Al overlap [um]
        #    4) TES length [um] 
        #    5) W/Al transmition/trapping probability
        # OUTPUTS: 
        #    1) Quasi-Particle Collection Efficiency 
        #    2) Diffusion Length 
        #    3) W/Al Surface Absorption Length
        #    4) W/Al Transmission Probability
        # https://www.stanford.edu/~rmoffatt/Papers/Diffusion%20and%20Absorption%20with%201D%20and%202D%20Solutions.pdf
        # DOI: 10.1007/s10909-015-1406-7
        # -------------------------------------------------------------------------------------------------------------
        
        l_tes = self.TES.l*1e6 #convert to [um]
        l_fin = self.l_fin*1e6 #convert to [um]
        h_fin = self.h_fin*1e6 #convert to [um]
        l_overlap = self.TES.l_overlap*1e6 #convert to [um]
        n_fin = self.TES.n_fin
        eff_absb = self.eff_absb
        # We assume pie shaped QP collection fins
        ci = 2 * l_tes # inner circle circumferance 
        ri = ci / (2 * np.pi) # inner radius

        # Outer circumferance of "very simplified" ellipse  
        co1 = 2 * l_tes + 2 * np.pi * l_fin

        # Another approximation...
        a = (l_fin + l_tes) / 2
        b = l_fin

        # https://www.mathsisfun.com/geometry/ellipse-perimeter.html
        h = ((a - b) / (a + b)) ** 2
        co = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        ro = co1 / (2 * np.pi)

        # -------- Relevant length scales ------------
        # Diffusion length
        ld = 567 * h_fin  # [µm] this is the fit in Jeff's published LTD 16 data
        self.ld = ld
        # Surface impedance length
        la = (1 /eff_absb)*(h_fin**2/l_overlap)  # [µm] these match the values used by Noah 
        la_chk = (1e6 + 1600 / (900 ** 2) * 5) * (h_fin ** 2)  # µm
        self.la = la
        
        # -------- Dimensionless Scales -------
        rhoi = ri / ld
        rhoo = ro / ld
        lambdaA = la / ld

        # QP collection coefficient

        fQP = (2 * rhoi / (rhoo ** 2 - rhoi ** 2)) \
        *(besseli(1, rhoo) * besselk(1, rhoi) - besseli(1, rhoi) * besselk(1, rhoo)) \
        / (besseli(1, rhoo) * (besselk(0, rhoi) + lambdaA * besselk(1, rhoi)) +
        (besseli(0, rhoi) - lambdaA * besseli(1, rhoi)) * besselk(1, rhoo))
        self.eQPabsb = fQP
    
    def set_qpabsb_eff(self):
        # From Effqp_2D_moffatt.m in Matt's dropbox 
        # Here we are using Robert Moffatt's full QP model. There are some pretty big assumptions:
        # 1) Diffusion length scales with Al thickness (trapping surface dominated and diffusion thickness limited)
        # 2) Absorption length scales with Al thickness**2/l_overlap 
        # Future: scale boundary impedance with W thickness 
        # INPUTS: 
        #    1) fin length [um]
        #    2) fin height [um]
        #    3) W/Al overlap [um]
        #    4) TES length [um] 
        #    5) W/Al transmition/trapping probability
        # OUTPUTS: 
        #    1) Quasi-Particle Collection Efficiency 
        #    2) Diffusion Length 
        #    3) W/Al Surface Absorption Length
        #    4) W/Al Transmission Probability
        # https://www.stanford.edu/~rmoffatt/Papers/Diffusion%20and%20Absorption%20with%201D%20and%202D%20Solutions.pdf
        # DOI: 10.1007/s10909-015-1406-7
        # -------------------------------------------------------------------------------------------------------------
        
        l_tes = self.TES.l*1e6 #convert to [um]
        l_fin = self.l_fin*1e6 #convert to [um]
        h_fin = self.h_fin*1e6 #convert to [um]
        l_overlap = self.TES.l_overlap*1e6 #convert to [um]
        n_fin = self.TES.n_fin
        aoverlap = self.TES.A_overlap*1e12 #[um^2]
        eff_absb = self.eff_absb
        ci = self.ci*1e6 #convert to [um]
        # We assume pie shaped QP collection fins
        ri = ci / (2 * np.pi) # inner radius
        #print("ri    ", ri)
        # Outer circumferance of "very simplified" ellipse  
        #co1 = 2 * l_TES + 2 * np.pi * l_fin

        # Another approximation...
        a = l_fin + l_tes/2
        b = l_fin

        # https://www.mathsisfun.com/geometry/ellipse-perimeter.html
        h = ((a - b) / (a + b)) ** 2
        co = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        ro = co / (2 * np.pi)
        #print("ro               ", ro)
        #print("l fin   = ", l_fin)
        #print("ro - ri = ", ro-ri )

        #print("aoverlap        ", aoverlap)
        # re-define loverlap based on actual area aoverlap 
        
        loverlap = np.sqrt((aoverlap/np.pi) + (ri**2)) - ri
        #print("l_overlap prime  ", loverlap) 
        #print("Overlap ", loverlap)
        # -------- Relevant length scales ------------
        # Diffusion length
        ld = 567 * h_fin  # [µm] this is the fit in Jeff's published LTD 16 data
        self.ld = ld
        # Surface impedance length
        la = (1 /eff_absb)*(h_fin**2/loverlap)  # [µm] these match the values used by Noah 
        la_chk = (1e6 + 1600 / (900 ** 2) * 5) * (h_fin ** 2)  # µm
        self.la = la
        #print("la                 ", la) 
        # -------- Dimensionless Scales -------
        rhoi = ri / ld
        rhoo = ro / ld
        lambdaA = la / ld

        # QP collection coefficient

        fQP = (2 * rhoi / (rhoo ** 2 - rhoi ** 2)) \
        *(besseli(1, rhoo) * besselk(1, rhoi) - besseli(1, rhoi) * besselk(1, rhoo)) \
        / (besseli(1, rhoo) * (besselk(0, rhoi) + lambdaA * besselk(1, rhoi)) +
        (besseli(0, rhoi) - lambdaA * besseli(1, rhoi)) * besselk(1, rhoo))
        #print("fQP   ", fQP)
        self.eQPabsb = fQP
        
        
        
    def print(self):

        print("---------------- QET PARAMETERS ----------------")
        print("ePQP =  %s" % self.ePQP)
        print(f"eQPabsb = {self.eQPabsb}")
        print("lfin =  %s" % self.l_fin)
        print("hfin =  %s" % self.h_fin)
        print("loverlap =  %s" % self.TES.l_overlap)
        print("ld =  %s" % self.ld)
        print("la =  %s " % self.la)
        print("Afin_empty =  %s" % self.afin_empty)
        print("Afin =  %s" % self.a_fin)
        print("------------------------------------------------\n")
        
