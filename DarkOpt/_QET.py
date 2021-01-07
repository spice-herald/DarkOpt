import numpy as np
from scipy.special import iv as besseli
from scipy.special import kv as besselk

class QET:
    """
    Class for storing QET related quantities. Calculates the QP Collection Effciency 
    (efficiency of quasiparticles reaching the W trap) based on TES parameters and 
    Al fin dimentions.
    """

    def __init__(self, l_fin, h_fin, n_fin, l_tes, l_overlap, ahole, ePQP=0.52,
                 wempty=6e-6, wempty_tes=7.5e-6):
        
        """
        l_fin : float, 
            Length of Al fins [m]
        h_fin : float, 
            hight of Al fins [m]
        TES : TES Object, 
            TES Object
        n_fin : int, 
            Number of fins in QET
        l_tes : float, 
            lenght of TES [m]
        l_overlap : float
            lenght of Al/W overlap region in [m]
        ahole : float
            ?
        ePQP : float, optional
            Phonon to QP Conversion Effciency, 
            Kaplan downconversion limits this to 52%
        wempty : float,
            ?
        wempty_tes : float, 
            ?
        """
        

        self.l_fin = l_fin
        self.h_fin = h_fin
        self.n_fin = n_fin
        self.TES = TES
        l_tes = TES.l_tes
        l_overlap = TES.l_overlap
        self.eQPabsb = None #gets set by method
        self.ePQP = ePQP # efficiency of phonon in subrate breaking cooper pair
                          # in Al
        # ---- QET Active Area ----
        self.wempty = wempty
        self.wempty_tes = wempty_tes
        self.nhole = 3 * n_fin
        self.ahole = ahole 
        self.afin_empty = n_fin * l_fin * wempty + 2 * l_tes * wempty_tes + self.nhole * ahole 
        #self._a_fin = np.pi * (self._l_fin ** 2) + 2 * self._l_fin * TES._l - self._afin_empty # SZ: bug?? 
        self.a_fin = np.pi*l_fin*(l_fin + (l_tes/2)) - self.afin_empty

#         if TES._print == True:
#             print("---------------- QET PARAMETERS ----------------")
#             print("ePQP %s" % self._ePQP)
#             print("lfin %s" % self._l_fin)
#             print("hfin %s" % self._h_fin)
#             print("loverlap %s" % self.l_overlap)
#             print("ld %s" % (567 * h_fin))
#             eff_absb = 1.22e-4
#             print("la %s " % (1 /eff_absb*h_fin**2/self.l_overlap))
#             print("Afin_empty %s" % self._afin_empty)
#             print("Afin %s" % self._a_fin)
#             print("------------------------------------------------\n")
        

    def set_qpabsb_eff_matt(self, eff_absb=1.22e-4, method=0):
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
        
        l_tes = self.l_tes*1e6 #convert to [um]
        l_fin = self.l_fin*1e6 #convert to [um]
        h_fin = self.h_fin*1e6 #convert to [um]
        l_overlap = self.l_overlap*1e6 #convert to [um]
        n_fin = self.n_fin
        
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

        # Surface impedance length
        la = (1 /eff_absb)*(h_fin**2/l_overlap)  # [µm] these match the values used by Noah 
        la_chk = (1e6 + 1600 / (900 ** 2) * 5) * (h_fin ** 2)  # µm

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
    
    def set_qpabsb_eff(self, l_fin, h_fin, aoverlap, ci, l_TES, eff_absb=1.22e-4):
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
        
        l_tes = self.l_tes*1e6 #convert to [um]
        l_fin = self.l_fin*1e6 #convert to [um]
        h_fin = self.h_fin*1e6 #convert to [um]
        l_overlap = self.l_overlap*1e6 #convert to [um]
        n_fin = self.n_fin
        # We assume pie shaped QP collection fins
        ri = ci / (2 * np.pi) # inner radius
        #print("ri    ", ri)
        # Outer circumferance of "very simplified" ellipse  
        #co1 = 2 * l_TES + 2 * np.pi * l_fin

        # Another approximation...
        a = l_fin + l_TES/2
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

        # Surface impedance length
        la = (1 /eff_absb)*(h_fin**2/loverlap)  # [µm] these match the values used by Noah 
        la_chk = (1e6 + 1600 / (900 ** 2) * 5) * (h_fin ** 2)  # µm
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
