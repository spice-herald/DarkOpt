import numpy as np
from scipy.special import iv as besseli
from scipy.special import kv as besselk

class QET:

    def __init__(self, n_fin, l_fin, h_fin, l_overlap, TES, eQPabsb=0.0, ePQP=0.52,
                 wempty=6e-6, wempty_tes=7.5e-6, ahole=1e-10):

        self._n_fin = n_fin
        self._l_fin = l_fin
        self._h_fin = h_fin
        self.l_overlap = l_overlap
        self._tes = TES
        self._eQPabsb = eQPabsb
        self._ePQP = ePQP
        # ---- QET Active Area ----
        self._wempty = wempty
        self._wempty_tes = wempty_tes
        self._nhole = 3 * self._n_fin
        self._ahole = ahole
        # SZ: shape of fins... recalculate area 
        self._afin_empty = n_fin * l_fin * wempty + 2 * TES.get_L() * wempty_tes + self._nhole * ahole 
        self._a_fin = np.pi * (self._l_fin ** 2) + 2 * self._l_fin * TES.get_L() - self._afin_empty
        """
        print("---------------- QET PARAMETERS ----------------")
        print("ePQP %s" % self._ePQP)
        print("lfin %s" % self._l_fin)
        print("hfin %s" % self._h_fin)
        print("loverlap %s" % self.l_overlap)
        print("ld %s" % (567 * h_fin))
        eff_absb = 1.22e-4
        print("la %s " % (1 /eff_absb*h_fin**2/l_overlap))
        print("Afin_empty %s" % self._afin_empty)
        print("Afin %s" % self._a_fin)
        print("nfin %s" % self._n_fin)
        print("------------------------------------------------\n")
        """

    def set_qpabsb_eff(self, l_fin, h_fin, loverlap, l_TES, eff_absb=1.22e-4):
        # From Effqp_2D_moffatt.m in Matt's dropbox 
        # Here we are using Robert Moffatt's full QP model. There are some pretty big assumptions:
        # 1) Diffusion length scales with Al thickness (trapping surface dominated and diffusion thickness limited)
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
        
	# We assume pie shaped QP collection fins
        ci = 2 * l_TES # inner circle circumferance 
        ri = ci / (2 * np.pi) # inner radius

        # Outer circumferance of "very simplified" ellipse  
        co1 = 2 * l_TES + 2 * np.pi * l_fin

        # Another approximation...
        a = (l_fin + l_TES) / 2
        b = l_fin

        # https://www.mathsisfun.com/geometry/ellipse-perimeter.html
        h = ((a - b) / (a + b)) ** 2
        co = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        ro = co1 / (2 * np.pi)

        # -------- Relevant length scales ------------
        # Diffusion length
        ld = 567 * h_fin  # [µm] this is the fit in Jeff's published LTD 16 data

        # Surface impedance length
        la = 1 /eff_absb*h_fin**2/loverlap  # [µm] these match the values used by Noah 
        la_chk = (1e6 + 1600 / (900 ** 2) * 5) * (h_fin ** 2)  # µm

        # -------- Dimensionless Scales -------
        rhoi = ri / ld
        rhoo = ro / ld
        lambdaA = la / ld

        # QP collection coefficient

        fQP = 2 * rhoi / (rhoo ** 2 - rhoi ** 2) \
        *(besseli(1, rhoo) * besselk(1, rhoi) - besseli(1, rhoi) * besselk(1, rhoo)) \
        / (besseli(1, rhoo) * (besselk(0, rhoi) + lambdaA * besselk(1, rhoi)) +
        (besseli(0, rhoi) - lambdaA * besseli(1, rhoi)) * besselk(1, rhoo))
        self._eQPabsb = fQP

    def get_epqp(self):
        return self._ePQP

    def get_eqpabsb(self):
        return self._eQPabsb

    def get_n_fin(self):
        return self._n_fin

    def get_l_fin(self):
        return self._l_fin

    def get_h_fin(self):
        return self._h_fin

    def get_l_overlap(self):
        return self.l_overlap

    def get_TES(self):
        return self._tes

    def get_wempty(self):
        return self._wempty

    def get_wempty_tes(self):
        return self._wempty_tes

    def get_nhole(self):
        return self._nhole

    def get_ahole(self):
        return self._ahole

    def get_afin_empty(self):
        return self._afin_empty

    def get_a_fin(self):
        return self._a_fin
