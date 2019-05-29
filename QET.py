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
        self._afin_empty = n_fin * l_fin * wempty + 2 * TES.get_L() * wempty_tes + self._nhole * ahole
        self._a_fin = np.pi * (self._l_fin ** 2) + 2 * self._l_fin * TES.get_L() - self._afin_empty

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


    def set_qpabsb_eff(self, l_fin, h_fin, loverlap, l_TES, eff_absb=1.22e-4):
        ci = 2 * l_TES
        ri = ci / (2 * np.pi)

        a = (l_fin + l_TES) / 2
        b = l_fin

        co1 = 2 * l_TES + 2 * np.pi * l_fin
        h = ((a - b) / (a + b)) ** 2
        co = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        ro = co1 / (2 * np.pi)

        # -------- Relevant length scales
        # Diffusion length
        ld = 567 * h_fin  # µm

        # Surface impedance length
        la = 1 /eff_absb*h_fin**2/loverlap  # µm
        la_chk = (1e6 + 1600 / (900 ** 2) * 5) * (h_fin ** 2)  # µm

        # -------- Dimensionless Scales
        rhoi = ri / ld
        rhoo = ro / ld
        lambdaA = la / ld

        # QP collection coefficient
        """fQP = 2 * rhoi / (rhoo ** 2 - rhoi ** 2) * \
        (iv(1, rhoo) * kv(1, rhoi) - iv(1, rhoi) * kv(1, rhoo)) / \
        (iv(1, rhoo) * kv(0, rhoi) + lambdaA * kv(1, rhoi) + (iv(0, rhoi) - lambdaA * iv(1, rhoi)) * kv(1, rhoo))

        fQP = 2 * rhoi / (rhoo**2 - rhoi**2) * (iv(1, rhoo) * kv(1, rhoi) - iv(1, rhoi) * kv(1, rhoo)) / \
              (iv(1, rhoo) * (kv(0, rhoi) + lambdaA * kv(1, rhoi)) + (iv(0, rhoi) - lambdaA*iv(1, rhoi))*kv(1, rhoo))"""

        fQP = 2 * rhoi / (rhoo ** 2 - rhoi ** 2) \
        *(besseli(1, rhoo) * besselk(1, rhoi) - besseli(1, rhoi) * besselk(1, rhoo)) \
        / (besseli(1, rhoo) * (besselk(0, rhoi) + lambdaA * besselk(1, rhoi)) +
        (besseli(0, rhoi) - lambdaA * besseli(1, rhoi)) * besselk(1, rhoo))
        self._eQPabsb = fQP
        # TODO SOMETHING IS WRONG IN THIS QET EFFICIENCY!
        print("###################### QET QP absb efficiency = %s" % self._eQPabsb)

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
