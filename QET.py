import numpy as np

class QET:

    def __init__(self, n_fin, l_fin, h_fin, l_overlap, TES,
                 wempty=6e-6, wempty_tes=7.5e-6, ahole=1e-10):

        self._n_fin = n_fin
        self._l_fin = l_fin
        self._h_fin = h_fin
        self.l_overlap = l_overlap
        self._tes = TES

        # ---- QET Active Area ----
        self._wempty = wempty
        self._wempty_tes = wempty_tes
        self._nhole = 3 * self._n_fin
        self._ahole = ahole
        self._afin_empty = n_fin * l_fin * wempty + 2 * TES.get_L() * wempty_tes + self._nhole * ahole
        self._a_fin = np.pi * (self._l_fin ** 2) + 2 * self._l_fin * TES.get_L() - self._afin_empty

        # TODO ASK ABOUT LINES 193-197 QUASIPARTICLE DIFFUSIVE STUFF!

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

    def get_a_fin(self):
        return self._a_fin
