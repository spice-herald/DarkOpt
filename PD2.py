import numpy as np
import math as m 
from detector import Detector
from absorber import Absorber

mu = 1.257e-6 #H/m  
class PD2(Detector):
    """
    A class to create a PD2 Detector object that includes the actual geometrical values 
    obtained from the l-edit mask  
    """
    def __init__(self, name, fridge, electronics, absorber, qet, tes, n_channel):
        super().__init__(name, fridge, electronics, absorber, qet, tes, n_channel)

    def calc_maskL(self):
        # ---- Calculate mask Inductance 
        r = 24000e-6 #m [radius of inner circle]
        d = 2060e-6 #m [distance between wire pairs]
        w = 8e-6 #m [width of wire]
        dl = 1600e-6 #m [Delta l between tes]
        # Because current through rails is in same direction, to first order all flux through parallel tes rails 
        # cancels and only inductance comes from the (circular) bias lines. Here is a rough calculation of the 
        # inductance due to these lines. 
        # between central bias line and inner circle bias line:
        l = 2*r # vetical length 
        dl = 4100e-6 # vertical distance between lines off central bias line 
        n = 12 # number of lines off central bias line 
        d_i = 22000e-6 # distance between bias lines
        d_o = 10400e-6 # distance between outer bias lines 
        Li_i = 0  
        Li_o = 0 
        for j in range(n):
            dL_i = (mu*dl/m.pi)*((n-j)/(2*n))*m.log((2*(d_i-(w/2)))/w)  
            dL_o = (mu*dl/m.pi)*((n-j)/(2*n))*m.log((2*(d_o-(w/2)))/w)  
            Li_i = Li_i +dL_i
            Li_o = Li_o +dL_o
        l_bias = 1/(2/Li_i+2/Li_o)
        l_mask = l_bias
        print("----- L bias: ", l_bias)
        return l_mask

 
    def set_leditvals(self):
        # _volume_TES is the same

        # Volume of the W/Al overlap
        # two end connectors 
        endcon_wal_overlap = 128.535e-12+228.535e-12+258e-12 # area from ledit
        midcon_wal_overlap = 480e-12 # area from ledit
         
        self._tes._vol_WAl_overlap = (4*midcon_wal_overlap + 2*endcon_wal_overlap)*self._tes._t 
        
        # Volume of the W only fin connector 
        #self._tes._vol_WFinCon = 2.5e-6 * n_fin * 4e-6 * self._t + 2.5e-6 * (2 * self._l * self._foverlap_width) * self._t
        midcon_w_only = 88.9e-12 # area from ledit
        endcon_w_only = 68.39e-12  # area from ledit
        self._tes._vol_WFinCon = (4*midcon_w_only + 2*endcon_w_only)*self._tes._t 
        
        # Total Volume 
        self._tes._volume = self._tes._volume_TES + self._tes._veff_WFinCon * self._tes._vol_WFinCon + \
                       self._tes._veff_WAloverlap * self._tes._vol_WAl_overlap
        
        print("Total Volume 1 TES: ", self._tes._volume_TES, " + ", self._tes._veff_WFinCon, " * ", self._tes._vol_WFinCon, " + ", self._tes._veff_WAloverlap, " * ", self._tes._vol_WAl_overlap)
        # reset n_tes 
        self._tes._nTES = 975 + 28+28 

        self._tes._tot_volume = self._tes._volume * self._tes._nTES  
        self._tes._K = self._tes._tot_volume * self._tes._sigma
        # Phonon electron thermal coupling
        self._tes._G = self._tes._n * self._tes._K * (self._tes._T_eq ** (self._tes._n-1))

        # Recalculate total Normal Resistance based on Actual number of TES 
        self._tes._total_res_n = self._tes._res1tes/self._tes._nTES 
        # Operating Resistance
        self._tes._res_o = self._tes._total_res_n * self._tes._fOp  
        
        # Percentage of surface area covered by QET Fins
        self._qet._a_fin = 28660e-12+28668e-12+28683e-12+28660e-12+28668e-12+28683e-12+self._qet._nhole *self._qet._ahole # check this
        print("Afin: ", self._qet._a_fin)
        self._SA_active = self._n_channel * self._tes._nTES * self._qet._a_fin
        
        # Average area per cell, annd corresponding length 
        a_cell = self._absorber.get_pattern_SA()/(self._n_channel*self._tes._nTES)
        self._l_cell = np.sqrt(a_cell)
        print("a_cell ", a_cell)
        print("l_cell ", self._l_cell)
        y_cell = 2*self._qet._l_fin + self._tes._l 
        print("y_cell ", y_cell)
         
        # Design is not close packed. Get passive Al/QET
        a_passive_qet = self._l_cell*self._w_rail_main + (self._l_cell-y_cell)*self._w_rail_qet
        tes_passive = a_passive_qet *self._n_channel * self._tes._nTES
        outer_ring = 2* np.pi * (self._absorber.get_R() - self._absorber.get_w_safety()) * self._w_rail_main
        inner_ring = outer_ring / (np.sqrt(2))
        inner_vertical_rail = 3 * (self._absorber.get_R() - self._absorber.get_w_safety()) * self._w_rail_main * (1 - np.sqrt(2)/2.0)
        outer_vertical_rail = (self._absorber.get_R() - self._absorber.get_w_safety()) * self._w_rail_main * (1 + np.sqrt(2)/2.0)
   
        self._SA_passive = tes_passive + outer_ring + inner_ring + inner_vertical_rail + outer_vertical_rail 
 
        # Fraction of surface area which has phonon absorbing aluminum
        self._fSA_qpabsorb = (self._SA_passive + self._SA_active) / self._absorber.get_SA()
        print("fSA: ", self._fSA_qpabsorb)
        # Fraction of Al which is QET fin which can produce signal
        self._ePcollect = self._SA_active / (self._SA_active + self._SA_passive)

        self._t_pabsb = 20e-6 # this is the measured value 
        
        self._w_collect = 1/self._t_pabsb

        #self._w_collect = 1/self._t_pabsb 

        # Total collection efficiency:
        self._eEabsb = self._e156 * self._ePcollect * self._qet._eQPabsb * self._qet._ePQP # * self._e_downconvert * self._fSA_qpabsorb 

        self._total_L = self._electronics._l_squid + self._electronics._l_p + self.calc_maskL() 

        print("---------------- LEDIT DETECTOR PARAMETERS ----------------")
        print("nP %s" % self._n_channel)
        print("vol1tes ",  self._tes._volume)
        print("N_TES %s" % self._tes._nTES)
        print("tot tes vol %s" % self._tes._tot_volume)
        print("K %s" % self._tes._K)
        print("G %s" % self._tes._G)
        print("Rn %s" % self._tes._total_res_n)
        print("Ro %s" % self._tes._res_o)
        print("SAactive %s" % self._SA_active)
        print("lcell %s" % self._l_cell)
        print("SApassive %s" % self._SA_passive)
        print("fSA_QPabsb %s" % self._fSA_qpabsorb)
        print("ePcollect %s" % self._ePcollect)
        print("tau_pabsb %s" % self._t_pabsb)
        print("w_pabsb %s" % (1/self._t_pabsb))
        print("eE156 %s" % self._e156)
        print("eEabsb %s" % self._eEabsb)
        print("Kpb %s" % self._kpb)
        print("nKpb %s" % self._nkpb)
        print("total_L %s" % self._total_L)

        print("------------------------------------------------\n")
