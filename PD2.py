import numpy as np 
from detector import Detector
from absorber import Absorber

class PD2(Detector):
    """
    A class to create a PD2 Detector object that includes the actual geometrical values 
    obtained from the l-edit mask  
    """
    def __init__(self, name, fridge, electronics, absorber, qet, tes, n_channel):
        super().__init__(name, fridge, electronics, absorber, qet, tes, n_channel)
 
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
        self._SA_active = self._n_channel * self._tes._nTES * self._qet._a_fin
        
        # Average area per cell, annd corresponding length 
        a_cell = self._absorber.get_pattern_SA()/(self._n_channel*self._tes._nTES)
        self._l_cell = np.sqrt(a_cell)
        print("a_cell ", a_cell)
        print("l_cell ", self._l_cell)
        y_cell = 2*self._qet._l_fin + self._tes._l 
        print("y_cell ", y_cell)
        
        # changed to match matlab + ledit  -- SZ 
        w_rail_main = 8e-6
        w_rail_qet = 4e-6
        
        # Design is not close packed. Get passive Al/QET
        a_passive_qet = self._l_cell*w_rail_main + (self._l_cell-y_cell)*w_rail_qet
        tes_passive = a_passive_qet *self._n_channel * self._tes._nTES
        outer_ring = 2* np.pi * (self._absorber.get_R() - self._absorber.get_w_safety()) * self._w_rail_main
        inner_ring = outer_ring / (np.sqrt(2))
        inner_vertical_rail = 3 * (self._absorber.get_R() - self._absorber.get_w_safety()) * self._w_rail_main * (1 - np.sqrt(2)/2.0)
        outer_vertical_rail = (self._absorber.get_R() - self._absorber.get_w_safety()) * self._w_rail_main * (1 + np.sqrt(2)/2.0)
   
        self._SA_passive = tes_passive + outer_ring + inner_ring + inner_vertical_rail + outer_vertical_rail 
 
        # Fraction of surface area which has phonon absorbing aluminum
        self._fSA_qpabsorb = (self._SA_passive + self._SA_active) / self._absorber.get_SA()
        # Fraction of Al which is QET fin which can produce signal
        self._ePcollect = self._SA_active / (self._SA_active + self._SA_passive)

        #self._t_pabsb = absb_time * (absb_lscat / lscat) * (fSA_qpabsb / self._fSA_qpabsorb)
        #self._t_pabsb = 20e-6 
        
        # Use PD2 values to estimate tau_pabsb: this estimation doensn't seem to be working 
        # Doesn't match the matlab code  
        PD2_absb_time = 20e-6
        absb_lscat = self._absorber.scattering_length()
        #PD2_fSA_qpabsb = 0.0071453736535236241
        PD2_fSA_qpabsb = 0.0214 # changed to match actual PD2 aluminum surface area - SZ 
        PD2_lscat = 0.001948849104859335

        print("absorber scattering length ", absb_lscat)
        print("PD2_fSA_qpabsb ", PD2_fSA_qpabsb)
        self._t_pabsb = PD2_absb_time * (absb_lscat / PD2_lscat) * (PD2_fSA_qpabsb / self._fSA_qpabsorb)

        self._w_collect = 1/self._t_pabsb

        #self._w_collect = 1/self._t_pabsb 

        # Total collection efficiency:
        self._eEabsb = self._e156 * self._ePcollect * self._qet._eQPabsb * self._qet._ePQP # * self._e_downconvert * self._fSA_qpabsorb 

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