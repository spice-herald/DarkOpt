from tes import TES
from QET import QET
from MaterialProperties import TESMaterial, DetectorMaterial
from electronics import Electronics
import numpy as np
import sys

# Some hard-coded numbers:
k_b = 1.38e-3 # J/K
t_tes = 40e-9 # m [set by fab constraints, and noise concerns, for now] 
n = 5 # used to define G (related to phonon/electron DOF)

class Detector:

    def __init__(self, name, fridge, electronics, absorber, qet, tes, n_channel):
        
        """
        
        PD2 Detector Object
        
        :param name: Detector Name
        :param fridge: Fridge in which detector is in
        :param absorber: Absorbing part of detector
        :param n_channel: Number of channels
        :param N_TES: Number of TES on detector
        :param l_TES: Length of TES
        :param l_fin: Length of QET Fin
        :param h_fin: Height of QET Fin
        :param l_overlap: Length of Overlap of W and Al???

        """
        # Figure out where these come from!!! 
        w_rail_main = 8e-6
        w_rail_qet = 4e-6 

        self._name = name
        self._fridge = fridge
        self._absorber = absorber
        self._n_channel = n_channel
        self._l_TES = tes._l
        self._l_fin = qet._l_fin
        self._h_fin = qet._h_fin
        self._l_overlap = qet.l_overlap
        self._w_rail_main = w_rail_main
        self._w_rail_qet = w_rail_qet
        self._electronics = electronics
        self._sigma_energy = 0
        self._qet = qet 
        self._tes = tes 

        # Set the QP Absorbtion Efficiency 
        self._qet.set_qpabsb_eff(self._l_fin, self._h_fin, self._l_overlap, self._l_TES) 

        # Total volume of Tungsten
        self._total_TES_vol = self._tes._volume

        # ------------- QET Fins -----------------
        # Percentage of surface area covered by QET Fins 
        # SZ: should be multiplied by n_fin??? 
        self._SA_active = self._n_channel * self._tes._nTES * self._qet._a_fin

        # Average area per cell, and corresponding length
        a_cell = self._absorber.get_pattern_SA() / (n_channel * self._tes._nTES) # 1/2 channels on each side
        self._l_cell = np.sqrt(a_cell)
        
        print("l_cell: ", self._l_cell)
        y_cell = 2 * self._qet._l_fin + self._tes._l
        print("y_cell: ", y_cell)
        if self._l_cell > y_cell:
            # Design is not close packed. Get passive Al/QET
            a_passive_qet = self._l_cell * w_rail_main + (self._l_cell - y_cell) * w_rail_qet

        else:
            # Design is close packed. No vertical rail to QET
            x_cell = a_cell / y_cell
            a_passive_qet = x_cell * self._w_rail_main
        print("x_cell: ", self._l_cell)
        print("w_rail_main ", w_rail_main)
        print("w_rail_qet ", w_rail_qet)
        print("a_passive_qet: ", a_passive_qet)
        tes_passive = a_passive_qet * n_channel * self._tes._nTES
        outer_ring = 2 * np.pi * (self._absorber.get_R() - self._absorber.get_w_safety()) * self._w_rail_main
        inner_ring = outer_ring / (np.sqrt(2))
        inner_vertical_rail = 3 * (self._absorber.get_R() - self._absorber.get_w_safety()) * self._w_rail_main * (1 - np.sqrt(2)/2.0)
        outer_vertical_rail = (self._absorber.get_R() - self._absorber.get_w_safety()) * self._w_rail_main * (1 + np.sqrt(2)/2.0)

        print("tes_passive ", tes_passive)
        # Total Passive Surface Area
        # 1. TES Passive Area
        # 2. Outer Ring
        # 3. Inner Ring
        # 4. Inner Vertical Rail
        # 5. Outer Vertical Rail
        self._SA_passive = tes_passive + outer_ring + inner_ring + inner_vertical_rail + outer_vertical_rail
        print("SA_passive = ",self._SA_passive)
        # Fraction of surface area which has phonon absorbing aluminum
        self._fSA_qpabsorb = (self._SA_passive + self._SA_active) / self._absorber.get_SA()
        print("fSA qpabsorb: ", self._fSA_qpabsorb)
        # Fraction of Al which is QET fin which can produce signal
        self._ePcollect = self._SA_active / (self._SA_active + self._SA_passive)
 
        PD2_absb_time = 20e-6
        absb_lscat = absorber.scattering_length()
        PD2_fSA_qpabsb = 0.0214 # percentage of total surface area 
        PD2_lscat = 0.0019488

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
        self._e_downconvert = 1/4000

        # Let's combine 1), 5), and 6) together and assume that it is the same as the measured/derived value from iZIP4
        self._e156 = 0.8690

        # Total collection efficiency:
        self._eEabsb = self._e156 * self._ePcollect * self._qet._eQPabsb * self._qet._ePQP # * self._e_downconvert * self._fSA_qpabsorb 

        # ------------ Thermal Conductance to Bath ---------------
        self._kpb = 1.55e-4
        # Thermal conductance coefficient between detector and bath
        self._nkpb = 4

        # ----------- Electronics ----------
        self._total_L = self._electronics._l_squid + self._electronics._l_p + self._tes._L 

        # ---------- Response Variables to Be Set in Simulation of Noise ---------------
        self._response_omega = 0
        self._response_dPtdE = 0
        self._response_dIdP = 0
        self._response_z_tes = 0
        self._response_z_tot = 0
        self._response_dIdV = 0
        self._response_dIdV0 = 0
        self._response_dIdV_step = 0
        self._response_t = 0

        
        print("---------------- DETECTOR PARAMETERS ----------------")
        print("nP %s" % self._n_channel)
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
        print("N_TES %s" % self._tes._nTES)
        print("l_squid %s" % self._electronics._l_squid)
        print("l_p %s" % self._electronics._l_p)
        print("tes_l %s" % self._tes._L)
        print("total_L %s" % self._total_L)
        print("------------------------------------------------\n")
        


    def set_response_omega(self, omega):
        self._response_omega = omega

    def set_dPtdE(self, val):
        self._response_dPtdE = val

    def set_dIdP(self, val):
        self._response_dIdP = val

    def get_dIdP(self):
        return self._response_dIdP

    def set_ztes(self, val):
        self._response_z_tes = val

    def set_ztot(self, val):
        self._response_z_tot = val

    def set_dIdV(self, val):
        self._response_dIdV = val

    def get_dIdV(self):
        return self._response_dIdV

    def set_dIdV0(self, val):
        self._response_dIdV0 = val

    def set_dIdV_step(self, val):
        self._response_dIdV_step = val

    def set_t(self, val):
        self._response_t = val

