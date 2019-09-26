from fridge import Fridge
from tes import TES
from QET import QET
from detector import Detector
from PD2 import PD2 
from electronics import Electronics
from absorber import Absorber
from simulated_noise import simulate_noise
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from MaterialProperties import TESMaterial

fSnolab = Fridge("SNOLAB", 20e-3, 145e-3, 900e-3, 4.8, 0)
# Absorber: Silicon. Height 1mm. Radius 38.1mm. W safety 3mm. 
#absorber = Absorber("Si", "cylinder", 1e-3, 38.1e-3, 3e-3) # same as matlab
absorber = Absorber("Si", "cylinder", 1e-3, 38.1e-3, 3e-3) # same as matlab
eSnolab = Electronics(fSnolab, fSnolab.get_TCP(), fSnolab.get_TMC())
eSLAC = Electronics(fSnolab, fSnolab.get_TMC(), fSnolab.get_TMC(), 5e-3, 6e-3, 25e-9, 25e-9, 4e-12)

tungsten = TESMaterial()

#PD2 input valies 
tes_l = 140e-6 # same as matlab
#tes_l = 300e-6 # same as matlab
#tes_w = 3.5e-6 # matlab = 4e-6 
tes_w = 1.5e-6 # matlab = 4e-6 
foverlap = 1.2 # same as matlab (why greater than 1??)
#foverlap = 0.8 # same as matlab (why greater than 1??)
l_overlap = 10e-6 # same as matlab 
n_fin = 6
l_fin = 200e-6 # same as matlab
h_fin = 600e-9 # same as matlab
sigma = tungsten._gPep_v 
T_eq = -100  
res_n = 300e-3 

# define the TES and QET with PD2 input values  
tes = TES(tes_l, tes_w, foverlap, n_fin, sigma, T_eq, res_n, tungsten )
qet = QET( l_fin, h_fin, l_overlap, tes)


det = PD2("PD2", fSnolab, eSLAC, absorber, qet, tes, 1)

det.set_leditvals()

#eres = simulate_noise(det)
