from fridge import Fridge
from detector import Detector
from electronics import Electronics
from absorber import Absorber
from tes import TES 
from QET import QET 
from simulated_noise import simulate_noise
from MaterialProperties import TESMaterial
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# testing out some plotting

#                 name     T_MC   T_CP    T_Still  T_4K  parasitic  %%% CHECK ALL OF THESE  
fSnolab = Fridge("SNOLAB", 20e-3, 145e-3, 900e-3,  4.8,  0)
#                   name  shape   h     r     w_safe 
absorber = Absorber("Si", "cube", 1e-2, 1e-2, 3e-3)
eSnolab = Electronics(fSnolab, fSnolab.get_TCP(), fSnolab.get_TMC())
eSLAC = Electronics(fSnolab, fSnolab.get_TMC(), fSnolab.get_TMC(), 5e-3, 6e-3, 25e-9, 25e-9, 4e-12)

tungsten = TESMaterial()

# Define Nominal Input Values 
tes_l = 100e-6
tes_w = 2e-6
foverlap = 1 # If == 1 --> total perimeter around TES has W/Al connector coverage 
l_fin = 200e-6

if l_fin > 100e-6:
    n_fin = 6
else:
    n_fin = 4 

sigma = 0 # what is this?
T_eq = -100 # Equilibrium Temperature what is this?
h_fin = 600e-9 
l_overlap = 10e-6

res_n = 300e-3 # Normal Resistance of TES channel  


def plot_efftes_l():
    tes_l = np.arange(0, 500e-6, 5e-6)
    eff = []

    fig, ax = plt.subplots()
    for l in np.nditer(tes_l):
       tes = TES(l, tes_w, foverlap, n_fin, sigma, T_eq, res_n, tungsten)
       qet = QET(n_fin, l_fin, h_fin, l_overlap, tes)
       det = Detector("cubes", fSnolab, eSLAC, absorber, qet, tes, 1)
       e = det._eEabsb
       eff.append(e)
    plt.plot(tes_l, eff) 
    fig.savefig("test.png")
    plt.show()

def plot_signoise_tesl():
    # something very strange is going on here 
    tes_l = np.arange(0, 300e-6, 20e-6)
    sig_noise = []

    fig, ax = plt.subplots()
    for l in np.nditer(tes_l):
       tes = TES(l, tes_w, foverlap, n_fin, sigma, T_eq, res_n, tungsten)
       qet = QET(n_fin, l_fin, h_fin, l_overlap, tes)
       det = Detector("cubes", fSnolab, eSLAC, absorber, qet, tes, 1)
       e = det._eEabsb
       noise = simulate_noise(det)
       sig_noise.append(e/noise)
    plt.plot(tes_l, sig_noise) 
    fig.savefig("test.png")
    plt.show()
