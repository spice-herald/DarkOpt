from fridge import Fridge
from detector import Detector
from electronics import Electronics
from absorber import Absorber
from simulated_noise import simulate_noise

fSnolab = Fridge("SNOLAB", 20e-3, 145e-3, 900e-3, 4.8, 0)
absorber = Absorber("Si", 1e-3, 38.1e-3, 3e-3, 2.329e3)
eSnolab = Electronics(fSnolab, fSnolab.get_TCP(), fSnolab.get_TMC())
eSLAC = Electronics(fSnolab, fSnolab.get_TMC(), fSnolab.get_TMC(), 5e-3, 6e-3, 25e-9, 25e-9, 4e-12)

# Name, fridge, absorber, n_channel
det = Detector("PD2", fSnolab, eSLAC, absorber, 1)
simulate_noise(det)

