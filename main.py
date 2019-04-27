from fridge import Fridge
from detector import Detector
from electronics import Electronics
from absorber import Absorber
from simulated_noise import simulate_noise

fSnolab = Fridge("SNOLAB", 20e-3, 145e-3, 900e-3, 4.8, 0)
absorber = Absorber("Si", 1e-3, 38.1e-3, 3e-3, 2.329e3)
eSnolab = Electronics(fSnolab)

# Name, fridge, absorber, n_channel
det = Detector("PD2", fSnolab, absorber, 1)
simulate_noise(det)

