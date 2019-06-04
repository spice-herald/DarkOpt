from fridge import Fridge
from detector import Detector
from electronics import Electronics
from absorber import Absorber
from simulated_noise import simulate_noise
from scipy.optimize import minimize

fSnolab = Fridge("SNOLAB", 20e-3, 145e-3, 900e-3, 4.8, 0)
absorber = Absorber("Si", 1e-3, 38.1e-3, 3e-3, 2.329e3)
eSnolab = Electronics(fSnolab, fSnolab.get_TCP(), fSnolab.get_TMC())
eSLAC = Electronics(fSnolab, fSnolab.get_TMC(), fSnolab.get_TMC(), 5e-3, 6e-3, 25e-9, 25e-9, 4e-12)

# Name, fridge, absorber, n_channel
det = Detector("PD2", fSnolab, eSLAC, absorber, 1)
a = simulate_noise(det)
print(a)


def main_func(params):
    n_TES, l_TES, l_fin, h_fin, l_overlap, w_rail_main, w_rail_qet = params
    det = Detector("PD2", fSnolab, eSLAC, absorber, 1, n_TES, l_TES, l_fin, h_fin, l_overlap, w_rail_main, w_rail_qet)
    return simulate_noise(det)

initial_guess = [1185, 140e-6, 200e-6, 600e-9, 10e-6, 8e-6, 4e-6]
bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (8e-6, None), (4e-6, None))
result = minimize(main_func, initial_guess, bounds=bnds)

det = Detector("PD2", fSnolab, eSLAC, absorber, 1, *result.x)
lit = simulate_noise(det)

print("Params are %s, gives value %s" % (result.x, lit))
print(result.success)
print(result.message)