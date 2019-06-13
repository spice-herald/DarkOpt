from fridge import Fridge
from detector import Detector
from electronics import Electronics
from absorber import Absorber
from simulated_noise import simulate_noise
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

fSnolab = Fridge("SNOLAB", 20e-3, 145e-3, 900e-3, 4.8, 0)
absorber = Absorber("Si", 1e-3, 38.1e-3, 3e-3, 2.329e3)
eSnolab = Electronics(fSnolab, fSnolab.get_TCP(), fSnolab.get_TMC())
eSLAC = Electronics(fSnolab, fSnolab.get_TMC(), fSnolab.get_TMC(), 5e-3, 6e-3, 25e-9, 25e-9, 4e-12)

# Name, fridge, absorber, n_channel
PD2 = Detector("PD2", fSnolab, eSLAC, absorber, 1)
simulate_noise(PD2)
#print(a)

h_fin = 600e-9

def main_func(TES_L, fin_L, overlap_L):
    #l_TES, l_fin, l_overlap = params
    #n_TES = 1185
    #l_overlap = 15e-6
    # TES number is garbage because will be set later.
    det = Detector("PD2", fSnolab, eSLAC, absorber, 1, 69, TES_L, fin_L, h_fin, overlap_L)
    return simulate_noise(det)

initial_guess = np.array([140e-6, 200e-6, 10e-6])
#bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (8e-6, None), (4e-6, None))
bnds = ((0, None), (0, None), (0, None))
"""result = minimize(main_func, initial_guess, bounds=bnds)

det = Detector("PD2", fSnolab, eSLAC, absorber, 1, *result.x)
lit = simulate_noise(det)

print("Params are %s, gives value %s" % (result.x, lit))
#print(result.success)
#print(result.message)"""

size = 25
lower_limit_fin = 50e-6
upper_limit_fin = 300e-6
upper_limit_tes = 500e-6
lower_limit_tes = 70e-6
overlap = 10e-6

l_TES_range = np.linspace(lower_limit_tes, upper_limit_tes, 2*size)
l_fin_range = np.linspace(lower_limit_fin, upper_limit_fin, size)

l_overlap_range = np.arange(0, 15e-6, 1e-7)

z = np.zeros(shape=(size, 2*size))
minimum = 1e12
x_coord, y_coord = 0, 0
for i in range(size):
    for j in range(2*size):
        TES_L = l_TES_range[j]
        fin_L = l_fin_range[i]
        val = main_func(TES_L, fin_L, overlap)
        z[i, j] = val

        if val < minimum:
            minimum = val
            x_coord = fin_L
            y_coord = TES_L

        print("i: %s j: %s - Value: %s" % (i, j, val))

print(">>>>> Min Location: (%s, %s) Value: %s" % (x_coord, y_coord, minimum))

#z = main_func(x_TES, y_FIN, 10e-6)
plt.imshow(z, interpolation='bilinear',
           origin='lower', extent=[lower_limit_tes*1e6,upper_limit_tes*1e6,lower_limit_fin*1e6, upper_limit_fin*1e6],cmap='Reds')
plt.colorbar()
plt.xlabel("TES Length / µm")
plt.ylabel("Fin Length / µm")

#z = main_func(l_TES_range, l_fin_range, 15e-6)


TES_L = y_coord
fin_L = x_coord
det = Detector("PD2", fSnolab, eSLAC, absorber, 1, 1185, TES_L, fin_L, h_fin, overlap)

a = simulate_noise(det)
print(a)
print(np.amin(z))

plt.title("Overlap Length %.1f µm. \n Min = %.3f at TES Length = %.1f Fin Length = %.1f" % (overlap*1e6, minimum, TES_L*1e6, fin_L*1e6))

plt.show()



