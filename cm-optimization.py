import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from fridge import Fridge
from tes import TES
from QET import QET
from detector import Detector
from electronics import Electronics
from absorber import Absorber
from simulated_noise import simulate_noise
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from MaterialProperties import TESMaterial

printing = False
si_squid = 2e-12
fSnolab = Fridge("SNOLAB", 20e-3, 145e-3, 900e-3, 4.8, 0)
# Absorber: Silicon. Height 1mm. Radius 38.1mm. W safety 3mm. 
# Q: W safety removes a huge % of patterned surface area
absorber = Absorber("Si", "square", 1e-3, 10e-3, 3e-3, printing) # same as matlab
eSnolab = Electronics(fSnolab, fSnolab.get_TCP(), fSnolab.get_TMC(), si_squid)
eSLAC = Electronics(fSnolab, fSnolab.get_TMC(), fSnolab.get_TMC(),si_squid, 5e-3, 6e-3, 25e-9, 25e-9 )
tungsten = TESMaterial()

n_channel = 1
type_qp_eff = 0 

tes_w = 2.5e-6
h_tes = 40e-9
h_fin = 900e-9 
ahole = 49e-12
sigma = tungsten._gPep_v
T_eq = -100
l_overlaps = []
l_fins = []
tes_lengths = []

l_over_min = 5e-6
l_over_delta = 5e-6
#l_fin_min = 25e-6
#l_fin_min = 25e-6
l_fin_min = 50e-6
l_fin_delta = 50e-6
#l_fin_delta = 15e-6 # testing
tes_min = 25e-6
tes_delta = 10e-6

while l_over_min < 35e-6:
    #while l_over_min < 25e-6:
    l_overlaps.append(l_over_min)
    l_over_min = l_over_min + l_over_delta
#while l_fin_min < 300e-6:
while l_fin_min < 100e-6:
    l_fins.append(l_fin_min)
    l_fin_min = l_fin_min + l_fin_delta 
#while tes_min < 200e-6:
while tes_min < 30e-6:
    tes_lengths.append(tes_min)
    tes_min = tes_min + tes_delta 

l_fins_mu = []
l_overlaps_mu = []
tes_lengths_mu = []
for l_o in l_overlaps:
    l_overlaps_mu.append(l_o*(10**6))
for l_f in l_fins:
    l_fins_mu.append(l_f*(10**6))
for t in tes_lengths:
    tes_lengths_mu.append(t*(10**6))

n_fins = [2, 4]

#coverages = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04]\n",
#coverages = [ 0.0025, 0.005, 0.01, 0.04]
coverages = [0.01, 0.04]

passive = 0
optimum_detectors_mod = []
optimum_detectors_ell = []

for cov in coverages:
    print(cov*100, " % Coverage")
    for_cov_mod = []
    for_cov_ell = []
    for n_fin in n_fins:
        print("   -- ", n_fin, " Fins")
        min_resolution_mod = 200e10
        min_resolution_ell = 200e10
        optimum_detector_mod = []
        optimum_detector_ell = []
        for l_f in l_fins:
            print("     -- ", l_f , " Length Fin ")
            for tes_l in tes_lengths:
                print("        --", tes_l, " Length TES")
                for l_over in l_overlaps:
                    print("             -- ", l_over, " Overlap")
                    # Calculate QET Active Area
                    wempty = 6e-6
                    wempty_tes = 7.5e-6
                    nhole = 3*n_fin 
                    afin_empty = n_fin * l_f * wempty + 2 * tes_l * wempty_tes + nhole * ahole
                    a_fin = np.pi*l_f*(l_f + (tes_l/2)) - afin_empty
                    # Calculate number of TES for given Coverage
                    N_tes = cov*absorber._SA/a_fin
                    print("SA active ", N_tes*a_fin)
                    N_tes = int(N_tes)
                    if N_tes == 0: N_tes =1 
                    # Calculate Normal Resistance
                    
                    res_n = tungsten._rho_electrical*tes_l/(tes_w*h_tes*N_tes)
                    tes_ell = TES(tes_l, tes_w, l_over, n_fin, sigma, T_eq, res_n, 0.45, 'ellipse', tungsten, printing)
                    qet_ell = QET( l_f, h_fin, tes_ell, ahole)
                    det_ell = Detector("det name", fSnolab, eSLAC, absorber, qet_ell, tes_ell, passive,n_channel, type_qp_eff)
                    e_res_ell = simulate_noise(det_ell)
                    print("           -- Resolution (Elliptic Overlap) ", e_res_ell)
                    if e_res_ell < min_resolution_ell:
                        print("SET OPTIMUM")
                        min_resolution_ell = e_res_ell
                        tes_opt_ell = tes_ell
                        qet_opt_ell = qet_ell
                        det_opt_ell = det_ell 
                    # Only do modern fin connector design optimization if connectors fit
                    perim = tes_l*2 + 14e-6*2 - 6e-6*n_fin
                    overlap_p = n_fin*l_over*2
                    if overlap_p < perim:
                        tes_mod = TES(tes_l, tes_w, l_over, n_fin, sigma, T_eq, res_n, 0.45, 'modern', tungsten, printing)
                        qet_mod = QET( l_f, h_fin, tes_mod, ahole)
                        det_mod = Detector("det name", fSnolab, eSLAC, absorber, qet_mod, tes_mod, passive, 1, 0)
                        e_res_mod = simulate_noise(det_mod)
                        print("           -- E Resolution ", e_res_mod)
                        if e_res_mod < min_resolution_mod:
                            print("               set optimum")
                            min_resolution_mod = e_res_mod
                            tes_opt_mod = tes_mod
                            qet_opt_mod = qet_mod
                            det_opt_mod = det_mod
        optimum_detector_mod.append(tes_opt_mod)
        optimum_detector_mod.append(qet_opt_mod)
        optimum_detector_mod.append(det_opt_mod)
        optimum_detector_ell.append(tes_opt_ell)
        optimum_detector_ell.append(qet_opt_ell)
        optimum_detector_ell.append(det_opt_ell)
        for_cov_mod.append(optimum_detector_mod)
        for_cov_ell.append(optimum_detector_ell)
        #print("      -- Optumum Detector ", optimum_detector)
        #print("      -- Both N Fins      ", for_cov)
        #print("      -- BASELINE RESOLUTION: ", min_resolution) 
        #pActiveArea = det_opt._SA_active/det_opt._absorber.get_SA()\n",
        #pPassiveArea = det_opt._SA_passive/det_opt._absorber.get_SA()\n",
        print("      -- TES Length             ", tes_opt_mod._l)
        print("      -- Fin Length             ", qet_opt_mod._l_fin)
        print("      -- Overlap                ", qet_opt_mod.l_overlap)
        print("      -- N Fins                 ", tes_opt_mod._n_fin)
        print("      -- Percent Active SA      ", det_opt_mod._SA_active/det_opt_mod._absorber.get_SA())
        print("      -- Percent Passive SA     ", det_opt_mod._SA_passive/det_opt_mod._absorber.get_SA())
        print("      -- Percent QP Absorb SA   ", det_opt_mod._fSA_qpabsorb)
        print("      -- N tes                  ", det_opt_mod._tes._nTES)
        print("      -- Rn                     ", det_opt_mod._tes._total_res_n )
        print("      -- fQP Absorb             ", det_opt_mod._qet._eQPabsb)
        print("      -- Phonon Absorption Time ", det_opt_mod._t_pabsb)
        print("      -- Time ETF               ", det_opt_mod._tes._tau_etf)
        print("      -- Cells Fit              ", det_opt_mod._cells_fit)
        print("      -- Efficiency             ", det_opt_mod._eEabsb)
        print("      -- Resolution             ", simulate_noise(det_opt_mod))
    optimum_detectors_mod.append(for_cov_mod)
    optimum_detectors_ell.append(for_cov_ell)

passive_mod = [[],[]]
active_mod = [[],[]]
passive_tot_mod = [[],[]]

overlap_area_mod = [[],[]]
overlap_area_tes_l_mod = [[],[]]
e_resolution_mod = [[],[]]

qp_eff_mod = [[],[]]
for c in optimum_detectors_mod:
    for n in range(len(c)):
        tes = c[n][0]
        qet = c[n][1]
        det = c[n][2]
        passive_mod[n].append(det._SA_passive/det._absorber._SA)
        active_mod[n].append(det._SA_active/det._absorber._SA)
        passive_tot_mod[n].append(det._SA_passive/(det._SA_passive+det._SA_active))
        qp_eff_mod[n].append(qet._eQPabsb)
        overlap_area_mod[n].append(tes._A_overlap)
        overlap_area_tes_l_mod[n].append(tes._A_overlap/tes._l)
        e_resolution_mod[n].append(simulate_noise(det))

# ------------------------------------------------------------------------------------
passive_ell = [[],[]]
active_ell = [[],[]]
passive_tot_ell = [[],[]]
overlap_area_ell = [[],[]]
overlap_area_tes_l_ell = [[],[]]
e_resolution_ell = [[],[]]
qp_eff_ell = [[],[]]

for c in optimum_detectors_ell:
    for n in range(len(c)):
        tes = c[n][0]
        qet = c[n][1]
        det = c[n][2]
        passive_ell[n].append(det._SA_passive/det._absorber._SA)
        active_ell[n].append(det._SA_active/det._absorber._SA)
        passive_tot_ell[n].append(det._SA_passive/(det._SA_passive+det._SA_active))
        qp_eff_ell[n].append(qet._eQPabsb)
        overlap_area_ell[n].append(tes._A_overlap)
        overlap_area_tes_l_ell[n].append(tes._A_overlap/tes._l)
        e_resolution_ell[n].append(simulate_noise(det))

f, ax = plt.subplots(1)
ax.plot(active_mod[1], e_resolution_mod[1], "o", label = "4" )
ax.plot(active_mod[0], e_resolution_mod[0], "s", label = "2" )
plt.legend(title = "# of Fins")
ax.set_title("Passive Al vs Active Al")
ax.set_xlabel("Active Al Coverage")
ax.set_ylabel("Baseline Energy Resolution [eV]")
plt.show()
plt.savefig('e_res.png')

# Why is 2 fin design always Better?
f, ax = plt.subplots(1)
ax.plot(active_mod[1], qp_eff_mod[1], "o", label = "4" )
ax.plot(active_mod[0], qp_eff_mod[0], "s", label = "2" )
plt.legend(title = "# of Fins")
ax.set_title("QP Absorption Efficiency")
ax.set_xlabel("Active Al Coverage")
ax.set_ylabel("QP Absorption Efficiency")
plt.show()
plt.savefig('qp_absb_eff.png')
 
f, ax = plt.subplots(1)
ax.plot(active_mod[1], overlap_area_mod[1], "o", label = "4" )
ax.plot(active_mod[0], overlap_area_mod[0], "s", label = "2" )
plt.legend(title = "# of Fins")
ax.set_title("Passive Al vs Active Al")
ax.set_xlabel("Active Al Coverage")
ax.set_ylabel("Overlap Area")
plt.show()
plt.savefig('overlap_area.png')

f, ax = plt.subplots(1)
ax.plot(active_mod[1], overlap_area_tes_l_mod[1], "o", label = "4" )
ax.plot(active_mod[0], overlap_area_tes_l_mod[0], "s", label = "2" )
plt.legend(title = "# of Fins")
ax.set_title("Passive Al vs Active Al")
ax.set_xlabel("Active Al Coverage")
ax.set_ylabel("Overlap Area/TES Length")
plt.show()

f, ax = plt.subplots(1)
ax.plot(active_mod[1], passive_mod[1], "o", label = "4" )
ax.plot(active_mod[0], passive_mod[0], "s", label = "2" )
plt.legend(title = "# of Fins")
ax.set_title("Passive Al vs Active Al")
ax.set_xlabel("Active Al Coverage")
ax.set_ylabel("Passive Al Coverage")
plt.show()

f, ax = plt.subplots(1)
ax.plot(active_mod[1], passive_tot_mod[1], "o", label = "4" )
ax.plot(active_mod[0], passive_tot_mod[0], "s", label = "2" )
plt.legend(title = "# of Fins")
ax.set_title("Passive Al vs Active Al")
ax.set_xlabel("Active Al Coverage")
ax.set_ylabel("Passive Al Coverage")
plt.show()
