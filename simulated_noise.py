from simple_equilibrium import simple_equilibrium
import numpy as np
# """Final Result: sigPt_of = sqrt(Det.nP)*sigPt_of_1chan; %[eV]"""

def dynamical_response(detector):

    n_omega = 1e5
    omega = np.logspace(-1, 5.5, n_omega) * 2 * np.pi

    detector.set_response_omega(omega)

    # Plotting variables
    lgc_plt = False
    lgc_pltsimp = False

    w_Pabsb = detector.get_collection_bandwidth()
    dPtdE = detector.get_eEabsb() / (1 + 1j * omega/w_Pabsb)
    detector.set_dPtdE(dPtdE)


    lgc_1oF = False

    TES = detector.get_TES()

    # Calculate dIdP

    Gep = TES.get_G()
    LG = TES.get_LG()
    C = TES.get_C()
    Io = TES.get_Io()
    Lt = detector.get_electronics().get_lt()
    Rl = detector.get_electronics().get_RL()
    Ro = TES.get_Ro()
    beta = TES.get_beta()

    dIdPt = -(Gep * LG / (C * Io * Lt)) * (1 / (1j*omega + Gep * (1 - LG) / C) *
                                           (1j*omega + (Rl + Ro(1 + beta))/Lt) +
                                           (LG * (Ro/Lt) * (Gep/C) * (2 + beta)))

    detector.set_dIdP(dIdPt)

    # Calculate dIdV
    z_tes = Ro * (1 + beta) + \
            (Ro * LG /(1 - LG)) *\
            (2 + beta)/(1 + 1j*omega * C/(Gep * (1 - LG)))

    z_tot = Rl + 1j*omega * Lt + z_tes

    detector.set_ztes(z_tes)
    detector.set_ztot(z_tot)
    detector.set_dIdV(1/z_tot)

    # Calculate dIdV step functions
    tau0 = C/Gep
    TES.set_tau0(tau0)

    # Exponential rise time for the current biased circuit
    tau_I = tau0 / (1 - LG)
    TES.set_tau_I(tau_I)
    TES.set_w_I(1/tau_I)

    # L/R time constant under assumption of no pole mixing
    tau_el = Lt / (Rl + Ro * (1 + beta))
    TES.set_tau_el(tau_el)
    TES.set_w_el(1/tau_el)

    # ETF time constant under assumption of no pole mixing
    #     Det.TES.tau_etf_simp = Det.TES.tau0. / (
    #1 + (1 - Det.elec.Rl. / Det.TES.Ro). / (1 + Det.TES.beta + Det.elec.Rl / Det.TES.Ro). * Det.TES.LG)

    ratio = Rl/Ro
    tau_etf_simp = tau0 / (1 + (1 - ratio)/(1 + beta + ratio) * LG )
    TES.set_tau_etf_simp(tau_etf_simp)
    TES.set_w_etf_simp(1/tau_etf_simp)

    # ---------------------------------------------------------------

def simulate_noise(detector):
    # Put all relevant TES parameters at equilibrium value.
    simple_equilibrium(detector)
    dynamical_response(detector)



