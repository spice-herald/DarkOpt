import numpy as np


def simple_equilibrium(detector, beta=0, Qp=0):
    _det = detector
    _beta = beta
    _TES = _det.get_TES()
    _TES.set_Qp(Qp)

    #  ---- Bias Point Temperature ----
    # let's calculate the temperature of the operating point resistance.
    # [Notice that if the resistance of the TES changes with current, then this doesn't work]

    zeta_o = np.log(_TES.get_fOp()/(1 - _TES.get_fOp()))/2

    # Attempting to replicate SimpleEquilibrium_1TES line 51 with Tc_ResPt.m line 65-66. wTc doesn't show up
    # anywhere else! wTc calculated using this way in TES.py 32-33

    wTc = _TES.get_wTc()
    Tc = _TES.get_Tc()

    _TES.set_To(zeta_o * wTc + Tc) # K
    To = _TES.get_To()

    n = _TES.get_n()
    K = _TES.get_K()

    Gep = n * K * To ** (n-1)
    _TES.set_G(Gep)

    # ----- Alpha/Beta at Transition Point -----

    alpha = 2*To/wTc/np.exp(zeta_o)/(np.exp(zeta_o)+np.exp(-zeta_o))

    _TES.set_alpha(alpha)
    _TES.set_beta(beta)

    # ---- TES properties at equilibrium ----

    # Bias Power (Phonon/Electron coupling G already set in TES.py)

    K = _TES.get_K()
    n = _TES.get_n()
    T_MC = detector.get_fridge().get_TMC()

    po = K * ((To ** n) - (T_MC ** n)) - Qp # W
    _TES.set_Po(po)  # W

    # Loop Gain

    Gep = _TES.get_G()
    lg = alpha * po / (To * Gep)
    _TES.set_LG(lg)

    # Current At Equilibrium

    ro = _TES.get_Ro()
    Io = np.sqrt(po/ro)

    _TES.set_Io(Io)

    # Bias Voltage

    rl = _det.get_electronics().get_RL()
    Vbias = Io * (rl + ro)  # V
    _TES.set_Vbias(Vbias)

    # Heat Capacity

    # Tungsten values taken from MaterialProperties.m line 385 / 376
    fCsn = _TES.get_material().get_fCsn()
    gC_v = _TES.get_material().get_gC_v() 
    vol = _TES.get_total_volume()
    C = fCsn * gC_v * To * vol
    _TES.set_C(C)

    # Inverse Bandwidth
    tau0 = C/Gep
    _TES.set_tau0(tau0)

    # Sensor Bandwidth
    r_ratio = rl / ro
    tau_etf = tau0 / (1 + lg * (1 - r_ratio)/(1 + beta + r_ratio))
    _TES.set_tau_etf(tau_etf)
    _TES.set_w_etf(1/tau_etf)

    Gep = _TES.get_G()
    LG = _TES.get_LG()
    C = _TES.get_C()
    Io = _TES.get_Io()
    Lt = detector.get_electronics().get_lt()
    Rl = detector.get_electronics().get_RL()
    Ro = _TES.get_Ro()
    beta = _TES.get_beta()
    Po = _TES.get_Po()
    Vbias = _TES.get_Vbias()
    t0 = _TES.get_tau0()
    tau_etf = _TES.get_tau_etf()
    w_etf = _TES.get_w_etf()
    To = _TES.get_To()


    print("---------------- EQUILIBRIUM PARAMETERS ----------------")
    print("To %s" % To)
    print("alpha %s" % _TES.get_alpha())
    print("beta %s" % beta)
    print("C %s" % C)
    print("Gep %s" % Gep)
    print("Po %s" % Po)
    print("LG %s" % LG)
    print("Io %s" % Io)
    print("VBias %s" % Vbias)
    print("tau0 %s" % t0)
    print("tau_etf %s" % tau_etf)
    print("w_etf %s" % w_etf)
    print("------------------------------------------------\n")



