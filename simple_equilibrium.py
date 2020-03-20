import numpy as np

# simple because beta=0 
# 
def simple_equilibrium(detector, beta=0, Qp=0):
    _det = detector
    _beta = beta
    _TES = _det._tes # the tes object
    _TES.set_Qp(Qp)
    #  ---- Bias Point Temperature ----
    # let's calculate the temperature of the operating point resistance.
    # [Notice that if the resistance of the TES changes with current, then this doesn't work]

    zeta_o = np.log(_TES._fOp/(1 - _TES._fOp))/2
    
    # Attempting to replicate SimpleEquilibrium_1TES line 51 with Tc_ResPt.m line 65-66. wTc doesn't show up
    # anywhere else! wTc calculated using this way in TES.py 32-33

    wTc = _TES._wTc
    Tc = _TES._T_c

    _TES.set_To(zeta_o * wTc + Tc) # K
    
    To = _TES._T_eq
    
    n = _TES._n
    K = _TES._K

    Gep = n * K * To ** (n-1)
    _TES.set_G(Gep)

    # ----- Alpha/Beta at Transition Point -----

    alpha = 2*To/wTc/np.exp(zeta_o)/(np.exp(zeta_o)+np.exp(-zeta_o))

    _TES.set_alpha(alpha)
    _TES.set_beta(beta)

    # ---- TES properties at equilibrium ----

    # Bias Power (Phonon/Electron coupling G already set in TES.py)

    K = _TES._K
    n = _TES._n
    T_MC = detector._fridge.get_TMC()

    po = K * ((To ** n) - (T_MC ** n)) - Qp # W
    _TES.set_Po(po)  # W

    # Loop Gain

    Gep = _TES._G
    lg = alpha * po / (To * Gep)
    _TES.set_LG(lg)

    # Current At Equilibrium

    ro = _TES._res_o
    Io = np.sqrt(po/ro)

    _TES.set_Io(Io)

    # Bias Voltage

    rl = _det._electronics._R_L
    Vbias = Io * (rl + ro)  # V
    _TES.set_Vbias(Vbias)

    # Heat Capacity

    # Tungsten values taken from MaterialProperties.m line 385 / 376
    fCsn = _TES._material._fCsn # matches matlab 
    gC_v = _TES._material._gC_v # matches matlab 
    vol = _TES._tot_volume
    C = fCsn * gC_v * To * vol
    _TES.set_C(C)
    

    # Inverse Bandwidth
    tau0 = C/Gep
    _TES.set_tau0(tau0)

    # Sensor Bandwidth
    r_ratio = rl / ro
    tau_etf = tau0 / (1 + lg * (1 - r_ratio)/(1 + beta + r_ratio))
    #tau_etf = 66e-6
    _TES.set_tau_etf(tau_etf)
    _TES.set_w_etf(1/tau_etf)

    Gep = _TES._G
    LG = _TES._LG
    C = _TES._C
    Io = _TES._Io
    Lt = detector._electronics._lt
    Rl = detector._electronics._R_L
    Ro = _TES._res_o
    beta = _TES._beta
    Po = _TES._Po
    Vbias = _TES._Vbias
    t0 = _TES._tau0
    tau_etf = _TES._tau_etf
    w_etf = _TES._w_etf
    To = _TES._T_eq

    if detector._tes._print == True:    
        print("---------------- EQUILIBRIUM PARAMETERS ----------------")
        print("wTc %s" %wTc)
        print("zeta_o %s" %zeta_o) 
        print("Tc %s" %Tc) 
        print("To %s" %To) 
        print("K %s" %K) 
        print("n %s" %n) 
        print("T_MC %s" %T_MC) 
        print("Ro %s" %ro) 
        print("Io %s" %Io) 
        print("Rl %s" %rl) 
        print("fCsn %s"%fCsn)
        print("gC_v %s" %gC_v) 
        print("vol %s "% vol)
        print("tau0 %s "% tau0)
        print("tau_etf %s "% tau_etf)
        print("r_ratio %s "% r_ratio)
        print("Lt%s "% Lt)
        print("alpha %s" % _TES._alpha)
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
        print("----------------------------------------------------\n")
