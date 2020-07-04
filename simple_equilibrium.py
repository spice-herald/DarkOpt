import numpy as np

# simple because beta=0 
# 
def dynamical_response(detector):
    n_omega = int(1e6)
    #omega = np.logspace(-1, 5.5, n_omega) * 2 * np.pi
    omega = np.logspace(-1, 7.5, n_omega) * 2 * np.pi

    detector.set_response_omega(omega)


    w_Pabsb = detector._w_collect
    dPtdE = detector._eEabsb / (1 + 1j * omega/w_Pabsb)
    detector.set_dPtdE(dPtdE)


    lgc_1oF = False

    TES = detector._tes # tes object

    # Calculate dIdP

    Gep = TES._G
    LG = TES._LG
    C = TES._C
    Io = TES._Io
    Lt = detector._electronics._lt
    Rl = detector._electronics._R_L
    Ro = TES._res_o
    beta = TES._beta

    #print(">>> Rl %s" % Rl)
    dIdPt = -(Gep * LG / (C * Io * Lt)) * (1 / (1j*omega + Gep * (1 - LG) / C) *
                                           (1j*omega + (Rl + Ro*(1 + beta))/Lt) +
                                           (LG * (Ro/Lt) * (Gep/C) * (2 + beta)))

    dIdPt =  -Gep * LG / (C * Io * Lt) / ((1j * omega + Gep * (1 - LG) / C) * (1j * omega + (Rl + Ro * (1 + beta)) / Lt) + LG * Ro / Lt * Gep / C * (2 + beta))

    detector.set_dIdP(dIdPt)

    # Calculate dIdV
    z_tes = Ro * (1 + beta) + Ro * LG / (1 - LG) * (2 + beta) / (1 + 1j * omega * C / Gep / (1 - LG))
    z_tot = Rl + 1j * omega * Lt + z_tes

    detector.set_ztes(z_tes)
    detector.set_ztot(z_tot)
    detector.set_dIdV(1/z_tot)

    # Calculate dIdV step functions
    tau0 = C/Gep
    #print("tau0 ", tau0)
    TES.set_tau0(tau0)
    #print("Tau0: C %s Gep %s" % (C, Gep))

    # Exponential rise time for the current biased circuit
    tau_I = tau0 / (1 - LG)
    w_I = 1/tau_I
    TES.set_tau_I(tau_I)
    TES.set_w_I(w_I)
# L/R time constant under assumption of no pole mixing
    tau_el = Lt / (Rl + Ro * (1 + beta))
    TES.set_tau_el(tau_el)
    TES.set_w_el(1/tau_el)

    # ETF time constant under assumption of no pole mixing
    #     Det.TES.tau_etf_simp = Det.TES.tau0. / (
    #1 + (1 - Det.elec.Rl. / Det.TES.Ro). / (1 + Det.TES.beta + Det.elec.Rl / Det.TES.Ro). * Det.TES.LG)

    ratio = Rl/Ro
    tau_etf_simp = tau0 / (1 + LG*(1 - ratio)/(1 + beta + ratio))
    TES.set_tau_etf_simp(tau_etf_simp)
    TES.set_w_etf_simp(1/tau_etf_simp)


    # ---------------------------------------------------------------

    # Frequencies of poles taking into account pole mixing
    wp_avg = (1/tau_el)/2+(1/tau_I)/2
    dw = np.sqrt(((1 / tau_el) - (1. / tau_I)) ** 2 - 4 * (Ro / Lt) * LG * (2 + beta) / tau0) / 2

    wp_p = wp_avg + dw
    wp_m = wp_avg - dw

    taup_p = 1/wp_p
    taup_m = 1/wp_m

    #print("taup_m %s" % taup_m)

    TES.set_wpp(wp_p)  # High Frequency pole ~ L/R
    TES.set_wpm(wp_m)  # Low Frequency Pole ~ tau_eff

    TES.set_taupp(taup_p)
    TES.set_taupm(taup_m)

    dIdV0 = (1-LG)/(Rl+Ro*(1+beta)+ LG*(Ro-Rl))
    #print(">>> dIdV0 %s" % dIdV0)
    detector.set_dIdV0(dIdV0)

    # --------- check inversion equations which take wp_p, wp_m, w_z, dIdV(0)  to give
    # --------- LG, Lt, tau0, and beta.

    w_elchk = wp_p + wp_m - w_I

    E = ((w_elchk - w_I) ** 2 - (wp_p - wp_m) ** 2) / w_elchk / w_I

    beta_chk = 4/(E+4)/(dIdV0/Ro-Rl/Ro -1)
    LG_chk = (1/dIdV0 - (Rl + Ro * (1 + beta_chk)))/(Ro - Rl + 1/dIdV0)
    Lt_chk = (Rl + Ro * (1 + beta)) / w_elchk
    tau0_chk = (1-LG_chk)/w_I
# -------- dIdV step function

    n_t = int(1e4)
    t = np.linspace(0, 10, n_t) * taup_m

    # Calculate amplitude of dIdV between peaks
    dIdVmid = 1/(Rl + Ro * (1 + beta))

    dIdV_chk = dIdV0 * (1 + 1j * omega / w_I) / (1 + 1j * omega / wp_p) /(1 + 1j * omega / wp_m)

    dIdV_step = dIdV0 * (1 - (taup_p - tau_I) / (taup_p - taup_m) * np.exp(-t / taup_p) - (taup_m - tau_I) / (taup_m - taup_p) * np.exp(-t / taup_m))

    detector.set_dIdV_step(dIdV_step)
    detector.set_t(t)
    def make_plots():
        # Plot dIdPt magnitude just as a test, can put the other plots later
        plt.plot(omega/(2 * np.pi), np.abs(dIdPt))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('dIdP [1/V]')
        plt.title('Magnitude of dIdPt')
        plt.semilogx()
        plt.semilogy()
        plt.grid()
        #plt.show()

        plt.plot(omega/(2*np.pi), np.angle(dIdPt) * (180/np.pi))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase of dIdPt [Deg]')
        plt.title('Phase of dIdPt')
        plt.semilogx()
        plt.grid()
        #plt.show()

        plt.plot(omega/(2*np.pi), np.abs(detector.get_dIdV()))
        plt.semilogx()
        plt.semilogy()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('dIdV [1/Ω]')
        plt.title('Magnitude of dIdV')
        plt.grid()
        #plt.show()

        plt.plot(omega / (2 * np.pi), np.angle(detector.get_dIdV()) * 180 / np.pi, c='b', label='Normal')
        plt.plot(omega / (2 * np.pi), np.angle(dIdV_chk) * 180 / np.pi, c='r',label='Check')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase of dIdV [Deg]')
        plt.title('Phase of dIdV')
        plt.semilogx()
        plt.grid()
        plt.legend()
        #plt.show()

        plt.plot(np.real(z_tot), np.imag(z_tot), label='Z_tot', c='black')
        plt.plot(np.real(z_tes), np.imag(z_tes), label='Z_tes', c='blue')
        plt.xlabel("Re(Z)")
        plt.ylabel("Im(Z)")
        plt.title("Im(Z) vs Re(Z)")
        plt.grid()
        plt.legend(loc='best')
        #plt.show()

        plt.plot(t, dIdV_step)
        plt.plot(t, dIdV0 * np.ones(n_t), c='green')
        plt.plot(t, dIdVmid * np.ones(n_t), 'blue')
        plt.xlabel('time [s]')
        plt.ylabel('dIdV Step Function [1/Ohm]')
        plt.title('Step function voltage response')
        plt.grid()
        #plt.show()
    if detector._tes._print == True:
        print("---------- Response Parameters------------")
        print("Gep %s" %Gep)
        print("LG  %s" %LG)
        print("C %s" %C)
        print("Io %s" %Io)
        print("Lt %s" %Lt)
        print("Rl %s" %Rl)
        print("Ro %s" %Ro)
        print("beta  %s" %beta)
        print("-----------------------------------------\n")

def Ftfn(Tl, Th, n, isBallistic):

    if isBallistic:
        return ((Tl/Th) ** (n + 1) + 1)/2
    return (n/(2*n+1)) * ((Tl/Th) ** (2*n + 1) - 1)/((Tl/Th) ** n - 1)




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
