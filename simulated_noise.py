from simple_equilibrium import simple_equilibrium
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k as kb
from scipy.constants import e

# """Final Result: sigPt_of = sqrt(Det.nP)*sigPt_of_1chan; %[eV]"""

def dynamical_response(detector):

    n_omega = int(1e5)
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
    tau_etf_simp = tau0 / (1 + (1 - ratio)/(1 + beta + ratio) * LG)
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

    if lgc_plt:
        # Plot dIdPt magnitude just as a test, can put the other plots later
        plt.plot(omega/(2 * np.pi), np.abs(dIdPt))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('dIdP [1/V]')
        plt.title('Magnitude of dIdPt')
        plt.semilogx()
        plt.semilogy()
        plt.grid()
        plt.show()

        plt.plot(omega/(2*np.pi), np.angle(dIdPt) * (180/np.pi))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase of dIdPt [Deg]')
        plt.title('Phase of dIdPt')
        plt.semilogx()
        plt.grid()
        plt.show()

        plt.plot(omega/(2*np.pi), np.abs(detector.get_dIdV()))
        plt.semilogx()
        plt.semilogy()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('dIdV [1/Ω]')
        plt.title('Magnitude of dIdV')
        plt.grid()
        plt.show()

        plt.plot(omega / (2 * np.pi), np.angle(detector.get_dIdV()) * 180 / np.pi, c='b', label='Normal')
        plt.plot(omega / (2 * np.pi), np.angle(dIdV_chk) * 180 / np.pi, c='r',label='Check')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase of dIdV [Deg]')
        plt.title('Phase of dIdV')
        plt.semilogx()
        plt.grid()
        plt.legend()
        plt.show()

        plt.plot(np.real(z_tot), np.imag(z_tot), label='Z_tot', c='black')
        plt.plot(np.real(z_tes), np.imag(z_tes), label='Z_tes', c='blue')
        plt.xlabel("Re(Z)")
        plt.ylabel("Im(Z)")
        plt.title("Im(Z) vs Re(Z)")
        plt.grid()
        plt.legend(loc='best')
        plt.show()

        plt.plot(t, dIdV_step)
        plt.plot(t, dIdV0 * np.ones(n_t), c='green')
        plt.plot(t, dIdVmid * np.ones(n_t), 'blue')
        plt.xlabel('time [s]')
        plt.ylabel('dIdV Step Function [1/Ohm]')
        plt.title('Step function voltage response')
        plt.grid()
        plt.show()



def Ftfn(Tl, Th, n, isBallistic):

    if isBallistic:
        return ((Tl/Th) ** (n + 1) + 1)/2
    return (n/(2*n+1)) * ((Tl/Th) ** (2*n + 1) - 1)/((Tl/Th) ** n - 1)


def simulate_noise(detector):
    # Put all relevant TES parameters at equilibrium value.
    simple_equilibrium(detector, beta=0, Qp=0)
    dynamical_response(detector)

    omega = detector.get_response_omega()
    n_omega = omega.size

    TES = detector.get_TES()

    TES.set_fSp_xtra(0)
    lgc_xtraNoise = False

    # Squid Noise -------------------------
    #print(">>>>>>>SI_Squid %s" % detector.get_electronics().get_si_squid())
    SI_squid = detector.get_electronics().get_si_squid() ** 2 * np.ones(n_omega) # A^2 / Hz

    dIdPt = detector.get_dIdP()
    SPt_squid = SI_squid / abs(dIdPt ** 2)  # W / Hz

    # Johnson Load Noise ------------------

    Tl = detector.get_electronics().get_TL()
    #print(">>>>> Tl %s" % Tl)
    Rl = detector.get_electronics().get_RL()
    Sv_l = 4 * kb * Tl * Rl  # V^2 / Hz
    Si_RlSC = Sv_l / (Rl ** 2)  # A^2 / Hz FIXME location of square

    dIdV = detector.get_dIdV()
    Si_Rl = abs(dIdV ** 2) * Sv_l # A^2 / Hz
    Spt_Rl = Si_Rl / abs(dIdPt ** 2)

    # Johnson TES noise ------------------
    To = TES.get_To()
    Ro = TES.get_Ro()
    beta = TES.get_beta()
    Sv_t = 4 * kb * To * Ro * (1 + beta) ** 2  # FIXME Is the square in the right place? Dimensionally this makes sense.

    Io = TES.get_Io()

    Si_Rt = abs(dIdV - Io * dIdPt) ** 2 * Sv_t # A^2 / Hz
    Spt_Rt = Si_Rt / abs(dIdPt ** 2) # W^2 / Hz

    # Phonon Cooling Noise across TES-Bath conductance ----------------
    Gep = TES.get_G()
    T_MC = detector.get_fridge().get_TMC()
    nPep = TES.get_n()
    Spt_Gtb = 4 * kb * To ** 2 * Gep * Ftfn(T_MC, To, nPep, False) * np.ones(n_omega)

    Si_Gtb = abs(dIdPt ** 2) * Spt_Gtb

    # Unexplained Noise scaling as Pt
    Spt_xtra_Sp = Spt_Gtb * TES.get_fSp_xtra() ** 2
    Si_extra_Sp = Si_Gtb * TES.get_fSp_xtra() ** 2

    # --------- TOTAL NOISE TERMS (EXCEPT 1/F) -----------

    Si_tot = Si_Rl + Si_Rt + Si_Gtb + SI_squid + Si_extra_Sp
    Spt_tot = Spt_Rl + Spt_Rt + Spt_Gtb + SPt_squid + Spt_xtra_Sp

    # -------- Optimal Filtering ------------
    domega = np.zeros(n_omega)
    domega[1: n_omega - 2]= (omega[2:n_omega-1]-omega[0:n_omega - 3])/2
    domega[0] = (omega[1] - omega[0]) / 2
    domega[n_omega-1] = (omega[n_omega-1] - omega[n_omega - 2]) / 2
    #print(">>>>>>>> omega:")
    #print(omega)
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    #print(">>>>>>>> domega:")
    #print(domega)
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    dPtdE = detector.get_dPtdE()
    sigPt_of_1chan = np.sqrt(1/(domega/(2*np.pi)*4*np.abs(dPtdE)**2/Spt_tot).sum())/e
    n_channel = detector.get_n_channel()
    #print(">>> n_channel %s" % n_channel)
    sigPt_of = np.sqrt(n_channel) * sigPt_of_1chan

    #print(">>>>>>>>>>>>>>>>>>>>>> RESOLUTION IS %s" % sigPt_of)
    return sigPt_of
    """
    plt.plot(omega/(2*np.pi),np.sqrt(SI_squid)*1e12,'yellow', label='Squid')
    plt.plot(omega/(2*np.pi),np.sqrt(Si_Rl)*1e12,'red', label='R_load')
    plt.plot(omega/(2*np.pi),np.sqrt(Si_Rt)*1e12,'green', label='R_tes')
    plt.plot(omega/(2*np.pi),np.sqrt(Si_Gtb)*1e12,'cyan', label='G TES-Bath')
    plt.plot(omega / (2 * np.pi), np.sqrt(Si_tot) * 1e12, 'black', label='Total')
    plt.grid()
    plt.legend(loc='best')
    plt.semilogx()
    plt.semilogy()
    plt.title("TES Current Noise", fontsize=20)
    plt.xlabel("F [Hz]", fontsize=20)
    plt.ylabel("S_I [pA/√Hz]", fontsize=20)
    #plt.show()

    plt.plot(omega/(2*np.pi),np.sqrt(SPt_squid),'yellow', label='Squid')
    plt.plot(omega/(2*np.pi),np.sqrt(Spt_Rl),'red', label='R_load')
    plt.plot(omega/(2*np.pi),np.sqrt(Spt_Rt),'green', label='R_tes')
    plt.plot(omega/(2*np.pi),np.sqrt(Spt_Gtb),'cyan', label='G TES-Bath')
    plt.plot(omega/(2*np.pi),np.sqrt(Spt_tot),'black', label='Total')
    plt.title("TES Power Noise", fontsize=20)
    plt.xlabel("F [Hz]", fontsize=20)
    plt.ylabel("S_P [W/√Hz]", fontsize=20)
    plt.grid()
    plt.legend(loc='best')
    plt.semilogy()
    plt.semilogx()
    #plt.show()"""

