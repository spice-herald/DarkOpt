import numpy as np

class SimpleEquilibrium:

    def __init__(self, detector, beta=0, Qp=0):
        self._det = detector
        self._beta = beta
        self._TES = self._det.get_TES()
        self._TES.set_Qp(Qp)

        #  ---- Bias Point Temperature ----
        # let's calculate the temperature of the operating point resistance.
        # [Notice that if the resistance of the TES changes with current, then this doesn't work]

        zeta_o = np.log(self._TES.get_fOp()/(1 - self._TES.get_fOp()))/2

        # Attempting to replicate SimpleEquilibrium_1TES line 51 with Tc_ResPt.m line 65-66. wTc doesn't show up
        # anywhere else! wTc calculated using this way in TES.py 32-33

        wTc = self._TES.get_wTc()
        Tc = self._TES.get_Tc()
        To = self._TES.get_To()

        self._TES.set_To(zeta_o * wTc + Tc) # K

        # ----- Alpha/Beta at Transition Point -----

        alpha = 2 * To / wTc / (np.exp(zeta_o)/(np.exp(zeta_o) + np.exp(-zeta_o)))

        self._TES.set_alpha(alpha)
        self._TES.set_beta(beta)

        # ---- TES properties at equilibrium ----

        # Heat Capacity

        # Bias Power (Phonon/Electron coupling G already set in TES.py)

        K = self._TES.get_K()
        n = self._TES.get_n()
        T_MC = detector.get_fridge().get_TMC()

        po = K * ((To ** n) - (T_MC ** n)) - Qp # W
        self._TES.set_Po(po)  # W

        # Loop Gain

        Gep = self._TES.get_G()
        lg = alpha * po / (To * Gep)
        self._TES.set_LG(lg)

        # Current At Equilibrium

        ro = self._TES.get_Ro()
        Io = np.sqrt(po/ro)
        self._TES.set_Io(Io)

        # Bias Voltage

        rl = detector.get_electronics().get_RL()
        Vbias = Io * (rl + ro) # V
        self._TES.set_Vbias(Vbias)

        # Heat Capacity

        # Tungsten values taken from MaterialProperties.m line 385 / 376
        # TODO Check if these values are the ones meant to be taken.
        fCsn = 2.43
        gC_v = 108
        vol = self._TES.get_volume()
        C = fCsn * gC_v * To * vol
        self._TES.set_C(C)

        # Inverse Bandwidth
        tau0 = C/Gep
        self._TES.set_tau0(tau0)

        # Sensor Bandwidth
        r_ratio = rl / ro
        tau_etf = tau0 / (1 + lg * (1 - r_ratio)/(1 + beta + r_ratio))
        self._TES.set_tau_etf(tau_etf)
        self._TES.set_w_etf(1/tau_etf)



