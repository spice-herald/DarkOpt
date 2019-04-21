class Electronics:

    def __init__(self, fridge, R_S=5e-3, R_P=2e-3, l_squid=75e-9, l_p=25e-9, si_squid=6e-12):
        """Default values from eSNOLAB.m"""
        self._fridge = fridge

        # -- Shunt Resistor
        self._R_S = R_S
        self._T_S = fridge.get_TCP()

        # -- Parasitic resistances
        self._R_P = R_P
        self._T_P = fridge.get_TMC()
        # self._T_PT = self._R_P * fridge.

        # -- Total Load Resistance
        self._R_L = self._R_P + self._R_S

        # -- Inductances
        self._l_squid = l_squid
        self._l_p = l_p

        # -- Current noise from squid A/√Hz
        self._si_squid = si_squid


    def get_l_squid(self):
        return self._l_squid

    def get_l_p(self):
        return self._l_p

    def get_RL(self):
        return self._R_L
