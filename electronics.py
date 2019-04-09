class Electronics:

    def __init__(self, fridge, R_S, R_P, l_squid, l_p, si_squid):

        self._fridge = fridge

        # -- Shunt Resistor
        self._R_S = R_S
        self._T_S = fridge.get_TCP()

        # -- Parasitic resistances
        self._R_P = R_P
        self._T_P = fridge.get_TMC()
        # self._T_PT = self._R_P * fridge.

        # -- Inductances
        self._l_squid = l_squid
        self._l_p = l_p

        # -- Current noise from squid A/√Hz
        self._si_squid = si_squid


    def get_l_squid(self):
        return self._l_squid

    def get_l_p(self):
        return self._l_p
