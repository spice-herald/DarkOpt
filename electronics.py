class Electronics:
    
    # future: 
    # electronics should always use the fridge 
    # electronic used in any fridge
    # Q: where do we put the cable valyes, inductance, capacitance??  
    def __init__(self, fridge, TS, TP, R_S=5e-3, R_P=6e-3, l_squid=75e-9, l_p=25e-9, si_squid=6e-12):
        """Default values from eSNOLAB.m"""
        self._fridge = fridge

        # -- Shunt Resistor
        self._R_S = R_S
        self._T_S = TS #fridge.get_TMC()

        # -- Parasitic resistances
        self._R_P = R_P
        self._T_P = TP #fridge.get_TMC()
        # self._T_PT = self._R_P * fridge.

        # -- Total Load Resistance
        self._R_L = self._R_P + self._R_S
        self._T_L = (R_P * self._T_P + R_S * self._T_S)/self._R_L

        # -- Inductances
        

        self._l_squid = l_squid
        self._l_p = l_p
        self._lt = l_squid + l_p

        # -- Current noise from squid A/√Hz
        self._si_squid = si_squid
    
