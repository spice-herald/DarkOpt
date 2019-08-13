class Electronics:
    
    # future: 
    # electronics should always use the fridge 
    # electronic used in any fridge 
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
        

        # ---- Calculate mask Inductance 
        # ---- Inner Circle: radius 24000um 24 pairs of wires each side, all of different length and different n
        r = 24000 #um [radius of inner circle]
        d = 2060 #um [distance between wire pairs]
        w = 8 #um [width of wire]
        dl = 1600 #um [Delta l between tes]
        dy = 2000 #um [y distance between wires]
        # ---- One half of inner circle 
        for i in range(12): # one quarter of inner circle
            l = sqrt(r**2 - (r-dy*i)**2)
            n = l/dl 
        self._l_squid = l_squid
        self._l_p = l_p
        self._lt = l_squid + l_p

        # -- Current noise from squid A/√Hz
        self._si_squid = si_squid

