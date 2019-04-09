class Fridge:

    def __init__(self, name, T_MC, T_CP, T_Still, T_4K, parasitic):
        """
        Fridge Class 
        :param name: Name of Fridge
        :param T_MC: XXX Find out what all these Temperatures are 
        :param T_CP: 
        :param T_Still: 
        :param T_4K: 
        :param parasitic: Some type of parasitic power? 
        """
        self._name = name
        self._T_MC = T_MC
        self._T_CP = T_CP
        self._T_Still = T_Still
        self._T_4K = T_4K
        self._parasitic = parasitic

    def get_name(self):
        return self._name

    def get_TMC(self):
        return self._T_MC

    def get_TCP(self):
        return self._T_CP

    def get_T_still(self):
        return self._T_Still

    def get_T_4K(self):
        return self._T_4K

    def get_parasitic(self):
        return self._parasitic
    