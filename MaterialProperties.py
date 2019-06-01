from numpy import pi, log
from scipy.constants import k, N_A

class DetectorMaterial:

    def __init__(self, name, A, Z, rho_mass, E_eh, E_gap, fano_ER, fano_NR, Er_NRDisp, Td,
                 lat_spacing, phonon_v_phase, a_iso_scat, a_ah_dc, fUThK=1.0):

        self._name = name
        # Nuclear Number
        self._A = A
        # Atomic Number
        self._Z = Z
        # Mass Density
        self._rho_m = rho_mass
        # Number Density
        self._rho_n = rho_mass/A*(N_A * 1e3)
        # Ionisation Energy
        self._E_eh = E_eh
        # Bandgap
        self._E_gap = E_gap
        # Fano Factor ER
        self._Fano_ER = fano_ER
        # Fano Factor NR
        self._Fano_NR = fano_NR
        # The minimum recoil energy required to displace a nucleus in a lattice
        self._Er_NRDisp = Er_NRDisp
        # Debye Temperature
        self._Td = Td
        # Lattice Spacing
        self._lat_spacing = lat_spacing
        # Phonon Speed along [100] direction
        self._v_phase = phonon_v_phase
        # Here we average over the density of states assuming spherical symmetry
        # dn/dw = differential density of states with respect to angular
        # frequency = 4*pi* w^2 / v^3
        self._v_phase_avg = ((1/phonon_v_phase**2).sum())/((1/phonon_v_phase**3).sum()) # m/s
        # Fraction of longitudinal phonons at a given frequency (isotropic assumption)
        self._f_long = 1/(((phonon_v_phase[2]/phonon_v_phase)**3).sum())
        # Isotopic Scattering Gamma = a_iso_scat * w^4
        self._a_iso_scat = a_iso_scat
        # Anharmonic Downconversion Gamma = a_ah_dc * w^5
        self.a_ah_dc = a_ah_dc
        # Phonon Heat Capacity Coefficient
        self._gC_v = 12*pi**4/5*k*self._rho_n/Td**3
        # Photon Background Ratio (compared to Ge)
        self._fUThK = fUThK


    def get_A(self):
        return self._A

    def get_Z(self):
        return self._Z

    def get_rho_mass(self):
        return self._rho_m

    def get_rho_number(self):
        return self._rho_n

    def get_E_eh(self):
        return self._E_eh

    def get_E_gap(self):
        return self._E_gap

    def get_fano_ER(self):
        return self._Fano_ER

    def get_fano_NR(self):
        return self._Fano_NR

    def get_recoil_E(self):
        return self._Er_NRDisp

    def get_debye_T(self):
        return self._Td

    def get_lattice_spacing(self):
        return self._lat_spacing

    def get_v_phase(self):
        return self._v_phase

    def get_v_phase_avg(self):
        return self._v_phase_avg

    def get_isotopic_coeff(self):
        return self._a_iso_scat

    def get_anharmonic_coeff(self):
        return self.a_ah_dc

    def get_gC_v(self):
        return self._gC_v

    def get_fUThK(self):
        return self._fUThK

class TESMaterial:

    def __init__(self, name="W", Z=74, A=183.84, rho_mass=19.25e3, gC_v=108, gC2_mol=1.01e-3, nC=1, fCsn=2.43,
                 gPep_v=0.22e9, nPep=5, nPee=2, rho_electrical=0.600*(40e-9*400e-6)/100e-6, Tc=40e-3):

        self._name = name,
        self._Z = Z
        self._A = A
        self._rho_m = rho_mass
        self._rho_n = rho_mass/A*(N_A * 1e3)
        self._gC_v = gC_v
        self._gC2_mol = gC2_mol
        self._gC2_v = gC2_mol / N_A * self._rho_n
        self._nC = nC
        self._fCsn = fCsn
        self._gPep_v = gPep_v
        self._nPep = nPep
        self._nPee = nPee
        self._rho_electrical = rho_electrical
        self._Tc = Tc
        self._wTc_1090 = 3.9e-4* Tc/40e-3
        self._wTc = self._wTc_1090/2/log(3)


    def get_A(self):
        return self._A

    def get_Z(self):
        return self._Z

    def get_rho_mass(self):
        return self._rho_m

    def get_rho_number(self):
        return self._rho_n

    def get_gC_v(self):
        return self._gC_v

    def get_gC2_mol(self):
        return self._gC2_mol

    def get_gC2_v(self):
        return self._gC2_v

    def get_nC(self):
        return self._nC

    def get_fCsn(self):
        return self._fCsn

    def get_gPep_v(self):
        return self._gPep_v

    def get_nPep(self):
        return self._nPep

    def get_nPee(self):
        return self._nPee

    def get_rho_electric(self):
        return self._rho_electrical

    def get_Tc(self):
        return self._Tc

    def get_wTc(self):
        return self._wTc



