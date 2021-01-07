from numpy import pi, log
from scipy.constants import k, N_A


class DetectorMaterial():

    def __init__(self, name, fUThK=1.0):
        self._name = name

# ----------- GERMANIUM PROPERTIES ----------------
        """if name == 'Ge':
            # ADD GERMANIUM PROPERTIES 
            # Ballistic Phonon Absorption Time
            t_pabsb = 750e-6 # TODO SET THIS PROPERLY"""
# ------------ SILICON PROPERTIES -----------------
        if name == 'Si':
            A = 28.085
            Z=14
            rho_mass = 2.329e3 # kg/m^3 --> Wikipedia
            E_eh = 3.82 # eV average energy per eh pair-standard definition/measured; assumes electron recoil
            E_gap = 1.4  # eV
            fano_ER = .155  # from Owens et al, 'On the experimental determination of the Fano factor in Si at soft X-ray wavelengths'
            # https://doi.org/10.1016/S0168-9002(02)01178-6
            fano_NR = 1.0
     
            # The minimum recoil energy required to displace a nucleus in a lattice 
            Er_NRDisp= 15 # [eV] http://scitation.aip.org/content/aip/journal/apl/60/12/10.1063/1.107267
     
            # Debeye Temperature
            Td=645 # [K] --> wikipedia
     
            #---- lattice spacing-----
            lat_spacing = [5.431e-10] # [m]
     
            #---- Phonon Physics ----
            # Here are the sound speeds along the [100 direction]
            # http://www.ioffe.ru/SVA/NSM/Semicond/Si/mechanic.html
            phonon_v_phase =  [4670,5840,9130]  # [100 direction]
            #isotopic scattering rate:  Gamma = a_iso_scat * w^4
            # from Tamura, "Quasidiffusive propagation of phonons in silicon: 
            # Monte Carlo calculations" , https://doi.org/10.1103/PhysRevB.48.13502
            a_iso_scat=  2.43e6*1e-48 /(2*pi)**4 # [s^3] from Ge_prop_lattice
 
            # anharmonic down conversion rate: Gamma = a_ah_dc * w^5
            # Tamura, 'Quasidiffusive propagation of phonons in silicon: 
            # Monte Carlo calculations', https://doi.org/10.1103/PhysRevB.48.13502
            # We're using the simplified Maris rate calculation
            a_ah_dc = 4.1e4*1e-60 /(2*pi)**5  # [s^4] from Ge_prop_lattice
            # want percentage of phonons downconvert at surface 
            fDownSurf = 1e-10 # zero for now
 
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
        v_phase_2 = [phonon_v_phase[0]**2, phonon_v_phase[1]**2,  phonon_v_phase[2]**2]
        v_phase_3 = [phonon_v_phase[0]**3, phonon_v_phase[1]**3,  phonon_v_phase[2]**3]
        #self._v_phase_avg = ((1/phonon_v_phase**2).sum())/((1/phonon_v_phase**3).sum()) # m/s CHECK IF THIS IS RIGHT!! 
        #self._v_phase_avg = ((1/v_phase_2).sum())/((1/v_phase_3).sum()) # m/s CHECK IF THIS IS RIGHT!! 
        # Fraction of longitudinal phonons at a given frequency (isotropic assumption)
        #self._f_long = 1/(((phonon_v_phase[2]/phonon_v_phase)**3).sum())
        # Isotopic Scattering Gamma = a_iso_scat * w^4
        self._a_iso_scat = a_iso_scat
        # Anharmonic Downconversion Gamma = a_ah_dc * w^5
        self.a_ah_dc = a_ah_dc
        # Phonon Heat Capacity Coefficient
        self._gC_v = 12*pi**4/5*k*self._rho_n/Td**3
        # Photon Background Ratio (compared to Ge)
        self._fUThK = fUThK
        # Percent Phonons that Downconvert at Bare Si Surface 
        self._fDownSurf = fDownSurf


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
        # resistivity mesured by TAMU (has been very consistent in getting this value) 
        # much larger than predicted resistivity of W at low temperatures possibly because
        # A) we are thickness limited at 40nm so scattering length much smaller than expected at low T 
        # B) Our W has a lot of crystal defects, limiting the scattering length
        self._rho_electrical = rho_electrical # 9.6e-8 Ohm*m at 100mK [Normal Resistivity]
        # not sure where these Tc values are coming from but new TES chip (Caleb) has Tc = 40e-3 K  
        self._Tc = Tc
        #self._wTc_1090 = 3.9e-4* Tc/40e-3 # matlab has 3.65e-4 -SZ 
        self._wTc_1090 = 3.65e-4* Tc/40e-3 # matlab has 3.65e-4 -SZ
        a_factor = 0.73 # factor to match measured tau_eft = 66 microseconds
        self._wTc = a_factor*self._wTc_1090/2/log(3)

class QETMaterial():
    def __init__(self, name):
        self._name = name
        if name == 'Al':
            Z=13
            A=26.981
     
            # Efficiency of energy transfer from athermal phonons to quasi-particles 
            ePQP = 0.52
     
            # QuasiParticle trapping length in our current Al fins
            # ldiffQP = 135e-6 # [m]
            #prop.ldiffQP = 300e-6; %[m]
     
            vf = 2.03e6 # [m/s] -> fermi velocity (Ashcroft and Mermin)
            Tc = 1.14 # [K] -> superconducting transition temperature (Ashcroft and Mermin)
            Eg = 1.76*k*Tc # [J] -> cooper pair potential energy CHECK THIS FORMULA! 
            tau0 =438e-9 # [s]
     
            # tau_al_debeye=3.5e-12 # from kozorezov prb vol 61 num 17 pg 11807
     
            # Debeye Temperature
            Td = 392 # [K] -> Ashcroft and Mermin
            vp_avg = 5100 #[m/s]
     
            # this is the average time to break a cooper pair with a debye
            # frequency phonon:
            tau_ep_debeye= 3.5e-12 # [s] %from kozorezov prb vol 61 num 17 pg 11807
     
            # Resistivity
            RRR = 11 #  -> Our iZIPs have an RRR of 11
            #prop.rho_300K = 2.74e-8;%[Ohm m]  http://journals.aps.org/prb/pdf/10.1103/PhysRevB.3.1941
            rho_300K = 27.33e-9 # [Ohm m]  http://journals.jps.jp/doi/pdf/10.1143/JPSJ.66.1253
            rho = rho_300K / RRR
     
            #number of charge carriers:
            n_e = 18.1e28 # [e/m^3] -> Ashcroft and Mermin
     
            # mass density
            rhoD_m = 2.70e3 #[kg/m^3] -> wikipedia
   
        # Nuclear Number
        self._A = A
        # Atomic Number
        self._Z = Z
        # Efficiency phonons --> quasiparticles
        self._ePQP = ePQP
        self._Tc = Tc
