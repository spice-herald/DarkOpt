from darkopt._TES import TES
from darkopt._QET import QET
from darkopt._detector import Detector
from darkopt._absorber import Absorber
from darkopt._MaterialProperties import TESMaterial, DetectorMaterial, QETMaterial
import numpy as np
import scipy.constants as constants
import qetpy as qp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize

#import matplotlib.colors as mcolors
from matplotlib import cm  

def _loss_func(params, absorber, tes, qet, det, per_Al=None, rtnDet=False):
    """
    Helper function to define the loss function to 
    minimize to optimize the detector parameters
    """
    
    l, l_overlap, l_fin, h_fin, n_fin = params
    n_fin = int(n_fin)
    abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                    height=absorber._h, width=absorber._width,
                    w_safety=absorber._w_safety)
    tes1 = TES(length=l, width=tes.w, l_overlap=l_overlap, n_fin=n_fin, sigma=tes.sigma,
             rn=tes.n, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
             h=tes.h, zeta_WAl_fin=tes.veff_WAloverlap, zeta_W_fin=tes.veff_WFinCon, 
             con_type=tes.con_type, material=tes.material, operating_point=tes.fOp,
             alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc)
    qet1 = QET(l_fin=l_fin, h_fin=h_fin, TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
               eff_absb=qet.eff_absb, wempty=qet.wempty, wempty_tes=qet.wempty_tes, 
               type_qp_eff=qet.type_qp_eff)
    det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                    freqs=det.freqs)
    if rtnDet:
        return det1
    else:
        # weight the energy resolution by the total surface area minus the target
        # Al surface coverage to find the minimum around the desired coverage
        #return det1.calc_res()*np.abs(det1._fSA_qpabsorb - per_Al)
        if per_Al is None:
            return det1.calc_res()
        else:
            delta = np.abs(det1._fSA_qpabsorb - per_Al)
            return det1.calc_res()*(per_Al + delta)
    



def optimize_detector(tes_length0, tes_l_overlap0, l_fin0, h_fin0, n_fin0, per_Al, rn,
                    abs_type, abs_shape, abs_height, abs_width, w_safety,
                    tes_width, sigma, rp, L_tot,  ahole, n_channel=1,
                    rsh=5e-3, tload=30e-3, tes_h=40e-9, zeta_WAl_fin=0.45, 
                    zeta_W_fin=0.88, con_type='ellipse', material=TESMaterial(), 
                    operating_point=0.45, alpha=None, beta=0,  n=5, Qp=0, 
                    t_mc=10e-3, ePQP=0.52, eff_absb = 1.22e-4, wempty=6e-6, 
                    wempty_tes=7.5e-6, type_qp_eff=0, freqs=None):
    """
    Function to minimize the energy resolution of a detector object. The following
    parameters are DOF: 
            TES length
            TES overlap length
            Al fin length
            Al fin height
            number of fins
    
    Parameters:
    -----------
    tes_length0 : float
        guess for length of TES in [m]
    tes_l_overlap0 : float
        guess for lenght of Al/W overlap region in [m]
    l_fin0 : float, 
        guess for Length of Al fins [m]
    h_fin0 : float, 
        guess for hight of Al fins [m]
    n_fin0 : int
        guess for number of Al fins for QET  
    per_AL : float,
        deired fraction of Aluminum surface coverage
    abs_type : string
        detector substrate type, can be
        either 'Si' or 'Ge'
    abs_shape : string 
        the geometric shape of the 
        absorber 'cylinder', 'square', or 'cube' 
    abs_height : float
        the height of the absorber [m]
    abs_width : float
        If a square or cube, the length is assumed
        to be the same as the width. If the shape
        is a cylinder, then the width
    w_safety : float, 
        Safety margin from edge where TES are not put [m]
    tes_width : float
        width of TES in [m]
    sigma : float
        Electron-phonon coupling constant [W/K^5/m^3]
    rn : float 
        Normal state resistance of channel in [Ohms]
    rsh : float
        The shunt resistance of the TES bias circuit [Ohms] 
    rp : float
        The parasitic resistance on the TES side [Ohms]
    L_tot : float
        total inductance (SQUID input coil + parasitic wire
        inductance) [H]
    ahole : float
        area of holes in fin [m^2]
    n_channel : int, optional
        Number of chennels in detector 
    tload : float, optional
        The effective noise temperature for the passive johnson 
        noise from the shunt resistor and parasitic resistance
        in [Ohms]
    tes_h : float
        thickness of TES in [m]
    zeta_WAl_fin : float, optional
        eff factor for the contribution of the W/Al overlap 
        region to the volume and heat capacity (0,1)
    zeta_W_fin : float, optional
        eff factor for the contribution of the W only part 
        of the fin connector to the volume and heat capacity (0,1)
    con_type : string, optional
        Either 'ellipse' or 'modern' fin connector style
    material : Material Object, optional
        Object containing W properties
    operating_point : float, optional
        The operational resistance [%Rn]
        (fractional percentage of normal resistance)
    alpha : float, optional
        The logarithmic temperature sensitivity.
        If None, it will be estimated.
    beta : float, optional
        The logarithmic current sensitivity.
        If not changed, the small beta (beta=0) 
        approximation will be used.
    n : int, optional
        Thermal powerlaw exponent
    Qp : float, optional
        Parasitic heating [J]
    t_mc : float, optional
        Temperature of the mixing chamber [K]
    
    ePQP : float, optional
        Phonon to QP Conversion Effciency, 
        Kaplan downconversion limits this to 52%
    eff_absb : float, optional
        W/Al transmition/trapping probability
    wempty : float, optional
        ?
    wempty_tes : float, optional
        ?
    type_qp_eff : int, optional
        how the efficiency should be calculated.
        0 : 'modern' estimate of overlap radius (small)
        1 : 'modern' estimate, but different effective l_overlap
        2 : Matt's method
    freqs : array-like
        frequencies used for plotting noise and 
        to calculate the expected energy resolution
        
    """
    
    absorb = Absorber(name=abs_type, shape=abs_shape, 
                      height=abs_height, width=abs_width, 
                      w_safety=w_safety)
    
    tes = TES(length=tes_length0, width=tes_width, l_overlap=tes_l_overlap0, 
              n_fin=n_fin0, sigma=sigma, rn=rn, rp=rp, L_tot=L_tot, rsh=rsh, 
              tload=tload, h=tes_h, zeta_WAl_fin=zeta_WAl_fin, zeta_W_fin=zeta_W_fin,
              con_type=con_type, material=material, operating_point=operating_point,
              alpha=alpha, beta=beta, n=n, Qp=Qp, t_mc=t_mc)
    
    qet = QET(l_fin=l_fin0, h_fin=h_fin0, TES=tes, ahole=ahole, ePQP=ePQP,
              eff_absb=eff_absb, wempty=wempty, wempty_tes=wempty_tes, 
              type_qp_eff=type_qp_eff)
    
    det = Detector(absorber=absorb, QET=qet, passive=1, 
                   n_channel=n_channel, freqs=freqs)
    
    x0 = np.array([tes_length0, tes_l_overlap0, l_fin0, h_fin0, n_fin0])
    bounds = [[50e-6, 300e-6], [5e-6, 50e-6],[50e-6, 300e-6], [300e-6, 1000e-6], [2, 8] ]
    res = minimize(_loss_func, x0, args=(absorb, tes, qet, det, per_Al, False), bounds=bounds )
    det1 = _loss_func(res['x'], absorb, tes, qet, det, None, True)
    
    print(f"resolution: {det1.calc_res()*1e3:.1f} [meV]")
    print(f"TES Length = {res['x'][0]*1e6:.1f} [μm]")
    print(f"Overlap Legth = {res['x'][1]*1e6:.1f} [μm]")
    print(f"Fin Length = {res['x'][2]*1e6:.1f} [μm]")
    print(f"Fin Height = {res['x'][3]*1e6:.1f} [μm]")
    print(f"N Fins = {int(res['x'][4])}")
    print(f'Total Al surface coverage = {det1._fSA_qpabsorb*100:.3} [%]')
    print(f'Number of TESs = {det1.QET.TES.nTES}')
    print(f'Close Packed: {det1._close_packed}')
    
    
    return det1, res['fun'], res['x']
    