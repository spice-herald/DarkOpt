from darkopt.core import TES, QET, Detector, Absorber
from darkopt.materials._MaterialProperties import TESMaterial, DetectorMaterial, QETMaterial
import numpy as np
import scipy.constants as constants
import qetpy as qp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize

#import matplotlib.colors as mcolors
from matplotlib import cm  

def _loss_func(params, n_fin, absorber, tes, qet, det, per_Al=None, rtnDet=False, fix_w_overlap=True):
    """
    Helper function to define the loss function to 
    minimize to optimize the detector parameters
    """
    
    if fix_w_overlap:
        l, l_overlap, l_fin = params
        w_overlap = tes.w_overlap
    else:
        l, l_overlap, l_fin, w_overlap = params


    abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                    height=absorber._h, width=absorber._width,
                    w_safety=absorber._w_safety)
    
    tes1 = TES(length=l, width=tes.w, l_overlap=l_overlap, n_fin=n_fin, sigma=tes.sigma,
               rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload, 
               w_overlap=w_overlap, w_fin_con=tes.w_fin_con, h=tes.h, 
               veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
               con_type=tes.con_type, material=tes.material, operating_point=tes.fOp,
               alpha=tes.alpha, beta=tes.beta, wempty_fin=tes.wempty_fin, 
               wempty_tes=tes.wempty_tes, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
               w_overlap_stem=tes.w_overlap_stem,  l_c=tes.l_c, 
               l_overlap_pre_ellipse=tes.l_overlap_pre_ellipse)
    
    qet1 = QET(l_fin=l_fin, h_fin=qet.h_fin, TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
               eff_absb=qet.eff_absb, nhole_per_fin=qet.nhole_per_fin, 
               type_qp_eff=qet.type_qp_eff)
    
    det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                    w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                    freqs=det.freqs, equal_spaced=det.equal_spaced )
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
    



def optimize_detector(tes_length0, tes_l_overlap0, l_fin0, n_fin0, per_Al, rn,
                      abs_type, abs_shape, abs_height, abs_width, w_safety, sigma, 
                      rp, L_tot,  ahole, tes_width=2.5e-6, h_fin=600e-9, n_channel=1,
                      rsh=5e-3, tload=30e-3, w_overlap=None, w_fin_con=2.5e-6, 
                      tes_h=40e-9, veff_WAloverlap=0.45, nhole_per_fin=3,
                      veff_WFinCon=0.88, con_type='ellipse', material=TESMaterial(), 
                      operating_point=0.45, alpha=None, beta=0,  n=5, Qp=0, 
                      t_mc=10e-3, ePQP=0.52, eff_absb = 1.22e-4, wempty_fin=6e-6, 
                      wempty_tes=6e-6, type_qp_eff=0, freqs=None, w_rail_main=6e-6, 
                      w_railQET=4e-6, bonding_pad_area=4.5e-8, l_c=5e-6, 
                      w_overlap_stem=4e-6,  l_overlap_pre_ellipse=2e-6,
                      bounds = [[10e-6, 300e-6], 
                               [5e-6, 50e-6],
                               [20e-6, 300e-6],  
                               [2, 8] ],
                      fix_w_overlap=True,
                      w_overlap_bounds = [4e-6, 50e-6], 
                      equal_spaced=True, 
                      verbose=True):
    """
    Function to minimize the energy resolution of a detector object. The following
    parameters are DOF: 
            TES length
            TES overlap length
            Al fin length
            Al fin height
            number of fins
    Rn can be made a free parameter buy changing the variable fix_w_overlap=False
    
    Note: if the overlap bounds are set too low, you will start to get unphysical
    results. recommended that the lower bound be ~5um. 
    
    Parameters:
    -----------
    tes_length0 : float
        guess for length of TES in [m]
    tes_l_overlap0 : float
        guess for lenght of Al/W overlap region in [m]
    l_fin0 : float, 
        guess for Length of Al fins [m]
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
    tes_width : float, optional
        width of TES in [m]
    h_fin : float, 
        Hight of Al fins [m]
    n_channel : int, optional
        Number of chennels in detector 
    tload : float, optional
        The effective noise temperature for the passive johnson 
        noise from the shunt resistor and parasitic resistance
        in [Ohms]
    w_overlap : float, optional
        Width of the W/Al overlap region (if None, a rough estimate is used
        for area calulations) [m]
    w_fin_con : float, optional
        Width of the W only part of the fin connector. Defaults
        to the standard width of the TES of 2.5e-6. [m]
    tes_h : float
        thickness of TES in [m]
    veff_WAloverlap : float, optional
        eff factor for the contribution of the W/Al overlap 
        region to the volume and heat capacity (0,1)
    veff_WFinCon : float, optional
        eff factor for the contribution of the W only part 
        of the fin connector to the volume and heat capacity (0,1)
    con_type : string, optional
        Either 'ellipse' or 'modern' fin connector style
    material : Material Object, optional
        Object containing W properties
    operating_point : float, optional
        The operational resistance [%Rn]
        (fractional percentage of normal resistance)
    nhole_per_fin : int, optional,
            Number of holes per Al fin
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
    wempty_fin : float, optional
        ? width of empty slot in the fin? [m]
    wempty_tes : float, optional
        ? width of empty space between TES and Al [m]
    type_qp_eff : int, optional
        how the efficiency should be calculated.
        0 : 'modern' estimate of overlap radius (small)
        1 : 'modern' estimate, but different effective l_overlap
        2 : Matt's method
    freqs : array-like
        frequencies used for plotting noise and 
        to calculate the expected energy resolution
    bounds : nested list, optional
        The upper and lower bounds for the free parameters
    w_rail_main : float, optional
        The width of the main bias rials. By
        default this is set to 6e-6 [m]
    w_railQET : float, optional
        The width of the secondary bias lines
        connecting the the QETs to the main rails.
        by default it is set to 4e-6. [m]
    bonding_pad_area : float, optional
        The area of passive Al used for the total
        number of bonding pads needed. The default
        is set to the area of 2 150um by 150um pads. [m^2]
    fix_w_overlap : Bool, optional
        If True, w_overlap is not a free parameter, if False
        then it is allowed to vary.
    w_overlap_bounds : list, array, optional
        Lower and upper bounds for w_overlap if it is an 
        optimization param
    l_c : float, optional
        The length of the fin connector before it widens
        to connect to the Al
    w_overlap_stem : float, optional
        The wider part of the conector at the Al [m]
    l_overlap_pre_ellipse : float, optional
        The length of the rectangular part of the overlap
        region before the half ellipse [m]
    equal_spaced : bool, optional
            If True, the QETs are spread out evenly 
            accross the instrumented surface area
            of the detector. If False, the QETs
            are spread out equally in one dimension, 
            but not in the other. (ie, the secondary
            bias rails are not used so a sparse design
            can still be close packed) 
    verbose : bool, optional
        If True, the optimum params are printed
    """
    
    absorb = Absorber(name=abs_type, shape=abs_shape, 
                      height=abs_height, width=abs_width, 
                      w_safety=w_safety)
    
    tes = TES(length=tes_length0, width=tes_width, l_overlap=tes_l_overlap0, 
              n_fin=n_fin0, sigma=sigma, rn=rn, rp=rp, L_tot=L_tot, rsh=rsh, 
              tload=tload, w_overlap=w_overlap,w_fin_con=w_fin_con, wempty_fin=wempty_fin,
              wempty_tes=wempty_tes, h=tes_h, veff_WAloverlap=veff_WAloverlap, 
              veff_WFinCon=veff_WFinCon, con_type=con_type, material=material, 
              operating_point=operating_point, alpha=alpha, beta=beta, n=n, 
              Qp=Qp, t_mc=t_mc, l_c=l_c, w_overlap_stem=w_overlap_stem,  
              l_overlap_pre_ellipse=l_overlap_pre_ellipse)
    tes.w_overlap = w_overlap #forces it to be 'circle' if this is chosen
    
    qet = QET(l_fin=l_fin0, h_fin=h_fin, TES=tes, ahole=ahole, ePQP=ePQP,
              eff_absb=eff_absb, nhole_per_fin=nhole_per_fin,  
              type_qp_eff=type_qp_eff)
    
    det = Detector(absorber=absorb, QET=qet, w_rail_main=w_rail_main, 
                   w_railQET=w_railQET, bonding_pad_area=bonding_pad_area, 
                   n_channel=n_channel, freqs=freqs, passive=1, 
                   equal_spaced=equal_spaced)

    if fix_w_overlap:
        x0 = np.array([tes_length0, tes_l_overlap0, l_fin0])
        bnds = bounds
    else:
        x0 = np.array([tes_length0, tes_l_overlap0, l_fin0, w_overlap])
        bnds = bounds.copy()
        bnds.append(w_overlap_bounds)
    res = minimize(_loss_func, x0, args=(n_fin0, absorb, tes, qet, det, per_Al, False, fix_w_overlap), bounds=bnds )
    det1 = _loss_func(res['x'],n_fin0, absorb, tes, qet, det, None, True, fix_w_overlap)
    
    if verbose:
        print(f"resolution: {det1.calc_res()*1e3:.1f} [meV]")
        print(f"TES Length = {res['x'][0]*1e6:.1f} [μm]")
        print(f"Overlap Legth = {res['x'][1]*1e6:.1f} [μm]")
        if det1.QET.TES.w_overlap is not None:
            print(f"Overlap Width = {det1.QET.TES.w_overlap*1e6:.1f} [μm]")
        print(f"Fin Length = {res['x'][2]*1e6:.1f} [μm]")
        print(f"Fin Height = {det1.QET.h_fin*1e6:.1f} [μm]")
        print(f"N Fins = {int(res['x'][3])}")
        print(f'Total Al surface coverage = {det1._fSA_qpabsorb*100:.3f} [%]')
        print(f'percent active Al = {det1.fSA_active*100:.3f} [%]')
        print(f'percent passive Al = {det1.fSA_passive*100:.3f} [%]')
        print(f'TES response time τ-  = {det1.QET.TES.taup_m*1e6:.2f} [μs]')
        print(f'Phonon collection time constant = {det1._t_pabsb*1e6:.2f} [μs]')
        print(f'Absolute phonon collection energy efficiency = {det1._eEabsb*100:.2f} [%]')
        print(f'Number of TESs = {det1.QET.TES.nTES}')
        print(f'Rn = {det1.QET.TES.rn*1e3:.1f} [mOhms]')
        if det1.equal_spaced:
            print(f'Close Packed: {det1._close_packed}')
        else:
            print(f'QETs are NOT equally spaced on surface')


        if det1.QET.TES.is_phase_sep:
            print('Design is phase separated')
        else:     
            phase_margin = (det1.QET.TES.max_phase_length - det1.QET.TES.l)/det1.QET.TES.l
            print(f'Phase margin = {phase_margin*100:.1f} [%] (phase_sep_length  - tes_length )/tes_length )')
            print('---------------------------------\n\n')
    
    return det1, res['fun'], res['x'], res
    