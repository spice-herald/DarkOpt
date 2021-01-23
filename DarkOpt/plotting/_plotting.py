from darkopt.core import TES, QET, Absorber, Detector
from darkopt.materials._MaterialProperties import TESMaterial, DetectorMaterial, QETMaterial
from darkopt.utils._utils import _line, arc_patch, calc_angles
import numpy as np
import scipy.constants as constants
import qetpy as qp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm  
from matplotlib import patches


nice_fonts = {
        # Use LaTeX to write all text
        #"text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 14,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
}

rcParams.update(nice_fonts)



def plot_ltes_vs_lfin(l_tes, l_fin, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    l_tes : array,
        array of tes lengths to calculate energy resolution
    l_fin :array
        array of Al fin lengths to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    l_opt = tes.l
    l_f_opt = qet.l_fin
    
    res = np.ones((len(l_tes), len(l_fin)))
    for ii in range(len(l_tes)):
        for jj in range(len(l_fin)):
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=l_tes[ii], width=tes.w, l_overlap=tes.l_overlap, n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=tes.material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                     wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=l_fin[jj], h_fin=qet.h_fin, TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(l_fin*1e6, l_tes*1e6, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(l_fin*1e6, l_tes*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(l_fin*1e6, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(l_fin*1e6, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(l_fin*1e6, l_tes*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        
    plt.plot(l_f_opt*1e6, l_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(l_f_opt*1e6, l_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Length [μm]")
    ax.set_ylabel('TES Length [μm]')
    
    return fig, ax
    

def plot_ltes_vs_hfin(l_tes, h_fin, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    l_tes : array,
        array of tes lengths to calculate energy resolution
    h_fin :array
        array of Al fin thickness to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    l_opt = tes.l
    h_f_opt = qet.h_fin
    
    res = np.ones((len(l_tes), len(h_fin)))
    for ii in range(len(l_tes)):
        for jj in range(len(h_fin)):
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=l_tes[ii], width=tes.w, l_overlap=tes.l_overlap, n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=tes.material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                      wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=qet.l_fin, h_fin=h_fin[jj], TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(h_fin*1e9, l_tes*1e6, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(h_fin*1e9, l_tes*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(h_fin*1e9, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(h_fin*1e9, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(h_fin*1e6, l_tes*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        
    plt.plot(h_f_opt*1e9, l_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(h_f_opt*1e9, l_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Thickness [nm]")
    ax.set_ylabel('TES Length [μm]')
    
    return fig, ax
    
def plot_ltes_vs_loverlap(l_tes, l_overlap, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    l_tes : array,
        array of tes lengths to calculate energy resolution
    l_overlap : array
        array of W/Al overlap lengths to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    l_opt = tes.l
    l_overlap_opt = tes.l_overlap
    
    res = np.ones((len(l_tes), len(l_overlap)))
    for ii in range(len(l_tes)):
        for jj in range(len(l_overlap)):
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=l_tes[ii], width=tes.w, l_overlap=l_overlap[jj], n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=tes.material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                      wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=qet.l_fin, h_fin=qet.h_fin, TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(l_overlap*1e6, l_tes*1e6, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(l_overlap*1e6, l_tes*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(l_overlap*1e6, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(l_overlap*1e6, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(l_overlap*1e6, l_tes*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')

        
    plt.plot(l_overlap_opt*1e6, l_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(l_overlap_opt*1e6, l_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("W/Al Overlap Length [μm]")
    ax.set_ylabel('TES Length [μm]')
    
    return fig, ax
    
    
def plot_loverlap_vs_lfin(l_overlap, l_fin, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    l_overlap : array
        array of W/Al overlap lengths to calculate energy resolution
    l_fin :array
        array of Al fin lengths to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    l_fin_opt = qet.l_fin
    l_overlap_opt = tes.l_overlap
    
    res = np.ones((len(l_overlap), len(l_fin)))
    for ii in range(len(l_overlap)):
        for jj in range(len(l_fin)):
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=tes.l, width=tes.w, l_overlap=l_overlap[ii], n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=tes.material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                      wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=l_fin[jj], h_fin=qet.h_fin, TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(l_fin*1e6, l_overlap*1e6, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(l_fin*1e6, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(l_fin*1e6, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(l_fin*1e6, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(l_fin*1e6, l_overlap*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')

        
    plt.plot(l_fin_opt*1e6, l_overlap_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(l_fin_opt*1e6, l_overlap_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Length [μm]")
    ax.set_ylabel("W/Al Overlap Length [μm]")
    
    return fig, ax
    
    
def plot_hfin_vs_lfin(h_fin, l_fin, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    h_fin : array
        array of AL fin thicknesses to calculate energy resolution
    l_fin :array
        array of Al fin lengths to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    l_fin_opt = qet.l_fin
    h_fin_opt = qet.h_fin
    
    res = np.ones((len(h_fin), len(l_fin)))
    for ii in range(len(h_fin)):
        for jj in range(len(l_fin)):
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=tes.l, width=tes.w, l_overlap=tes.l_overlap, n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=tes.material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                      wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=l_fin[jj], h_fin=h_fin[ii], TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(l_fin*1e6, h_fin*1e9, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(l_fin*1e6, h_fin*1e9, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(l_fin*1e6, h_fin*1e9, res, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(l_fin*1e6, h_fin*1e9, res, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(l_fin*1e6, h_fin*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')

        
    plt.plot(l_fin_opt*1e6, h_fin_opt*1e9, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(l_fin_opt*1e6, h_fin_opt*1e9, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Length [μm]")
    ax.set_ylabel("Al Fin Thickness [nm]")
    
    return fig, ax
    
    
    
def plot_loverlap_vs_hfin(l_overlap, h_fin, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    l_overlap : array
        array of W/Al overlap lengths to calculate energy resolution
    h_fin :array
        array of Al fin thicknesses to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    h_fin_opt = qet.h_fin
    l_overlap_opt = tes.l_overlap
    
    res = np.ones((len(l_overlap), len(h_fin)))
    for ii in range(len(l_overlap)):
        for jj in range(len(h_fin)):
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=tes.l, width=tes.w, l_overlap=l_overlap[ii], n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=tes.material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                      wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=qet.l_fin, h_fin=h_fin[jj], TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(h_fin*1e9, l_overlap*1e6, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(h_fin*1e9, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(h_fin*1e9, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(h_fin*1e9, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(h_fin*1e6, l_overlap*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')

        
    plt.plot(h_fin_opt*1e9, l_overlap_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(h_fin_opt*1e9, l_overlap_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Thickness [nm]")
    ax.set_ylabel("W/Al Overlap Length [μm]")
    
    return fig, ax


def plot_ltes_vs_tc(l_tes, tc, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    l_tes : array,
        array of tes lengths to calculate energy resolution
    tc :array
        array of superconducting trasition temperatures
        to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    
    
    l_opt = tes.l
    tc_opt = tes.tc
    
    res = np.ones((len(l_tes), len(tc)))
    for ii in range(len(l_tes)):
        for jj in range(len(tc)):
            material = TESMaterial(Tc=tc[jj])
            
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=l_tes[ii], width=tes.w, l_overlap=tes.l_overlap, n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                      wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=qet.l_fin, h_fin=qet.h_fin, TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(tc*1e3, l_tes*1e6, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(tc*1e3, l_tes*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(tc*1e3, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(tc*1e3, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(tc*1e3, l_tes*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        
    plt.plot(tc_opt*1e3, l_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(tc_opt*1e3, l_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel(r"$T_c$ [mK]")
    ax.set_ylabel('TES Length [μm]')
    
    return fig, ax


def plot_lfin_vs_tc(l_fin, tc, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    l_fin : array,
        array of Al fin lengths to calculate energy resolution
    tc :array
        array of superconducting trasition temperatures
        to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    
    
    l_f_opt = qet.l_fin
    tc_opt = tes.tc
    
    res = np.ones((len(l_fin), len(tc)))
    for ii in range(len(l_fin)):
        for jj in range(len(tc)):
            material = TESMaterial(Tc=tc[jj])
            
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=tes.l, width=tes.w, l_overlap=tes.l_overlap, n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                      wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=l_fin[ii], h_fin=qet.h_fin, TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(tc*1e3, l_fin*1e6, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(tc*1e3, l_fin*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(tc*1e3, l_fin*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(tc*1e3, l_fin*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(tc*1e3, l_fin*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        
    plt.plot(tc_opt*1e3, l_f_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(tc_opt*1e3, l_f_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel(r"$T_c$ [mK]")
    ax.set_ylabel('Al Fin Length [μm]')
    
    return fig, ax


def plot_hfin_vs_tc(h_fin, tc, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    l_fin : array,
        array of Al fin thicknesses to calculate energy resolution
    tc :array
        array of superconducting trasition temperatures
        to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    
    
    h_f_opt = qet.h_fin
    tc_opt = tes.tc
    
    res = np.ones((len(h_fin), len(tc)))
    for ii in range(len(h_fin)):
        for jj in range(len(tc)):
            material = TESMaterial(Tc=tc[jj])
            
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=tes.l, width=tes.w, l_overlap=tes.l_overlap, n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                      wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=qet.l_fin, h_fin=h_fin[ii], TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(tc*1e3, h_fin*1e9, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(tc*1e3, h_fin*1e9, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(tc*1e3, h_fin*1e9, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(tc*1e3, h_fin*1e9, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(tc*1e3, h_fin*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        
    plt.plot(tc_opt*1e3, h_f_opt*1e9, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(tc_opt*1e3, h_f_opt*1e9, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel(r"$T_c$ [mK]")
    ax.set_ylabel('Al Fin Thickness [nm]')
    
    return fig, ax



def plot_loverlap_vs_tc(l_overlap, tc, det, val='energy', figsize=(6.75, 4.455)):
    """
    Function to make 2D plot of energy resolution or collection efficiency as a 
    function of TES length as Al Fin length, given fixed parameters defined in 
    other passed arguments.
    
    l_loverlap : array,
        array of W/Al overlap lengths to calculate energy resolution
    tc :array
        array of superconducting trasition temperatures
        to calculate energy resolution
    det : Detector object
        used to set other Detector variables
        (TES, QET, Absorber params)
    val : string, optional
        If 'energy', the energy resolution is calculated, 
        if 'eff', the absolute phonon efficiency is calculated
        if 'tau_etf', the ETF fall time is plotted
        if 'tau_ph', the phonon collection time
        if 'al', Aluminum surface coverage
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    absorber = det._absorber
    qet = det.QET
    tes = det.QET.TES
    
    
    
    l_overlap_opt = tes.l_overlap
    tc_opt = tes.tc
    
    res = np.ones((len(l_overlap), len(tc)))
    for ii in range(len(l_overlap)):
        for jj in range(len(tc)):
            material = TESMaterial(Tc=tc[jj])
            
                  
            abso1 = Absorber(name=absorber._name, shape=absorber._shape,
                            height=absorber._h, width=absorber._width,
                            w_safety=absorber._w_safety)
            tes1 = TES(length=tes.l, width=tes.w, l_overlap=l_overlap[ii], n_fin=tes.n_fin, sigma=tes.sigma,
                     rn=tes.rn, rsh=tes.rsh, rp=tes.rp, L_tot=tes.L, tload=tes.tload,
                     h=tes.h, veff_WAloverlap=tes.veff_WAloverlap, veff_WFinCon=tes.veff_WFinCon, 
                     con_type=tes.con_type, material=material, operating_point=tes.fOp,
                     alpha=tes.alpha, beta=tes.beta, n=tes.n, Qp=tes.Qp, t_mc=tes.t_mc,
                      wempty_fin=tes.wempty_fin, wempty_tes=tes.wempty_tes)
            qet1 = QET(l_fin=qet.l_fin, h_fin=qet.h_fin, TES=tes1, ahole=qet.ahole, ePQP=qet.ePQP,
                       eff_absb=qet.eff_absb, type_qp_eff=qet.type_qp_eff)
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, 
                            freqs=det.freqs)
            if val == 'energy':
                res[ii,jj] = det1.calc_res()
            elif val == 'eff':
                res[ii,jj] = det1._eEabsb
            elif val == 'tau_etf':
                res[ii,jj] = det1.QET.TES.taup_m
            elif val == 'tau_ph':
                res[ii,jj] = det1._t_pabsb
            elif val == 'al':
                res[ii,jj] = det1._fSA_qpabsorb
            else:
                raise ValueError('Specify what to plot with the val argument')
                
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if val == 'energy':
        plt.pcolor(tc*1e3, l_overlap*1e6, res*1e3, cmap='plasma_r')
        plt.colorbar(label=r'$\sigma_E\, [\mathrm{meV}]$')
    elif val == 'eff':
        plt.pcolor(tc*1e3, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
    elif val == 'tau_etf':
        plt.pcolor(tc*1e3, l_overlap*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
    elif val == 'tau_ph':
        plt.pcolor(tc*1e3, l_overlap*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
    elif val == 'al':
        plt.pcolor(tc*1e3, l_overlap*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        
    plt.plot(tc_opt*1e3, l_overlap_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(tc_opt*1e3, l_overlap_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel(r"$T_c$ [mK]")
    ax.set_ylabel('W/Al Overlap [μm]')
    
    return fig, ax



def plot_qet(det, figsize=(6.75, 4.455)):
    """
    Function to plot the QET based on the optimum params. 
    This is only a visual aid and not exact
    
    Parameters:
    -----------
    det : detector object
    
    figsize : tuple, optional
        Size of figure to be drawn
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes object
    """
    
    fig, ax = plt.subplots(figsize=figsize)

    qet = det.QET
    tes = det.QET.TES
    
    b = qet.l_fin + tes.l/2
    a = qet.l_fin + tes.wempty_tes + tes.w/2

    # Al ellipse
    e1 = patches.Ellipse((0, 0), a*2, b*2,
                         angle=0, linewidth=2, fill=True, color='xkcd:blue',
                         zorder=2) 
    # remove white space
    e2 = patches.Rectangle((-(tes.w+tes.wempty_tes*2)/2,-tes.l/2),
                           tes.w+tes.wempty_tes*2, 
                           tes.l,
                           color='xkcd:white', zorder=4)
    # TES line
    e3 = patches.Rectangle((-tes.w/2,-tes.l/2), tes.w, 
                           tes.l,
                           color='xkcd:purple', zorder=4)

    # W fin connector
    x_ = tes.l_c + tes.w/2
    qet_con_L = patches.Rectangle((-x_,-tes.w_fin_con/2), tes.l_c, 
                           tes.w_fin_con,
                           color='xkcd:purple', zorder=4)

    # Wider part of W fin connector
    x_ = tes.l_c + tes.w/2 + tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c  
    y_ = tes.w_overlap_stem/2
    l_ = tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c
    qet_con_L_ = patches.Rectangle((-x_,-y_),  l_  , 
                           tes.w_overlap_stem,
                           color='xkcd:purple', zorder=4)
    # W/Al overlap half ellipse
    arc_L = arc_patch((-x_,0), tes.l_overlap, tes.w_overlap, 90, 270, zorder=4)

    # W fin connector
    qet_con_R = patches.Rectangle((tes.w/2,-tes.w_fin_con/2), tes.l_c, 
                           tes.w_fin_con,
                           color='xkcd:purple', zorder=4)
    # Wider part of W fin connector
    x_ = tes.w/2+ tes.l_c  
    y_ = tes.w_overlap_stem/2
    l_ = tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c
    qet_con_R_ = patches.Rectangle((x_, -y_), l_, 
                           tes.w_overlap_stem,
                           color='xkcd:purple', zorder=4)
    # W/Al overlap half ellipse
    arc_R = arc_patch((x_+l_,0), -tes.l_overlap, tes.w_overlap, 90, 270, zorder=4)


    # W fin connector
    x_ = tes.w_overlap_stem/2
    y_ = tes.l/2
    w_ = tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c
    qet_con_T_ = patches.Rectangle((-x_, y_), tes.w_overlap_stem, 
                           tes.l_overlap_pre_ellipse,
                           color='xkcd:purple', zorder=4)
    # W/Al overlap half ellipse
    arc_T = arc_patch((0,y_+tes.l_overlap_pre_ellipse), tes.w_overlap, tes.l_overlap, 0, 180, zorder=4)
    
    # W fin connector
    x_ = tes.w_overlap_stem/2
    y_ = tes.l/2 + tes.l_overlap_pre_ellipse
    w_ = tes.l_overlap_pre_ellipse + tes.wempty_tes - tes.l_c
    qet_con_B_ = patches.Rectangle((-x_, -y_), tes.w_overlap_stem, 
                           tes.l_overlap_pre_ellipse,
                           color='xkcd:purple', zorder=4)
    # W/Al overlap half ellipse
    arc_B = arc_patch((0,-y_), tes.w_overlap, -tes.l_overlap, 0, 180, zorder=4)

    
    
    ax.add_patch(e1)
    ax.add_patch(e2)
    ax.add_patch(e3)
    ax.add_patch(qet_con_L)
    ax.add_patch(qet_con_L_)
    ax.add_patch(arc_L)
    ax.add_patch(qet_con_R)
    ax.add_patch(qet_con_R_)
    ax.add_patch(arc_R)

    ax.add_patch(qet_con_T_)
    ax.add_patch(arc_T)

    ax.add_patch(qet_con_B_)
    ax.add_patch(arc_B)

    plt.plot(x, line(angle1, x), color='w', linewidth=4)
    plt.plot(-x, line(360-angle1, x), color='w', linewidth=4)
    ax.tick_params(which="both", direction="in", right=True, top=True, zorder=300)
    ax.set_title('Sample QET')

    ax.axis('equal')
    
    return fig, ax


    
    
    
    