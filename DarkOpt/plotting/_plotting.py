from darkopt.core import TES, QET, Absorber, Detector
from darkopt.materials._MaterialProperties import TESMaterial, DetectorMaterial, QETMaterial
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



def plot_ltes_vs_lfin(l_tes, l_fin, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_tes*1e6, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(l_fin*1e6, l_tes*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_tes*1e6, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(l_fin*1e6, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_tes*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(l_fin*1e6, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_tes*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(l_fin*1e6, l_tes*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_tes*1e6, res*100, 
                        levels=ncontours, colors=contour_cmap)
        
    plt.plot(l_f_opt*1e6, l_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(l_f_opt*1e6, l_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Length [μm]")
    ax.set_ylabel('TES Length [μm]')
    
    return fig, ax
    

def plot_ltes_vs_hfin(l_tes, h_fin, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_tes*1e6, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(h_fin*1e9, l_tes*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_tes*1e6, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(h_fin*1e9, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_tes*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(h_fin*1e9, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_tes*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(h_fin*1e6, l_tes*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_tes*1e6, res*100, 
                        levels=ncontours, colors=contour_cmap)
        
    plt.plot(h_f_opt*1e9, l_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(h_f_opt*1e9, l_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Thickness [nm]")
    ax.set_ylabel('TES Length [μm]')
    
    return fig, ax
    
def plot_ltes_vs_loverlap(l_tes, l_overlap, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(l_overlap*1e6, l_tes*1e6, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(l_overlap*1e6, l_tes*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(l_overlap*1e6, l_tes*1e6, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(l_overlap*1e6, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(l_overlap*1e6, l_tes*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(l_overlap*1e6, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(l_overlap*1e6, l_tes*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(l_overlap*1e6, l_tes*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(l_overlap*1e6, l_tes*1e6, res*100, 
                        levels=ncontours, colors=contour_cmap)

        
    plt.plot(l_overlap_opt*1e6, l_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(l_overlap_opt*1e6, l_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("W/Al Overlap Length [μm]")
    ax.set_ylabel('TES Length [μm]')
    
    return fig, ax
    
    
def plot_loverlap_vs_lfin(l_overlap, l_fin, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_overlap*1e6, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(l_fin*1e6, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_overlap*1e6, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(l_fin*1e6, l_overlap*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_overlap*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(l_fin*1e6, l_overlap*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_overlap*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(l_fin*1e6, l_overlap*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(l_fin*1e6, l_overlap*1e6, res*100, 
                        levels=ncontours, colors=contour_cmap)

        
    plt.plot(l_fin_opt*1e6, l_overlap_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(l_fin_opt*1e6, l_overlap_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Length [μm]")
    ax.set_ylabel("W/Al Overlap Length [μm]")
    
    return fig, ax
    
    
def plot_hfin_vs_lfin(h_fin, l_fin, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(l_fin*1e6, h_fin*1e9, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(l_fin*1e6, h_fin*1e9, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(l_fin*1e6, h_fin*1e9, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(l_fin*1e6, h_fin*1e9, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(l_fin*1e6, h_fin*1e9, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(l_fin*1e6, h_fin*1e9, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(l_fin*1e6, h_fin*1e9, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(l_fin*1e6, h_fin*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(l_fin*1e6, h_fin*1e9, res*100, 
                        levels=ncontours, colors=contour_cmap)

        
    plt.plot(l_fin_opt*1e6, h_fin_opt*1e9, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(l_fin_opt*1e6, h_fin_opt*1e9, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Length [μm]")
    ax.set_ylabel("Al Fin Thickness [nm]")
    
    return fig, ax
    
    
    
def plot_loverlap_vs_hfin(l_overlap, h_fin, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_overlap*1e6, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(h_fin*1e9, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_overlap*1e6, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(h_fin*1e9, l_overlap*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_overlap*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(h_fin*1e9, l_overlap*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_overlap*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(h_fin*1e6, l_overlap*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(h_fin*1e9, l_overlap*1e6, res*100, 
                        levels=ncontours, colors=contour_cmap)

        
    plt.plot(h_fin_opt*1e9, l_overlap_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(h_fin_opt*1e9, l_overlap_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel("Al Fin Thickness [nm]")
    ax.set_ylabel("W/Al Overlap Length [μm]")
    
    return fig, ax


def plot_ltes_vs_tc(l_tes, tc, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(tc*1e3, l_tes*1e6, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(tc*1e3, l_tes*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(tc*1e3, l_tes*1e6, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(tc*1e3, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(tc*1e3, l_tes*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(tc*1e3, l_tes*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(tc*1e3, l_tes*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(tc*1e3, l_tes*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(tc*1e3, l_tes*1e6, res*100, 
                        levels=ncontours, colors=contour_cmap)
        
    plt.plot(tc_opt*1e3, l_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(tc_opt*1e3, l_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel(r"$T_c$ [mK]")
    ax.set_ylabel('TES Length [μm]')
    
    return fig, ax


def plot_lfin_vs_tc(l_fin, tc, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(tc*1e3, l_fin*1e6, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(tc*1e3, l_fin*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(tc*1e3, l_fin*1e6, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(tc*1e3, l_fin*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(tc*1e3, l_fin*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(tc*1e3, l_fin*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(tc*1e3, l_fin*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(tc*1e3, l_fin*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(tc*1e3, l_fin*1e6, res*100, 
                        levels=ncontours, colors=contour_cmap)
        
    plt.plot(tc_opt*1e3, l_f_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(tc_opt*1e3, l_f_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel(r"$T_c$ [mK]")
    ax.set_ylabel('Al Fin Length [μm]')
    
    return fig, ax


def plot_hfin_vs_tc(h_fin, tc, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(tc*1e3, h_fin*1e9, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(tc*1e3, h_fin*1e9, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(tc*1e3, h_fin*1e9, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(tc*1e3, h_fin*1e9, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(tc*1e3, h_fin*1e9, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(tc*1e3, h_fin*1e9, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(tc*1e3, h_fin*1e9, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(tc*1e3, h_fin*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(tc*1e3, h_fin*1e9, res*100, 
                        levels=ncontours, colors=contour_cmap)
        
    plt.plot(tc_opt*1e3, h_f_opt*1e9, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(tc_opt*1e3, h_f_opt*1e9, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel(r"$T_c$ [mK]")
    ax.set_ylabel('Al Fin Thickness [nm]')
    
    return fig, ax



def plot_loverlap_vs_tc(l_overlap, tc, det, val='energy', figsize=(6.75, 4.455),
                      ncontours=None, contour_cmap='black'):
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
    ncontours : int, nonetype, optional,
        If not none, countours are drawn over the heatmap
    contour_cmap : str, optional
        matplotlib color map
        
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
            det1 = Detector(abso1, qet1, n_channel=det._n_channel, w_rail_main=det.w_rail_main, 
                            w_railQET=det.w_railQET, bonding_pad_area=det.bonding_pad_area,
                            freqs=det.freqs , equal_spaced=det.equal_spaced)
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
        if ncontours is not None:
            plt.contour(tc*1e3, l_overlap*1e6, res*1e3, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'eff':
        plt.pcolor(tc*1e3, l_overlap*1e6, res, cmap='plasma')
        plt.colorbar(label='Total Phonon \nCollection Efficiency')
        if ncontours is not None:
            plt.contour(tc*1e3, l_overlap*1e6, res, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_etf':
        plt.pcolor(tc*1e3, l_overlap*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{ETF}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(tc*1e3, l_overlap*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'tau_ph':
        plt.pcolor(tc*1e3, l_overlap*1e6, res*1e6, cmap='plasma')
        plt.colorbar(label=r'$\tau_{\mathrm{phonon}}\, [\mu\mathrm{s}]$')
        if ncontours is not None:
            plt.contour(tc*1e3, l_overlap*1e6, res*1e6, 
                        levels=ncontours, colors=contour_cmap)
    elif val == 'al':
        plt.pcolor(tc*1e3, l_overlap*1e6, res*100, cmap='plasma')
        plt.colorbar(label='Al Surface Coverage [%]')
        if ncontours is not None:
            plt.contour(tc*1e3, l_overlap*1e6, res*100, 
                        levels=ncontours, colors=contour_cmap)
        
    plt.plot(tc_opt*1e3, l_overlap_opt*1e6, linestyle=' ', marker='+', color='k',
            zorder=10000, ms='8')
    plt.plot(tc_opt*1e3, l_overlap_opt*1e6, linestyle=' ', marker='x', color='k',
            zorder=10000, ms='8')
    ax.set_xlabel(r"$T_c$ [mK]")
    ax.set_ylabel('W/Al Overlap [μm]')
    
    return fig, ax






    
    
    
    