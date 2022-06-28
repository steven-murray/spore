import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np


def plot_2D_PS(ps, kperp, kpar, kperp_logspace=False, kpar_logspace=False,
               make_label=True, fig=None, ax=None, interp=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if ps.shape != (len(kpar), len(kperp)):
        raise ValueError("Shape of PS must be (kpar, kperp)")

    PS = getattr(ps, "value", ps)
    KPERP = getattr(kperp, "value", kperp)
    KPAR = getattr(kpar, "value", kpar)

    if kperp_logspace:
        KPERP = np.log10(kperp)
    if kpar_logspace:
        KPAR = np.log10(kpar)


    cax = ax.imshow(PS, origin="lower", norm=LogNorm(), aspect='auto',
                    extent=(KPERP.min(), KPERP.max(), KPAR.min(), KPAR.max()),
                    interpolation=interp)

    if not kperp_logspace:
        ax.set_xscale('log')
    if not kpar_logspace:
        ax.set_yscale('log')

    cbar = fig.colorbar(cax)

    if make_label:
        cbar.ax.set_ylabel(r"Power, $[{\rm mK}^2 h^{-3}{\rm Mpc}^3]$", fontsize=13)

        if not kperp_logspace:
            plt.xlabel(r"$k_{\perp}$, $[h{\rm Mpc}^{-1}]$", fontsize=15)
        else:
            plt.xlabel(r"$\log_{10} k_{\perp}$, $[h{\rm Mpc}^{-1}]$", fontsize=15)
        if not kpar_logspace:
            plt.ylabel(r"$k_{||}$, $[h{\rm Mpc}^{-1}]$", fontsize=15)
        else:
            plt.ylabel(r"$\log_{10} k_{||}$, $[h{\rm Mpc}^{-1}]$", fontsize=15)

    plt.ylim(KPAR.min(),KPAR.max())
    plt.xlim(KPERP.min(),KPERP.max())

    return fig, ax, cbar


def plot_2D_PS_compare(ps_list, kperp, kpar, plt_labels=None, interp=None):
    fig, ax = plt.subplots(len(ps_list[0]), len(ps_list),
                           figsize=(5.5*len(ps_list), 5*len(ps_list[0])),
                           sharex=True, sharey=True,
                           subplot_kw={"xscale": 'log', "yscale": 'log'},
                           gridspec_kw={"hspace": 0.05, "wspace": 0.05},
                           squeeze=False)

    KPERP = getattr(kperp, "value", kperp)
    KPAR = getattr(kpar, "value", kpar)

    cax = []
    for i, sublist in enumerate(ps_list):
        cax.append([])
        for j, ps in enumerate(sublist):
            thisim = ax[j, i].imshow(getattr(ps,"value",ps), origin="lower",
                                     norm=LogNorm(vmin=np.min(ps_list), vmax=np.max(ps_list)),
                                     aspect='auto',
                                     extent=(KPERP.min(), KPERP.max(), KPAR.min(), KPAR.max()),
                                     interpolation=interp)
            cax[i].append(thisim)

    colax = fig.colorbar(cax[0][0], ax=ax.ravel().tolist())
    colax.ax.set_ylabel(r"$[{\rm mK}^2 h^{-3}{\rm Mpc}^3]$")

    for x in ax.T:
        x[-1].set_xlabel(r"$k_{\perp}$, $[h{\rm Mpc}^{-1}]$")

    for x in ax:
        x[0].set_ylabel(r"$k_{||}$, $[h{\rm Mpc}^{-1}]$")

    if plt_labels is not None:
        for i, sublist in enumerate(plt_labels):
            for j, label in enumerate(sublist):
                ax[j, i].text(0.075, 0.9, label, transform=ax[j, i].transAxes, color='white',
                              fontweight='bold')
    return fig, ax


def plot_sig_to_noise_compare(ratio_list, kperp, kpar, plt_labels=None, interp=None, kpar_logscale=False,
                              kperp_logscale=False):
    fig, ax = plt.subplots(len(ratio_list[0]), len(ratio_list),
                           figsize=(5.5*len(ratio_list), 5*len(ratio_list[0])),
                           sharex=True, sharey=True,
                           subplot_kw={"xscale": 'log' if not kperp_logscale else None,
                                       "yscale": 'log' if not kpar_logscale else None},
                           gridspec_kw={"hspace": 0.05, "wspace": 0.05},
                           squeeze=False)

    KPERP = getattr(kperp, "value", kperp)
    KPAR = getattr(kpar, "value", kpar)

    if kperp_logscale:
        KPERP = np.log10(KPERP)
    if kpar_logscale:
        KPAR = np.log10(KPAR)

    colscale = np.nanmax(np.abs(np.log10(ratio_list)))
    cax = []
    for i, sublist in enumerate(ratio_list):
        cax.append([])
        for j, ps in enumerate(sublist):
            thisim = ax[j, i].imshow(getattr(ps,"value",ps), origin="lower",
                                     norm=LogNorm(vmin=10**-colscale, vmax=10**colscale),
                                     aspect='auto',
                                     extent=(KPERP.min(), KPERP.max(), KPAR.min(), KPAR.max()),
                                     interpolation=interp,
                                     cmap='coolwarm')
            cax[i].append(thisim)

    colax = fig.colorbar(cax[0][0], ax=ax.ravel().tolist())
    colax.ax.set_ylabel(r"Signal to Noise")

    for x in ax.T:
        if not kperp_logscale:
            x[-1].set_xlabel(r"$k_{\perp}$, $[h{\rm Mpc}^{-1}]$")
        else:
            x[-1].set_xlabel(r"$\log_{10} k_{\perp}$, $[h{\rm Mpc}^{-1}]$")

    for x in ax:
        if not kpar_logscale:
            x[0].set_ylabel(r"$k_{||}$, $[h{\rm Mpc}^{-1}]$")
        else:
            x[0].set_ylabel(r"$\log_{10} k_{||}$, $[h{\rm Mpc}^{-1}]$")

    if plt_labels is not None:
        for i, sublist in enumerate(plt_labels):
            for j, label in enumerate(sublist):
                ax[j, i].text(0.075, 0.9, label, transform=ax[j, i].transAxes, color='white',
                              fontweight='bold')
    return fig, ax


def plot_2D_PS_ratio_diff(ps1, ps2, kperp, kpar, make_label=True, fig=None, ax=None, interp=None, lognorm=True):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(8,5), sharex=True, sharey=True,
                               subplot_kw={"xscale": 'log', "yscale": 'log'})

    if lognorm:
        norm = LogNorm(vmin=1)
    else:
        norm = None

    PS1 = getattr(ps1, "value", ps1)
    PS2 = getattr(ps2, "value", ps2)

    KPERP = getattr(kperp, "value", kperp)
    KPAR = getattr(kpar, "value", kpar)

    cax1 = ax[0].imshow(PS1/PS2, origin="lower", norm=norm,
                        aspect='auto', extent=(KPERP.min(), KPERP.max(), KPAR.min(), KPAR.max()),
                        interpolation=interp)
    cax2 = ax[1].imshow(PS1 - PS2, origin="lower", norm=LogNorm(np.abs(PS1 - PS2).min()),
                        aspect='auto', extent=(KPERP.min(), KPERP.max(), KPAR.min(), KPAR.max()),
                        interpolation=interp)

    cbar1 = plt.colorbar(cax1, ax=ax[0])
    cbar2 = plt.colorbar(cax2, ax=ax[1])


    if make_label:
        ax[0].text(0.5, 0.93, "Ratio", transform=ax[0].transAxes, fontweight='bold', color='white',
                   ha='center')
        ax[1].text(0.5, 0.93, "Diff", transform=ax[1].transAxes, fontweight='bold', color='white',
                   ha="center")

        # colax1.set_ylabel(r"Power Ratio", fontsize=13)
        # colax1.yaxis.set_label_position("right")

        cbar2.ax.set_ylabel(r"${\rm mK}^2 h^{-3}{\rm Mpc}^3$")

        ax[0].set_xlabel(r"$k_{\perp}$, $[h{\rm Mpc}^{-1}]$")
        ax[1].set_xlabel(r"$k_{\perp}$, $[h{\rm Mpc}^{-1}]$")
        ax[0].set_ylabel(r"$k_{||}$, $[h{\rm Mpc}^{-1}]$")

    plt.tight_layout()
    return fig, ax


def plot_2D_PS_frac_tot(ps1, ps2, kperp, kpar, make_label=True, fig=None, ax=None, interp=None, lognorm=True,):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True,
                               subplot_kw={"xscale": 'log', "yscale": 'log'})

    if lognorm:
        norm = LogNorm(vmin=1)
    else:
        norm = None

    PS1 = getattr(ps1, "value", ps1)
    PS2 = getattr(ps2, "value", ps2)

    KPERP = getattr(kperp, "value", kperp)
    KPAR = getattr(kpar, "value", kpar)

    cax1 = ax[0].imshow(PS1/PS2, origin="lower", #norm=norm,
                        aspect='auto', extent=(KPERP.min(), KPERP.max(), KPAR.min(), KPAR.max()),
                        interpolation=interp)
    cax2 = ax[1].imshow(PS2, origin="lower", norm=LogNorm(),
                        aspect='auto', extent=(KPERP.min(), KPERP.max(), KPAR.min(), KPAR.max()),
                        interpolation=interp)

    cbar1 = fig.colorbar(cax2, ax=ax[0])
    cbar2 = fig.colorbar(cax1, ax=ax[1])

    if make_label:
        ax[0].text(0.5, 0.93, "Clustering Fraction", transform=ax[0].transAxes, fontweight='bold', color='white',
                   ha='center')
        ax[1].text(0.5, 0.93, "Total Power", transform=ax[1].transAxes, fontweight='bold', color='white',
                   ha="center")

        # colax1.set_ylabel(r"Power Ratio", fontsize=13)
        # colax1.yaxis.set_label_position("right")

        cbar2.ax.set_ylabel(r"${\rm mK}^2 h^{-3}{\rm Mpc}^3$")
        cbar1.ax.set_ylabel("Ratio")

        ax[0].set_xlabel(r"$k_{\perp}$, $[h{\rm Mpc}^{-1}]$")
        ax[1].set_xlabel(r"$k_{\perp}$, $[h{\rm Mpc}^{-1}]$")
        ax[0].set_ylabel(r"$k_{||}$, $[h{\rm Mpc}^{-1}]$")

    plt.tight_layout()
    return fig, ax