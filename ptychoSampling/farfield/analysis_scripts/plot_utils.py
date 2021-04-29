import ptychoSampling.farfield.analysis_scripts.combined_analysis_utils as cbut
import ptychoSampling.farfield.analysis_scripts.analysis_utils as anut
import numpy as np
import scipy
from copy import deepcopy
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
import matplotlib.pyplot as plt
import dataclasses as dt

@dt.dataclass
class PlotData:
    xys: list = dt.field(default_factory=list)
    labels: list = dt.field(default_factory=list)
    lows: list = dt.field(default_factory=list)
    highs: list = dt.field(default_factory=list)
    keys: list = dt.field(default_factory=list)
    yscale: str = ''
    xscale: str = ''
    ylim: list = None
    yticks: list = None
    xlim: list = None
    xticks: list = None
    x_lows: list = dt.field(default_factory=list)
    x_highs: list = dt.field(default_factory=list)




def xy(df, y_key, x_key='epoch', method=None, n_end=None, n_average=None):
    df_this = df.dropna()
    out = df_this[x_key], df_this[y_key]
    if method == 'ePIE':
        if n_average is None:
            n_average = 20
        df2 = df_this.rolling(n_average).mean().dropna()
        yout = df2[y_key] #* 1024 if y_key == 'train_loss' else df2[y_key]
        out = df2[x_key], yout
    elif n_average is not None:
        df2 = df_this.rolling(n_average).mean().dropna()
        yout = df2[y_key]
        out = df2[x_key], yout
    if n_end is not None:
        return out[0][:n_end], out[1][:n_end]
    return out

def filterDatasetsForPlots(data_all, admm_beta):
    data_all_new = []
    for d in data_all:
        if 'Adam' in str(d.init_params):
            for adam_pars in [(0, 0.1), (0, 0.1, 0.1)]:  # , (256, 0.01), (256, 0.01, 0.1)]:
                data_all_new = data_all_new + cbut.selectAdam(d, *adam_pars)
        elif ('Curveball' in str(d.init_params)) and not ('JointCurveball' in str(d.init_params)):
            if d.iterable_params['apply_precond']: continue
            # print(d.init_params, d.iterable_params['reconstruct_probe'])
            data_all_new = data_all_new + cbut.selectCurveball(d)
        elif 'ADMM' in str(d.init_params):
            if not d.iterable_params['reconstruct_probe']: continue
            if d.iterable_params['beta'] != admm_beta: continue
            data_all_new.append(deepcopy(d))
        elif 'LMA' in str(d.init_params):
            if 'stochastic_diag_estimator_type' in d.iterable_params:
                if d.iterable_params['stochastic_diag_estimator_type'] != 'martens':
                    continue
            if 'stochastic_diag_estimator_iters' in d.iterable_params:
                if d.iterable_params['stochastic_diag_estimator_iters'] != 20:
                    continue
            data_all_new.append(deepcopy(d))


        # elif 'CG' in str(d.init_params):
        #    #print(d.iterable_params)
        #    if not d.iterable_params['apply_precond']: continue
        #    data_all_new.append(deepcopy(d))
        else:
            data_all_new.append(deepcopy(d))

    return data_all_new


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def getData(key_fname_dict, reconstruct_probe=True):
    data_all = anut.combineDatasets(key_fname_dict)
    objs_all = {}

    if reconstruct_probe:
        probes_all = {}
    for d in data_all:
        if reconstruct_probe:
            if d.iterable_params['reconstruct_probe']:
                probes_all[d.iterable_params['method']] = np.array(d.final_probes)
                objs_all[d.iterable_params['method']] = np.array(d.final_objs)
        else:
            if not d.iterable_params['reconstruct_probe']:
                objs_all[d.iterable_params['method']] = np.array(d.final_objs)
    if reconstruct_probe:
        return objs_all, probes_all
    return objs_all

def _plotProbeAmpl(absvals, ax, min_level=0.2, ampl_ticks=[0.1, 1.0, 10], labelsize=15):
    new_cmap = truncate_colormap(plt.get_cmap('coolwarm'), minval=0.5, maxval=1)

    plt.sca(ax)
    ax.set_axis_off()

    cax = plt.pcolormesh(absvals, cmap=new_cmap,
                         norm=LogNorm(vmin=min_level, vmax=absvals.max()))
    cax.axes.set_axis_off()
    cb = plt.colorbar(cax, ticks=ampl_ticks, pad=0.02)
    cb.ax.tick_params(labelsize=labelsize)
    ax.set_aspect('equal')

def _plotProbePhase(angvals, ax, labelsize=15):
    plt.sca(ax)
    ax.set_axis_off()

    cax = plt.pcolormesh(angvals, cmap='coolwarm',
                         norm=MidpointNormalize(vmin=-np.pi, vmax=np.pi, midpoint=0))
    cax.axes.set_axis_off()
    cb = plt.colorbar(cax, ticks=[-np.pi, 0, np.pi], pad=0.02)
    cb.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
    cb.ax.tick_params(labelsize=labelsize)
    ax.set_aspect('equal')


def plotProbeAmplPhase(probe_wavefront, save_fname_prefix=None, ampl_level=0.2, ampl_ticks=[0.1, 1.0, 10]):
    absvals = np.abs(np.fft.fftshift(probe_wavefront))
    angvals = np.angle(np.fft.fftshift(probe_wavefront))
    absvals[absvals < ampl_level] = ampl_level

    plt.figure(figsize=[5, 4])
    ax = plt.subplot(1, 1, 1)
    _plotProbeAmpl(absvals, ax, ampl_level, ampl_ticks)
    plt.tight_layout()
    if save_fname_prefix is not None:
        plt.savefig(f'{save_fname_prefix}_probe_ampl.pdf', bbox_inches='tight')
    plt.show()

    # angvals[absvals <= 0.1] = 0
    plt.figure(figsize=[5, 4])
    ax = plt.subplot(1, 1, 1)
    _plotProbePhase(angvals, ax)
    plt.tight_layout()
    if save_fname_prefix is not None:
        plt.savefig(f'{save_fname_prefix}_probe_phase.pdf', bbox_inches='tight')
    plt.show()


def _plotObjAmpl(absvals, ax, min_level=0, ampl_ticks=[0, 0.5, 1.0], labelsize=14):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.sca(ax)
    ax.set_axis_off()
    max_level = np.max([np.max(absvals), 1.0])
    print(max_level)
    im = plt.pcolormesh(absvals, cmap='gray', vmin=min_level, vmax=max_level)
    im.axes.set_axis_off()
    ax.set_aspect('equal')

    #cb = plt.colorbar(cax, ticks=ampl_ticks, pad=0.02)
    #cb.ax.tick_params(labelsize=labelsize)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = plt.colorbar(im, cax=cax, ticks=ampl_ticks)
    cb.ax.tick_params(labelsize=labelsize)


def _plotObjPhase(angvals, ax, labelsize=14):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.sca(ax)
    ax.set_axis_off()
    im = plt.pcolormesh(angvals, cmap='gray')#, vmin=min_level, vmax=max_level)
    im.axes.set_axis_off()
    ax.set_aspect('equal')

    #cax = plt.pcolormesh(angvals, cmap='gray')  # , vmin=-np.pi, vmax=np.pi)
    #cax.axes.set_axis_off()
    #cb = plt.colorbar(cax, pad=0.02)
    # cb = plt.colorbar(cax, ticks=[-np.pi, 0, np.pi], pad=0.02)
    # cb.ax.set_yticklabels([r'$-\pi/2$', r'0', r'$\pi/2$'])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = plt.colorbar(im, cax=cax, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])

    cb.ax.set_yticklabels([r'$-\pi$', r'$-\dfrac{\pi}{2}$', r'0', r'$\dfrac{\pi}{2}$', r'$\pi$'])
    cb.ax.tick_params(labelsize=labelsize)


def plotObjAmplPhase(obj_array, save_fname_prefix=None):
    absvals = np.abs(obj_array)
    angvals = np.angle(obj_array)
    # print(scipy.stats.circmean(angvals), angvals.max(), angvals.min())
    angvals -= scipy.stats.circmean(angvals, low=-np.pi, high=np.pi)

    # absvals[absvals<0.2] = 0.2

    # new_cmap = truncate_colormap(plt.get_cmap('coolwarm'), minval=0.5, maxval=1)
    plt.figure(figsize=[5, 4])
    ax = plt.subplot(1, 1, 1)
    _plotObjAmpl(absvals, ax)
    plt.tight_layout()
    if save_fname_prefix is not None:
        plt.savefig(f'{save_fname_prefix}_obj_ampl.pdf', bbox_inches='tight')
    plt.show()

    diff = np.max(angvals) - np.pi / 2
    angvals -= diff
    # angvals[absvals <= 0.1] = 0
    plt.figure(figsize=[5, 4])
    ax = plt.subplot(1, 1, 1)
    _plotObjPhase(angvals, ax)

    plt.tight_layout()
    if save_fname_prefix is not None:
        plt.savefig(f'{save_fname_prefix}_obj_phase.pdf', bbox_inches='tight')
    plt.show()