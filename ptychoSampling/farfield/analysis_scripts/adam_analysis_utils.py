import ptychoSampling.farfield.analysis_scripts.analysis_utils as alut
import matplotlib.pyplot as plt

def filterDatasets(means, lows, highs, limit=0.3):
    means_filt = {}
    lows_filt = {}
    highs_filt = {}
    for key, v in means.items():
        if v.obj_error.iat[-1] < limit:
            means_filt[key] = v
            lows_filt[key] = lows[key]
            highs_filt[key] = highs[key]
    return means_filt, lows_filt, highs_filt

def plotAdamData(means, lows, highs, suptitle,
                 obj_error_ylim=[0, 0.2],
                 rfactor_ylim=[0, 0.2],
                 logscale=False,
                 fname=None):
    colors = ['red', 'black', 'blue', 'green', 'purple', 'orange']
    linestyles = [':', '--', '-.', '-']
    markers = ['o', '<', '*', 's', 'X']

    plot_rows = {0: 0, 256: 1}
    cgens = {k: alut.generator(colors) for k in plot_rows}
    lgens = {k: alut.generator(linestyles) for k in plot_rows}
    mgens = {k: alut.generator(markers) for k in plot_rows}

    fig, axs = plt.subplots(len(plot_rows), 2, figsize=[9, 4 * len(plot_rows)])

    for key, v in means.items():
        k1 = key[0]
        label = str(key[1])
        if len(key) == 3:
            label += f'_{key[2]}'
        c = next(cgens[k1])
        l = next(lgens[k1])
        m = next(mgens[k1])

        axs[plot_rows[k1], 0].fill_between(v.epoch, lows[key].obj_error, highs[key].obj_error,
                                           color=c, alpha=0.1)
        axs[plot_rows[k1], 0].plot(v.epoch, v.obj_error, color=c, ls=l, marker=m,
                                           label=label, markevery=40)
        if logscale:
            axs[plot_rows[k1], 0].set_yscale('log', nonposy='clip')
        else:
            axs[plot_rows[k1], 0].set_ylim(obj_error_ylim)

        axs[plot_rows[k1], 1].fill_between(v.epoch, lows[key].r_factor, highs[key].r_factor,
                                           color=c, alpha=0.1)
        axs[plot_rows[k1], 1].plot(v.epoch, v.r_factor, color=c, ls=l, marker=m,
                                           label=label, markevery=40)
        if logscale:
            axs[plot_rows[k1], 1].set_yscale('log', nonposy='clip')
        else:
            axs[plot_rows[k1], 1].set_ylim(rfactor_ylim)

        axs[plot_rows[k1], 0].set_ylabel('Recons. Err.', fontsize=17)
        axs[plot_rows[k1], 1].set_ylabel(r'$R_f$', fontsize=17)

    axs[0, 1].legend(loc='best')#bbox_to_anchor=(0.75, 1.2), ncol=3, labelspacing=0.3)
    plt.figtext(.5, 1.0, r'$b=0$', fontsize=15, ha='center')

    axs[1, 1].legend(loc='best')#bbox_to_anchor=(0.75, 1.2), ncol=3, labelspacing=0.3)
    plt.figtext(.5, 0.5, r'$b=256$', fontsize=15, ha='center')

    plt.suptitle(suptitle, fontsize=17, x=0.5, y=1.1, ha='center')
    plt.tight_layout(h_pad=3.0)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()
