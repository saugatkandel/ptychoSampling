import ptychoSampling.farfield.analysis_scripts.analysis_utils as anut
import dill
dill._dill._reverse_typemap['ClassType'] = type

import dataclasses as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy

def combineDatasets(key_fname_dict):
    data_all = []
    for k, v in key_fname_dict.items():
        with open(v, 'rb') as f:
            datasets = dill.load(f)
        for d in datasets:
            d.iterable_params['method'] = k
            data_all.append(d)
    return data_all


def getXFinalPerMethod(means, training_batch_size, method, convergence_window_size=100,convergence_epsilon=1e-3):
    epochs_per_registration = 5 if method == 'Adam' else 1
    if training_batch_size == 256:
        iterations_per_epoch = 4
        index_shift = 1
    elif training_batch_size == 1:
        iterations_per_epoch = 1024
        index_shift = 1
    else:
        iterations_per_epoch = 1
        index_shift = 0
    index = anut.getConvergenceIndex(means, epochs_per_registration,
                                     iterations_per_epoch, index_shift, window_size_epochs=convergence_window_size,
                                     verbose=False, epsilon=convergence_epsilon)
    return index


def plotData(means, lows, highs, suptitle,
             df_keys_ordered,
             label_key,
             obj_error_ylims=[0, 0.4],
             rfactor_ylims=[0.2, 0.7],
             xlim_max=None,
             log_yscale=False,
             log_xscale=False,
             row_key=None,
             plot_until_convergence=False,
             fill_between=True,
             fname=None,
             flops_logscale=False,
             flops_xlims=None,
             flops_xticks=None,
             obj_error_flops_ylims=None,
             plot_rfactor=False,
             plot_loss=False,
             convergence_epsilon=1e-3,
             convergence_window_size=100,
             select_row_key=None):

    colors = {'LM-A':'red', 'PLM-A':'black', 'PHeBIE': 'blue', 'NCG': 'green', 'ADMM': 'gray',
              'NAG':'purple', 'PNCG':'cyan', 'PLM-J': 'magenta', 'ePIE':'orange', 'Adam': 'black',
              'St-PLM-A': 'red', 'St-PLM-J':'magenta',
              'LM-A-S': 'red', 'PLM-A-S': 'red', 'LM-J-S': 'magenta', 'PLM-J-S': 'magenta'}
    linestyles = {'LM-A':':', 'PLM-A':'--', 'PHeBIE':'-.', 'NCG':'-', 'ADMM':':', 'NAG':'-.',
                  'PNCG': '-.', 'PLM-J': ':', 'ePIE':':', 'Adam': '--',
                  'St-PLM-A': '-.', 'St-PLM-J': ':',
                  'LM-A-S':':', 'PLM-A-S':':', 'PLM-J-S':'--', 'LM-J-S':'--'}
    markers = {'LM-A':'o', 'PLM-A': '<', 'PHeBIE':'*', 'NCG':'<', 'ADMM':'s', 'NAG':'x', 'PNCG':'*',
               'St-PLM-A': 's', 'St-PLM-J': '<',
               'PLM-J':'*', 'ePIE':'o', 'Adam': '<', 'LM-A-S':'o', 'PLM-J-S':'<', 'PLM-A-S':'o', 'PLM-J-S':'<'}
    #colors = ['red', 'gray', 'blue', 'green', 'magenta']
    #linestyles = [':', '--', '-.', '-']
    #markers = ['o', '<', '*', 's', 'x']

    mix = df_keys_ordered.index('method')
    bix = df_keys_ordered.index('training_batch_size')

    lix = df_keys_ordered.index(label_key)
    if row_key is not None:
        rix = df_keys_ordered.index(row_key)
        rows_list = []
        for k in means:
            if k[rix] not in rows_list:
                if select_row_key is not None:
                    if k[rix] != select_row_key: continue
                rows_list.append(k[rix])
        rows = len(rows_list)
    else:
        rows = 1

    #cgens = [anut.generator(colors) for r in range(rows)]
    #mgens = [anut.generator(markers) for r in range(rows)]
    #lgens = [anut.generator(linestyles) for r in range(rows)]

    cols = 2
    if plot_rfactor:
        cols += 1
    if plot_loss:
        cols += 1
    fig, axs = plt.subplots(rows, cols, figsize=[4.7 * cols, 4 * rows])
    if rows == 1:
        axs = axs[None, :]

    for key, v in means.items():
        label = str(key[lix])
        print(key)
        ix = rows_list.index(key[rix]) if rows > 1 else 0
        if select_row_key is not None:
            if key[rix] != select_row_key: continue
        #c = next(cgens[ix])
        #l = next(lgens[ix])
        #m = next(mgens[ix])
        c = colors[key[mix]]
        l = linestyles[key[mix]]
        m = markers[key[mix]]

        lows_this = lows[key]
        highs_this = highs[key]

        convergence_index = getXFinalPerMethod(v, key[bix], key[mix],
                                               convergence_epsilon=convergence_epsilon,
                                               convergence_window_size=convergence_window_size)
        if plot_until_convergence:
            v = v.loc[:convergence_index]
            lows_this = lows[key].loc[:convergence_index]
            highs_this = highs[key].loc[:convergence_index]

        x = v.epoch
        flops = v.flops.loc[:convergence_index]
        #if key[mix] == 'CB' and key[bix] == 256:
        #    if v.epoch.iloc[-1] > 1000:
        #        x = x / 2
        #        flops = flops / 2
        if fill_between:
            axs[ix, 0].fill_between(x, lows_this.obj_error, highs_this.obj_error,
                                    color=c, alpha=0.1)
        axs[ix, 0].plot(x, v.obj_error, color=c, ls=l, marker=m,
                        label=label, markevery=0.2)
        axs[ix, 0].set_ylim(obj_error_ylims)
        axs[ix, 0].set_xlim(left=0)
        axs[ix, 0].set_xlabel('Epochs', fontsize=17)
        axs[ix, 0].set_ylabel('Recons. Err.', fontsize=17)
        if log_yscale:
            axs[ix, 0].set_yscale('log')
        if log_xscale:
            axs[ix, 0].set_xscale('log')
        if xlim_max is not None:
            axs[ix, 0].set_xlim(right=xlim_max)

        axs[ix, 1].plot(flops, v.obj_error.loc[:convergence_index],
                                color=c, ls=l, marker=m,
                                label=label, markevery=0.2)
        if flops_logscale:
            axs[ix, 1].set_xscale('log')
        if obj_error_flops_ylims is not None:
            axs[ix, 1].set_ylim(obj_error_flops_ylims)
        if flops_xlims is not None:
            axs[ix, 1].set_xlim(flops_xlims)
        if flops_xticks is not None:
            axs[ix, 1].set_xticks(flops_xticks)
            axs[ix, 1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        if log_yscale:
            axs[ix, 1].set_yscale('log')
        axs[ix, 1].set_ylabel('Recons. Err.', fontsize=17)
        axs[ix, 1].set_xlabel('Flops', fontsize=17)

        col_index = 1
        if plot_rfactor:
            col_index += 1
            if fill_between:
                axs[ix, col_index].fill_between(x, lows_this.r_factor, highs_this.r_factor,
                                        color=c, alpha=0.1)
            axs[ix, col_index].plot(x, v.r_factor, color=c, ls=l, marker=m,
                            label=label, markevery=0.2)
            axs[ix, col_index].set_ylim(rfactor_ylims)
            axs[ix, col_index].set_xlim(left=0)
            axs[ix, col_index].set_xlabel('Epochs')
            axs[ix, col_index].set_ylabel(r'$R_f$', fontsize=17)
            if log_yscale:
                axs[ix, col_index].set_yscale('log')
            if log_xscale:
                axs[ix, col_index].set_xscale('log')
            if xlim_max is not None:
                axs[ix, col_index].set_xlim(right=xlim_max)

        col_index += 1

        if plot_loss:
            if fill_between:
                axs[ix, col_index].fill_between(x, lows_this.train_loss, highs_this.train_loss,
                                        color=c, alpha=0.1)
            axs[ix, col_index].plot(x, v.train_loss, color=c, ls=l, marker=m,
                            label=label, markevery=0.2)
            #axs[ix, col_index].set_ylim(rfactor_ylims)
            axs[ix, col_index].set_xlim(left=0)
            axs[ix, col_index].set_xlabel('Epochs')
            axs[ix, col_index].set_ylabel(r'Loss', fontsize=17)
            if log_yscale:
                axs[ix, col_index].set_yscale('log')
            if log_xscale:
                axs[ix,col_index].set_xscale('log')
            if xlim_max is not None:
                axs[ix, col_index].set_xlim(right=xlim_max)


    axs[0, 1].legend(loc='best')  # bbox_to_anchor=(0.75, 1.2), ncol=3, labelspacing=0.3)

    if row_key is not None:
        for i, r in enumerate(rows_list):
            plt.figtext(.5, 1.0 / (i + 1), f'{row_key}={r}', fontsize=15, ha='center')
            if i + 1 >= rows:
                continue
            axs[i + 1, 1].legend(loc='best')  # bbox_to_anchor=(0.75, 1.2), ncol=3, labelspacing=0.3)
    # plt.figtext(.5, 0.5, r'$b=256$', fontsize=15, ha='center')

    plt.suptitle(suptitle, fontsize=17, x=0.5, y=1.1, ha='center')
    plt.tight_layout(h_pad=3.0)
    if fname is not None and (row_key is not None):
        for n, row in enumerate(rows_list):
            extent = axs[n, 0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'error_{row}_{fname}', bbox_inches=extent)

            extent = axs[n, 1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'flops_{row}_{fname}', bbox_inches=extent)

        if select_row_key is not None:
            fname = f'{select_row_key}_{fname}'
        plt.savefig(fname, bbox_inches='tight')

    plt.show()


def selectAdam(dataset, batch_size, lr_obj, lr_probe=None):
    pars = deepcopy(dataset.iterable_params)
    if pars['training_batch_size'] != batch_size:
        return []
    if pars['learning_rate_obj'] != lr_obj:
        return []

    if bool(lr_probe is not None) != bool('learning_rate_probe' in pars):
        return []
    if lr_probe is not None:
        if pars['learning_rate_probe'] != lr_probe:
            return []
    return [deepcopy(dataset)]

def selectCurveball(dataset):
    pars = deepcopy(dataset.iterable_params)
    if pars['reconstruct_probe']:
        if pars['training_batch_size'] == 0:
            if not pars['method'] == 'cb_seq':
                return []
            pars['method'] = 'CB'
        else:
            if not pars['method'] == 'cb_alt':
                return []
            pars['method'] = 'CB'
    pars['method'] = 'CB' #if pars['training_batch_size'] == 0 else 'CB-M'
    dataset = dt.replace(deepcopy(dataset), iterable_params=pars)
    return [dataset]